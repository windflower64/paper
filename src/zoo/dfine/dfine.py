"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from ...core import register
from .sam_mask_aggregation import SAMMaskAggregation, region_loss
from .sam_support_shape import SupportShapeAggregation, support_shape_loss
from .sam_group_contrast import group_loss, supervision_scale
from .sam_scale_supervision import rescue_loss
from .stql_qcer import (
    QueryConditionedEvidenceReader,
    SharedQueryProjection,
    STQLPixelProjection,
)
from .qdmf import (
    QueryConditionedDynamicModalityFusion,
    qdmf_basic_statistics,
)
from .sam_query_evidence_reader import SAMQueryEvidenceReader
from .target_evidence_thermal_fusion import TargetEvidenceThermalFusion

__all__ = [
    "DFINE",
]


class AdaptiveBackgroundSmoothing(nn.Module):
    """SET Eq.3 channel-reduce/expand background smoothing operation."""

    def __init__(self, channels, kernel_size, reduction=4):
        super().__init__()
        hidden_channels = max(1, int(channels) // int(reduction))
        padding = int(kernel_size) // 2
        self.reduce = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.expand = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.activation = nn.ReLU(inplace=False)

    def forward(self, background):
        smoothed = self.activation(self.reduce(background))
        smoothed = self.activation(self.expand(smoothed))
        return background + smoothed


class HierarchicalBackgroundSmoothing(nn.Module):
    """Training-only SET HBS adapted to D-FINE encoder feature levels."""

    def __init__(self, channels, strides, reduction=4, strict_background=True):
        super().__init__()
        if len(channels) != len(strides):
            raise ValueError("HBS channels and strides must have equal length")
        self.channels = [int(value) for value in channels]
        self.strides = [int(value) for value in strides]
        self.strict_background = bool(strict_background)
        self.kernel_sizes = [
            int(math.ceil(math.log2(stride) / 2.0) * 2 + 1)
            for stride in self.strides
        ]
        self.operations = nn.ModuleList(
            AdaptiveBackgroundSmoothing(channel, kernel, reduction)
            for channel, kernel in zip(self.channels, self.kernel_sizes)
        )
        self.last_masks = None
        self.last_residuals = None

    @staticmethod
    def _feature_mask(feature, targets):
        batch, _, height, width = feature.shape
        mask = feature.new_zeros((batch, 1, height, width))
        for batch_index, target in enumerate(targets):
            for center_x, center_y, box_width, box_height in target["boxes"]:
                left = int(torch.floor((center_x - box_width / 2) * width).item())
                top = int(torch.floor((center_y - box_height / 2) * height).item())
                right = int(torch.ceil((center_x + box_width / 2) * width).item())
                bottom = int(torch.ceil((center_y + box_height / 2) * height).item())
                left = max(0, min(width - 1, left))
                top = max(0, min(height - 1, top))
                right = max(left + 1, min(width, right))
                bottom = max(top + 1, min(height, bottom))
                mask[batch_index, 0, top:bottom, left:right] = 1.0
        return mask

    def forward(self, features, targets):
        if len(features) != len(self.operations):
            raise ValueError(
                f"HBS expected {len(self.operations)} levels, got {len(features)}"
            )
        enhanced_features = []
        masks = []
        residuals = []
        for feature, expected_channels, operation in zip(
            features, self.channels, self.operations
        ):
            if feature.shape[1] != expected_channels:
                raise ValueError(
                    f"HBS expected {expected_channels} channels, got {feature.shape[1]}"
                )
            mask = self._feature_mask(feature, targets)
            foreground = feature * mask
            background = feature * (1.0 - mask)
            smoothed_background = operation(background)
            if self.strict_background:
                # The source Eq.3 convolution can spill background responses
                # back into tiny foreground cells.  Reapply the complement
                # mask to enforce the stated foreground-preservation intent.
                smoothed_background = smoothed_background * (1.0 - mask)
            enhanced = foreground + smoothed_background
            enhanced_features.append(enhanced)
            masks.append(mask.detach())
            residuals.append((enhanced - feature).detach())
        self.last_masks = masks
        self.last_residuals = residuals
        return enhanced_features


class TargetGuidedFrequencyLocalizationResidual(nn.Module):
    """Read directional S8 high frequency only to refine final boxes.

    The pretrained detector, standard S8->S16 downsample, classification
    logits, and all auxiliary decoder boxes remain untouched. Predicted boxes
    provide the target location: three points are sampled on each of the four
    box sides from fixed Haar directional bands. A zero-initialized, bounded
    logit-space residual adjusts only the final ``pred_boxes`` tensor.

    ``frequency_mode`` is evaluation-only causal instrumentation.
    """

    VALID_FREQUENCY_MODES = {"full", "zero", "shifted", "swap_direction"}

    def __init__(
        self,
        in_channels=512,
        edge_channels=16,
        hidden_dim=64,
        samples_per_side=3,
        max_logit_delta=0.25,
    ):
        super().__init__()
        if samples_per_side != 3:
            raise ValueError("TFLR currently preregisters exactly 3 samples per box side")
        self.in_channels = int(in_channels)
        self.edge_channels = int(edge_channels)
        self.samples_per_side = int(samples_per_side)
        self.max_logit_delta = float(max_logit_delta)
        self.frequency_mode = "full"

        # A 1x1 projection commutes with the fixed Haar transform, so reducing
        # channels before DWT preserves its spatial-frequency meaning.
        self.channel_projection = nn.Conv2d(
            self.in_channels, self.edge_channels, kernel_size=1, bias=False
        )
        side_dim = 2 * self.edge_channels
        self.regressor = nn.Sequential(
            nn.Linear(4 * side_dim, hidden_dim, bias=False),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, 4, bias=False),
        )
        nn.init.zeros_(self.regressor[-1].weight)

        self.last_residual = None
        self.last_residual_abs_mean = None
        self.last_residual_abs_max = None

    @staticmethod
    def _haar_high_frequency(feature):
        if feature.shape[-2] % 2 or feature.shape[-1] % 2:
            raise ValueError(
                "TFLR requires even S8 height/width, "
                f"got {tuple(feature.shape[-2:])}"
            )
        x00 = feature[..., 0::2, 0::2]
        x01 = feature[..., 0::2, 1::2]
        x10 = feature[..., 1::2, 0::2]
        x11 = feature[..., 1::2, 1::2]
        lh = (-x00 + x01 - x10 + x11) * 0.5
        hl = (-x00 - x01 + x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return lh, hl, hh

    def _intervene(self, bands):
        lh, hl, hh = bands
        if self.frequency_mode == "full":
            return lh, hl, hh
        if self.frequency_mode == "zero":
            return torch.zeros_like(lh), torch.zeros_like(hl), torch.zeros_like(hh)
        if self.frequency_mode == "shifted":
            shift = (max(1, lh.shape[-2] // 2), max(1, lh.shape[-1] // 2))
            return tuple(
                torch.roll(band, shifts=shift, dims=(-2, -1))
                for band in (lh, hl, hh)
            )
        if self.frequency_mode == "swap_direction":
            return hl, lh, hh
        raise ValueError(
            f"unsupported TFLR frequency_mode {self.frequency_mode!r}; "
            f"expected one of {sorted(self.VALID_FREQUENCY_MODES)}"
        )

    @staticmethod
    def _side_grid(boxes, side):
        center_x, center_y, width, height = boxes.unbind(-1)
        offsets = boxes.new_tensor((-0.5, 0.0, 0.5))
        if side == "left":
            x = (center_x - width / 2).unsqueeze(-1).expand(-1, -1, 3)
            y = center_y.unsqueeze(-1) + height.unsqueeze(-1) * offsets
        elif side == "right":
            x = (center_x + width / 2).unsqueeze(-1).expand(-1, -1, 3)
            y = center_y.unsqueeze(-1) + height.unsqueeze(-1) * offsets
        elif side == "top":
            x = center_x.unsqueeze(-1) + width.unsqueeze(-1) * offsets
            y = (center_y - height / 2).unsqueeze(-1).expand(-1, -1, 3)
        elif side == "bottom":
            x = center_x.unsqueeze(-1) + width.unsqueeze(-1) * offsets
            y = (center_y + height / 2).unsqueeze(-1).expand(-1, -1, 3)
        else:
            raise ValueError(f"unsupported side: {side}")
        return torch.stack((x, y), dim=-1).clamp(0.0, 1.0).mul(2.0).sub(1.0)

    @staticmethod
    def _sample_side(feature, boxes, side):
        grid = TargetGuidedFrequencyLocalizationResidual._side_grid(boxes, side)
        sampled = F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.mean(dim=-1).transpose(1, 2)

    @staticmethod
    def _inverse_sigmoid(value, eps=1e-5):
        value = value.clamp(0.0, 1.0)
        return torch.log(value.clamp_min(eps) / (1.0 - value).clamp_min(eps))

    def forward(self, s8_feature, boxes, logits):
        projected = self.channel_projection(s8_feature)
        lh, hl, hh = self._intervene(self._haar_high_frequency(projected))
        # LH responds to left/right changes; HL responds to top/bottom changes.
        # HH is retained in both families as diagonal boundary evidence.
        x_edges = torch.cat((lh, hh), dim=1)
        y_edges = torch.cat((hl, hh), dim=1)
        side_features = torch.cat(
            (
                self._sample_side(x_edges, boxes, "left"),
                self._sample_side(x_edges, boxes, "right"),
                self._sample_side(y_edges, boxes, "top"),
                self._sample_side(y_edges, boxes, "bottom"),
            ),
            dim=-1,
        )
        raw_delta = self.regressor(side_features)
        confidence = logits.detach().sigmoid().amax(dim=-1, keepdim=True)
        residual = self.max_logit_delta * confidence * torch.tanh(raw_delta)
        base_logits = self._inverse_sigmoid(boxes)
        # Subtract the numerically reconstructed zero-residual output before
        # adding the correction. This is bit-exact at initialization while
        # retaining the sigmoid/logit derivative for the zero-initialized head.
        refined = boxes + (
            (base_logits + residual).sigmoid() - base_logits.sigmoid()
        )
        self.last_residual = residual.detach()
        self.last_residual_abs_mean = residual.detach().abs().mean()
        self.last_residual_abs_max = residual.detach().abs().max()
        return refined


class SpatiallyDecoupledThermalConditioner(nn.Module):
    """Condition RGB channels with permutation-invariant thermal statistics.

    The thermal CNN/encoder still extracts local appearance.  This interface
    deliberately removes every spatial coordinate before cross-modal mixing:
    mean, standard deviation and generalized-mean statistics are projected by
    MLPs, then used for one bounded channel-wise update of RGB encoder memory.
    """

    def __init__(
        self,
        hidden_dim=128,
        num_levels=2,
        bottleneck_dim=64,
        max_rms_ratio=0.03,
        eps=1e-6,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_levels = int(num_levels)
        self.bottleneck_dim = int(bottleneck_dim)
        self.max_rms_ratio = float(max_rms_ratio)
        self.eps = float(eps)
        if self.hidden_dim <= 0 or self.num_levels <= 0:
            raise ValueError("M-SD2 hidden_dim and num_levels must be positive")
        if self.bottleneck_dim <= 0:
            raise ValueError("M-SD2 bottleneck_dim must be positive")
        if not 0.0 < self.max_rms_ratio <= 0.10:
            raise ValueError("M-SD2 max_rms_ratio must be in (0, 0.10]")

        statistics_dim = 3 * self.hidden_dim
        self.thermal_level_projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(statistics_dim, self.bottleneck_dim),
                nn.LayerNorm(self.bottleneck_dim),
                nn.SiLU(inplace=False),
            )
            for _ in range(self.num_levels)
        )
        self.thermal_context = nn.Sequential(
            nn.Linear(
                self.num_levels * self.bottleneck_dim,
                self.bottleneck_dim,
            ),
            nn.LayerNorm(self.bottleneck_dim),
            nn.SiLU(inplace=False),
        )
        self.visible_level_projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(statistics_dim, self.bottleneck_dim),
                nn.LayerNorm(self.bottleneck_dim),
                nn.SiLU(inplace=False),
            )
            for _ in range(self.num_levels)
        )
        interaction_dim = 4 * self.bottleneck_dim
        self.channel_generators = nn.ModuleList(
            nn.Sequential(
                nn.Linear(interaction_dim, 2 * self.bottleneck_dim),
                nn.LayerNorm(2 * self.bottleneck_dim),
                nn.SiLU(inplace=False),
                nn.Linear(2 * self.bottleneck_dim, 2 * self.hidden_dim),
            )
            for _ in range(self.num_levels)
        )
        self.residual_scales = nn.Parameter(torch.zeros(self.num_levels))

        for generator in self.channel_generators:
            nn.init.normal_(generator[-1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(generator[-1].bias)

        self.last_scale_by_level = None
        self.last_rms_ratio_by_level = None
        self.last_content_ratio = None
        self.last_context_abs_mean = None

    def _statistics(self, feature):
        if feature.ndim != 4 or feature.shape[1] != self.hidden_dim:
            raise ValueError(
                "M-SD2 expected [B, hidden_dim, H, W], got "
                f"{tuple(feature.shape)}"
            )
        flattened = feature.flatten(2).float()
        mean = flattened.mean(dim=-1)
        std = flattened.var(dim=-1, unbiased=False).add(self.eps).sqrt()
        generalized_mean = (
            flattened.abs().clamp_min(self.eps).pow(3.0).mean(dim=-1)
        ).pow(1.0 / 3.0)
        return torch.cat((mean, std, generalized_mean), dim=-1).to(feature.dtype)

    def forward(self, visible_features, thermal_features, content_mask=None):
        if not isinstance(visible_features, (list, tuple)) or not isinstance(
            thermal_features, (list, tuple)
        ):
            raise ValueError("M-SD2 requires visible and thermal feature lists")
        if len(visible_features) != self.num_levels or len(thermal_features) != self.num_levels:
            raise ValueError(
                "M-SD2 feature level mismatch: "
                f"visible={len(visible_features)} thermal={len(thermal_features)} "
                f"expected={self.num_levels}"
            )

        thermal_levels = [
            projection(self._statistics(feature))
            for feature, projection in zip(
                thermal_features, self.thermal_level_projections
            )
        ]
        thermal_context = self.thermal_context(torch.cat(thermal_levels, dim=-1))
        if content_mask is None:
            content_mask = torch.ones(
                thermal_context.shape[0],
                device=thermal_context.device,
                dtype=torch.bool,
            )
        content_mask = content_mask.to(
            device=thermal_context.device, dtype=thermal_context.dtype
        ).view(-1, 1, 1, 1)

        conditioned = []
        realized_ratios = []
        realized_scales = []
        for level, (visible, projection, generator) in enumerate(
            zip(
                visible_features,
                self.visible_level_projections,
                self.channel_generators,
            )
        ):
            visible_context = projection(self._statistics(visible))
            interaction = torch.cat(
                (
                    visible_context,
                    thermal_context,
                    visible_context * thermal_context,
                    (visible_context - thermal_context).abs(),
                ),
                dim=-1,
            )
            channel_parameters = generator(interaction)
            gamma, beta = channel_parameters.chunk(2, dim=-1)
            normalized_visible = F.group_norm(visible, num_groups=1)
            residual = (
                normalized_visible * torch.tanh(gamma)[:, :, None, None]
                + torch.tanh(beta)[:, :, None, None]
            )
            visible_rms = visible.float().square().mean(
                dim=(1, 2, 3), keepdim=True
            ).add(self.eps).sqrt()
            residual_rms = residual.float().square().mean(
                dim=(1, 2, 3), keepdim=True
            ).add(self.eps).sqrt()
            residual = residual * (visible_rms / residual_rms).to(residual.dtype)
            scale = self.max_rms_ratio * torch.tanh(self.residual_scales[level])
            update = content_mask * scale * residual
            output = visible + update
            conditioned.append(output)

            realized_ratio = (
                update.float().square().mean(dim=(1, 2, 3)).sqrt()
                / visible.float().square().mean(dim=(1, 2, 3)).add(self.eps).sqrt()
            )
            realized_ratios.append(realized_ratio.detach())
            realized_scales.append(scale.detach())

        self.last_scale_by_level = torch.stack(realized_scales)
        self.last_rms_ratio_by_level = torch.stack(realized_ratios, dim=-1)
        self.last_content_ratio = content_mask.detach().mean()
        self.last_context_abs_mean = thermal_context.detach().abs().mean()
        return conditioned


class SpatiallyDecoupledThermalTokenConditioner(nn.Module):
    """Use coordinate-free thermal set tokens to condition RGB channels.

    Learned semantic queries pool thermal feature *sets* without positional
    encodings.  RGB locations then select from the pooled content tokens using
    their own appearance.  Consequently RGB decides where a correction is
    useful, while thermal content only decides which channels to adjust.
    """

    def __init__(
        self,
        hidden_dim=128,
        num_levels=2,
        bottleneck_dim=64,
        num_tokens=4,
        num_heads=4,
        max_rms_ratio=0.03,
        thermal_contrastive=False,
        zero_anchored=False,
        zero_anchored_levels=None,
        eps=1e-6,
        final_residual_scale=1.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_levels = int(num_levels)
        self.bottleneck_dim = int(bottleneck_dim)
        self.num_tokens = int(num_tokens)
        self.num_heads = int(num_heads)
        self.max_rms_ratio = float(max_rms_ratio)
        self.thermal_contrastive = bool(thermal_contrastive)
        self.final_residual_scale = float(final_residual_scale)
        if not 0.0 <= self.final_residual_scale <= 1.0:
            raise ValueError("final_residual_scale must be between 0 and 1")
        self.zero_anchored = bool(zero_anchored)
        if zero_anchored_levels is None:
            zero_anchored_levels = ()
        self.zero_anchored_levels = tuple(
            sorted({int(level) for level in zero_anchored_levels})
        )
        if any(
            level < 0 or level >= self.num_levels
            for level in self.zero_anchored_levels
        ):
            raise ValueError(
                "M-SD2 zero_anchored_levels must be valid feature levels, "
                f"got {self.zero_anchored_levels} for {self.num_levels} levels"
            )
        if self.zero_anchored and self.zero_anchored_levels:
            raise ValueError(
                "M-SD2 zero_anchored=True already covers all levels; do not "
                "also set zero_anchored_levels"
            )
        if self.thermal_contrastive and self.zero_anchored:
            raise ValueError(
                "M-SD2 thermal_contrastive and zero_anchored are mutually "
                "exclusive residual definitions"
            )
        self._zero_anchored_by_level = tuple(
            self.zero_anchored or level in self.zero_anchored_levels
            for level in range(self.num_levels)
        )
        self.eps = float(eps)
        if self.hidden_dim <= 0 or self.num_levels <= 0:
            raise ValueError("M-SD2.1 dimensions must be positive")
        if self.bottleneck_dim <= 0 or self.num_tokens <= 0:
            raise ValueError("M-SD2.1 token dimensions must be positive")
        if self.bottleneck_dim % self.num_heads != 0:
            raise ValueError("M-SD2.1 bottleneck_dim must divide num_heads")
        if not 0.0 < self.max_rms_ratio <= 0.10:
            raise ValueError("M-SD2.1 max_rms_ratio must be in (0, 0.10]")

        self.semantic_queries = nn.Parameter(
            torch.empty(
                self.num_levels,
                self.num_tokens,
                self.bottleneck_dim,
            )
        )
        nn.init.trunc_normal_(self.semantic_queries, std=0.02)
        self.thermal_projections = nn.ModuleList(
            nn.Linear(self.hidden_dim, self.bottleneck_dim)
            for _ in range(self.num_levels)
        )
        self.thermal_poolers = nn.ModuleList(
            nn.MultiheadAttention(
                self.bottleneck_dim,
                self.num_heads,
                batch_first=True,
            )
            for _ in range(self.num_levels)
        )
        self.thermal_norms = nn.ModuleList(
            nn.LayerNorm(self.bottleneck_dim)
            for _ in range(self.num_levels)
        )
        self.visible_projections = nn.ModuleList(
            nn.Linear(self.hidden_dim, self.bottleneck_dim)
            for _ in range(self.num_levels)
        )
        self.visible_readers = nn.ModuleList(
            nn.MultiheadAttention(
                self.bottleneck_dim,
                self.num_heads,
                batch_first=True,
                bias=not self._zero_anchored_by_level[level],
            )
            for level in range(self.num_levels)
        )
        interaction_dim = 4 * self.bottleneck_dim
        self.channel_generators = nn.ModuleList()
        for zero_anchored_level in self._zero_anchored_by_level:
            if zero_anchored_level:
                # Every interaction term and every generator layer maps an
                # empty thermal context to exact zero. RGB appearance still
                # controls where evidence is useful through multiplicative
                # and distance-difference terms.
                generator = nn.Sequential(
                    nn.Linear(
                        interaction_dim,
                        2 * self.bottleneck_dim,
                        bias=False,
                    ),
                    nn.LayerNorm(
                        2 * self.bottleneck_dim,
                        elementwise_affine=False,
                    ),
                    nn.SiLU(inplace=False),
                    nn.Linear(
                        2 * self.bottleneck_dim,
                        2 * self.hidden_dim,
                        bias=False,
                    ),
                )
            else:
                generator = nn.Sequential(
                    nn.Linear(interaction_dim, 2 * self.bottleneck_dim),
                    nn.LayerNorm(2 * self.bottleneck_dim),
                    nn.SiLU(inplace=False),
                    nn.Linear(2 * self.bottleneck_dim, 2 * self.hidden_dim),
                )
            self.channel_generators.append(generator)
        # Exact identity at construction without a second near-zero scalar.
        # The output projection learns on step one; all upstream token layers
        # receive gradients as soon as that projection opens.
        for generator in self.channel_generators:
            nn.init.zeros_(generator[-1].weight)
            if generator[-1].bias is not None:
                nn.init.zeros_(generator[-1].bias)

        self.last_scale_by_level = None
        self.last_rms_ratio_by_level = None
        self.last_content_ratio = None
        self.last_context_abs_mean = None
        self.last_token_diversity = None
        self.last_null_raw_rms_by_level = None
        self.last_contrast_raw_rms_by_level = None
        # Evaluation-only decomposition controls.  Defaults preserve the
        # trained model exactly and add no checkpoint state.
        self.intervention_level_mode = "both"
        self.intervention_token_mode = "learned"

    def _thermal_tokens(self, thermal_features):
        pooled_levels = []
        batch = thermal_features[0].shape[0]
        for level, (feature, projection, pooler, norm) in enumerate(
            zip(
                thermal_features,
                self.thermal_projections,
                self.thermal_poolers,
                self.thermal_norms,
            )
        ):
            if feature.ndim != 4 or feature.shape[1] != self.hidden_dim:
                raise ValueError(
                    "M-SD2.1 expected thermal [B, C, H, W], got "
                    f"{tuple(feature.shape)}"
                )
            # FP16 attention reductions change slightly when the same set is
            # traversed in a different order.  Keep this small interface in
            # FP32 so the coordinate-free contract remains numerically stable
            # before the detector's discrete top-k query selection.
            with torch.autocast(
                device_type=feature.device.type,
                enabled=False,
            ):
                thermal_set = projection(
                    feature.float().flatten(2).transpose(1, 2)
                )
                queries = self.semantic_queries[level].unsqueeze(0).expand(
                    batch, -1, -1
                )
                pooled, _ = pooler(
                    queries,
                    thermal_set,
                    thermal_set,
                    need_weights=False,
                )
                pooled_levels.append(norm(pooled + queries))
        return torch.cat(pooled_levels, dim=1)

    def forward(self, visible_features, thermal_features, content_mask=None):
        if not isinstance(visible_features, (list, tuple)) or not isinstance(
            thermal_features, (list, tuple)
        ):
            raise ValueError("M-SD2.1 requires visible and thermal feature lists")
        if len(visible_features) != self.num_levels or len(thermal_features) != self.num_levels:
            raise ValueError(
                "M-SD2.1 feature level mismatch: "
                f"visible={len(visible_features)} thermal={len(thermal_features)} "
                f"expected={self.num_levels}"
            )

        thermal_tokens = self._thermal_tokens(thermal_features)
        if self.intervention_token_mode == "mean_repeat":
            thermal_tokens = thermal_tokens.mean(dim=1, keepdim=True).expand_as(
                thermal_tokens
            )
        elif self.intervention_token_mode == "zero":
            thermal_tokens = torch.zeros_like(thermal_tokens)
        elif self.intervention_token_mode != "learned":
            raise ValueError(
                "unknown M-SD2.1 token intervention: "
                f"{self.intervention_token_mode!r}"
            )
        if content_mask is None:
            content_mask = torch.ones(
                thermal_tokens.shape[0],
                device=thermal_tokens.device,
                dtype=torch.bool,
            )
        content_mask = content_mask.to(
            device=thermal_tokens.device,
            dtype=thermal_tokens.dtype,
        ).view(-1, 1, 1, 1)

        normalized_tokens = F.normalize(thermal_tokens.float(), dim=-1)
        similarities = normalized_tokens @ normalized_tokens.transpose(-1, -2)
        token_count = similarities.shape[-1]
        if token_count > 1:
            off_diagonal = (
                similarities.sum(dim=(-1, -2)) - token_count
            ) / (token_count * (token_count - 1))
            token_diversity = 1.0 - off_diagonal.mean()
        else:
            token_diversity = similarities.new_zeros(())

        conditioned = []
        realized_ratios = []
        context_means = []
        null_raw_ratios = []
        contrast_raw_ratios = []
        for level, (visible, projection, reader, generator) in enumerate(
            zip(
                visible_features,
                self.visible_projections,
                self.visible_readers,
                self.channel_generators,
            )
        ):
            if visible.ndim != 4 or visible.shape[1] != self.hidden_dim:
                raise ValueError(
                    "M-SD2.1 expected visible [B, C, H, W], got "
                    f"{tuple(visible.shape)}"
                )
            batch, channels, height, width = visible.shape
            with torch.autocast(
                device_type=visible.device.type,
                enabled=False,
            ):
                visible_float = visible.float()
                visible_queries = projection(
                    visible_float.flatten(2).transpose(1, 2)
                )
                thermal_context, _ = reader(
                    visible_queries,
                    thermal_tokens,
                    thermal_tokens,
                    need_weights=False,
                )
                zero_anchored_level = self._zero_anchored_by_level[level]
                if zero_anchored_level:
                    interaction = torch.cat(
                        (
                            thermal_context,
                            visible_queries * thermal_context,
                            (visible_queries - thermal_context).abs()
                            - visible_queries.abs(),
                            thermal_context.square(),
                        ),
                        dim=-1,
                    )
                else:
                    interaction = torch.cat(
                        (
                            visible_queries,
                            thermal_context,
                            visible_queries * thermal_context,
                            (visible_queries - thermal_context).abs(),
                        ),
                        dim=-1,
                    )
                gamma, beta = generator(interaction).chunk(2, dim=-1)
                gamma = gamma.transpose(1, 2).reshape(
                    batch, channels, height, width
                )
                beta = beta.transpose(1, 2).reshape(
                    batch, channels, height, width
                )
                normalized_visible = F.group_norm(
                    visible_float, num_groups=1
                )
                full_raw_update = (
                    normalized_visible * torch.tanh(gamma)
                    + torch.tanh(beta)
                )

                if self.thermal_contrastive and not zero_anchored_level:
                    null_tokens = torch.zeros_like(thermal_tokens)
                    null_context, _ = reader(
                        visible_queries,
                        null_tokens,
                        null_tokens,
                        need_weights=False,
                    )
                    null_interaction = torch.cat(
                        (
                            visible_queries,
                            null_context,
                            visible_queries * null_context,
                            (visible_queries - null_context).abs(),
                        ),
                        dim=-1,
                    )
                    null_gamma, null_beta = generator(null_interaction).chunk(
                        2, dim=-1
                    )
                    null_gamma = null_gamma.transpose(1, 2).reshape(
                        batch, channels, height, width
                    )
                    null_beta = null_beta.transpose(1, 2).reshape(
                        batch, channels, height, width
                    )
                    null_raw_update = (
                        normalized_visible * torch.tanh(null_gamma)
                        + torch.tanh(null_beta)
                    )
                    raw_update = full_raw_update - null_raw_update
                else:
                    null_raw_update = torch.zeros_like(full_raw_update)
                    raw_update = full_raw_update

                visible_rms = visible_float.square().mean(
                    dim=(1, 2, 3), keepdim=True
                ).add(self.eps).sqrt()
                raw_mean_square = raw_update.square().mean(
                    dim=(1, 2, 3), keepdim=True
                )
                rms_cap = self.max_rms_ratio * visible_rms
                # Approximately identity for small updates and asymptotically
                # bounded by rms_cap for large updates.  Unlike hard clipping,
                # the limiter remains smooth at the transition.
                limiter = rms_cap / torch.sqrt(
                    raw_mean_square + rms_cap.square() + self.eps
                )
                update = content_mask * raw_update * limiter
                if self.intervention_level_mode == "s16_only" and level != 0:
                    update = torch.zeros_like(update)
                elif self.intervention_level_mode == "s32_only" and level != 1:
                    update = torch.zeros_like(update)
                elif self.intervention_level_mode not in (
                    "both",
                    "s16_only",
                    "s32_only",
                ):
                    raise ValueError(
                        "unknown M-SD2.1 level intervention: "
                        f"{self.intervention_level_mode!r}"
                    )
                output = visible_float + update
            output = output.to(visible.dtype)
            if self.final_residual_scale != 1.0:
                # Scale the FINAL residual, after contrast, limiter and cast.
                # Default 1 preserves the original inference arithmetic.
                output = visible + self.final_residual_scale * (output - visible)
                update = output.float() - visible.float()
            conditioned.append(output)

            realized_ratio = (
                update.float().square().mean(dim=(1, 2, 3)).sqrt()
                / visible.float().square().mean(
                    dim=(1, 2, 3)
                ).add(self.eps).sqrt()
            )
            realized_ratios.append(realized_ratio.detach())
            context_means.append(thermal_context.detach().abs().mean())
            null_raw_ratios.append(
                (
                    null_raw_update.float().square().mean(
                        dim=(1, 2, 3)
                    ).sqrt()
                    / visible.float().square().mean(
                        dim=(1, 2, 3)
                    ).add(self.eps).sqrt()
                ).detach()
            )
            contrast_raw_ratios.append(
                (
                    raw_update.float().square().mean(
                        dim=(1, 2, 3)
                    ).sqrt()
                    / visible.float().square().mean(
                        dim=(1, 2, 3)
                    ).add(self.eps).sqrt()
                ).detach()
            )

        ratios = torch.stack(realized_ratios, dim=-1)
        self.last_rms_ratio_by_level = ratios
        self.last_scale_by_level = ratios.mean(dim=0)
        self.last_content_ratio = content_mask.detach().mean()
        self.last_context_abs_mean = torch.stack(context_means).mean()
        self.last_token_diversity = token_diversity.detach()
        self.last_null_raw_rms_by_level = torch.stack(
            null_raw_ratios, dim=-1
        )
        self.last_contrast_raw_rms_by_level = torch.stack(
            contrast_raw_ratios, dim=-1
        )
        return conditioned


@register()
class DFINE(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        hbs_enabled=False,
        hbs_channels=None,
        hbs_strides=None,
        hbs_reduction=4,
        hbs_strict_background=True,
        tflr_enabled=False,
        tflr_in_channels=512,
        tflr_edge_channels=16,
        tflr_hidden_dim=64,
        tflr_samples_per_side=3,
        tflr_max_logit_delta=0.25,
        rgbt_enabled=False,
        rgbt_thermal_dropout=0.15,
        rgbt_thermal_intervention="normal",
        rgbt_train_sdtec_only=False,
        rgbt_lock_norm_stats=False,
        rgbt_sd2_enabled=False,
        rgbt_sd2_hidden_dim=128,
        rgbt_sd2_num_levels=2,
        rgbt_sd2_bottleneck_dim=64,
        rgbt_sd2_max_rms_ratio=0.03,
        rgbt_sd2_variant="global",
        rgbt_sd2_num_tokens=4,
        rgbt_sd2_num_heads=4,
        rgbt_sd2_thermal_contrastive=False,
        rgbt_sd2_zero_anchored=False,
        rgbt_sd2_zero_anchored_levels=None,
        rgbt_sd2_final_residual_scale=1.0,
        mote_enabled=False,
        mote_embed_dim=64,
        mote_num_candidates=4,
        mote_max_rms_ratio=0.03,
        mote_final_residual_scale=0.5,
        mote_objectness_loss_weight=0.1,
        mote_layout_fix=False,
        rgbt_freeze_thermal_stream=False,
        rgbt_thermal_forward_chunk_size=0,
        rgbt_checkpoint_decoder_training=False,
        mfam_enabled=False,
        mfam_width=32,
        mfam_supervision='sam',
        mfam_aux_weight=1.0,
        mfam_variant='single',
        sgc_enabled=False,
        sgc_supervision='sam',
        sgc_aux_weight=10.0,
        sgc_decay_start=-1,
        sgc_decay_end=-1,
        sgc_s4_rescue='none',
        sbra_enabled=False,
        sbra_width=64,
        sbra_supervision='sam',
        sbra_aux_weight=1.0,
        sqmi_train_only=False,
        sqmi_train_gate=True,
        stql_enabled=False,
        stql_s8_channels=256,
        stql_qcer_dim=128,
        sqer_enabled=False,
        sqer_bypass=False,
        sqer_bypass_until_epoch=-1,
        sqer_evidence_only=False,
        sqer_topk=64,
        sqer_roi_size=16,
        sqer_roi_expand=2.0,
        sqer_min_roi_width_px=32.0,
        sqer_min_roi_height_px=32.0,
        qcer_enabled=False,
        qcer_thermal_channels=(128, 128),
        qcer_topk=64,
        qcer_temperature=0.1,
        qcer_uniform_attention=False,
        qcer_bypass=False,
        qdmf_enabled=False,
        qdmf_bypass=False,
        qdmf_scales=(8, 16, 32),
        qdmf_roi_expand=1.5,
        qdmf_roi_grid=3,
        qdmf_num_heads=4,
        qdmf_gate_dim=64,
        qdmf_gate_hidden=128,
        qdmf_gate_init_bias=-2.0,
        qdmf_output_init_std=0.001,
        qdmf_ir_dropout=0.10,
        qdmf_detach_ir_features=True,
        qdmf_detach_base_boxes=True,
        qdmf_detach_rgb_uncertainty=True,
        qdmf_matcher_use_base_outputs=True,
        qdmf_residual_warmup_start=0,
        qdmf_residual_warmup_end=6,
        qdmf_residual_init_scale=0.10,
        qdmf_residual_final_scale=1.00,
        qdmf_log_diagnostics=True,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.stql_enabled = bool(stql_enabled)
        self.sqer_enabled = bool(sqer_enabled)
        self.sqer_bypass = bool(sqer_bypass)
        self.sqer_bypass_until_epoch = int(sqer_bypass_until_epoch)
        self.sqer_evidence_only = bool(sqer_evidence_only)
        self.qcer_enabled = bool(qcer_enabled)
        self.qcer_bypass = bool(qcer_bypass)
        self.qdmf_enabled = bool(qdmf_enabled)
        self.qdmf_bypass = bool(qdmf_bypass)
        self.qdmf_matcher_use_base_outputs = bool(
            qdmf_matcher_use_base_outputs
        )
        self.qdmf_log_diagnostics = bool(qdmf_log_diagnostics)
        if tuple(int(value) for value in qdmf_scales) != (8, 16, 32):
            raise ValueError("QDMF v1 scales are frozen to [8, 16, 32]")
        if self.qdmf_enabled and self.qcer_enabled:
            raise ValueError("QDMF replaces QCER; both cannot be enabled")
        if self.sqer_enabled and (self.qcer_enabled or self.qdmf_enabled):
            raise ValueError("S-QER1 v1 is screened independently of QCER/QDMF")
        self.stql_qcer_dim = int(stql_qcer_dim)
        if self.stql_enabled or self.qcer_enabled or self.sqer_enabled:
            query_dim = int(getattr(self.decoder, "hidden_dim", 256))
            self.stql_qcer_query = (
                SharedQueryProjection(
                    query_dim=query_dim,
                    retrieval_dim=self.stql_qcer_dim,
                )
                if self.stql_enabled or self.qcer_enabled
                else None
            )
            self.decoder.return_query_features = True
        else:
            self.stql_qcer_query = None
        if self.qdmf_enabled:
            self.decoder.return_query_features = True
        self.stql_pixel_projection = (
            STQLPixelProjection(
                in_channels=int(stql_s8_channels),
                retrieval_dim=self.stql_qcer_dim,
            )
            if self.stql_enabled
            else None
        )
        if self.sqer_enabled:
            return_idx = list(getattr(self.backbone, "return_idx", ()))
            if len(return_idx) < 3 or return_idx[0] != 1:
                raise ValueError(
                    "S-QER1 requires backbone return_idx starting with [1, 2, 3] "
                    "to expose RGB S8/S16/S32"
                )
            channels = list(getattr(self.backbone, "_out_channels", ()))
            self.sqer = SAMQueryEvidenceReader(
                s4_channels=int(channels[0]),
                s8_channels=int(channels[return_idx[0]]),
                query_dim=int(getattr(self.decoder, "hidden_dim", 128)),
                num_classes=int(getattr(self.decoder, "num_classes", 1)),
                topk=int(sqer_topk),
                roi_size=int(sqer_roi_size),
                roi_expand=float(sqer_roi_expand),
                min_roi_width_px=float(sqer_min_roi_width_px),
                min_roi_height_px=float(sqer_min_roi_height_px),
                evidence_only=self.sqer_evidence_only,
            )
        else:
            self.sqer = None
        self.qcer = None
        self.qdmf = None
        self.sqmi_train_only = bool(sqmi_train_only)
        self.sqmi_train_gate = bool(sqmi_train_gate)
        self.sbra_enabled = bool(sbra_enabled)
        self.sbra_supervision = str(sbra_supervision)
        self.sbra_aux_weight = float(sbra_aux_weight)
        if self.sbra_supervision not in ('sam', 'box'):
            raise ValueError(self.sbra_supervision)
        if self.sbra_enabled:
            from ...nn.backbone.sam_boundary_relation import SAMBoundaryRelation
            self.backbone.stages[2].sbra = SAMBoundaryRelation(256, int(sbra_width))
        if mfam_variant not in ('single', 'support_shape'):
            raise ValueError(mfam_variant)
        self.mfam_variant = mfam_variant
        mfam_class = SupportShapeAggregation if mfam_variant == 'support_shape' else SAMMaskAggregation
        self.mfam = mfam_class(width=int(mfam_width)) if mfam_enabled else None
        self.mfam_supervision = str(mfam_supervision)
        self.mfam_aux_weight = float(mfam_aux_weight)
        self.sgc_enabled = bool(sgc_enabled)
        if sgc_supervision not in ('sam', 'box'):
            raise ValueError(sgc_supervision)
        self.sgc_supervision = str(sgc_supervision)
        if sgc_s4_rescue not in ('none', 'sam', 'box'):
            raise ValueError(sgc_s4_rescue)
        self.sgc_s4_rescue = str(sgc_s4_rescue)
        if self.sgc_s4_rescue != 'none' and (not self.sgc_enabled or sgc_supervision != 'sam'):
            raise ValueError('S4 fallback requires unchanged S8 SAM supervision')
        self.sgc_aux_weight = float(sgc_aux_weight)
        self.sgc_decay_start = int(sgc_decay_start)
        self.sgc_decay_end = int(sgc_decay_end)
        self.sgc_training_epoch = 0
        supervision_scale(0, self.sgc_decay_start, self.sgc_decay_end)
        self.rgbt_enabled = bool(rgbt_enabled)
        self.rgbt_train_sdtec_only = bool(rgbt_train_sdtec_only)
        self.rgbt_lock_norm_stats = bool(rgbt_lock_norm_stats)
        self.rgbt_sd2_enabled = bool(rgbt_sd2_enabled)
        self.mote_enabled = bool(mote_enabled)
        self.rgbt_sd2_variant = str(rgbt_sd2_variant)
        self.rgbt_sd2_thermal_contrastive = bool(
            rgbt_sd2_thermal_contrastive
        )
        self.rgbt_sd2_zero_anchored = bool(rgbt_sd2_zero_anchored)
        self.rgbt_sd2_zero_anchored_levels = (
            []
            if rgbt_sd2_zero_anchored_levels is None
            else list(rgbt_sd2_zero_anchored_levels)
        )
        self.rgbt_freeze_thermal_stream = bool(rgbt_freeze_thermal_stream)
        self.rgbt_thermal_forward_chunk_size = int(
            rgbt_thermal_forward_chunk_size
        )
        self.rgbt_checkpoint_decoder_training = bool(
            rgbt_checkpoint_decoder_training
        )
        if self.rgbt_thermal_forward_chunk_size < 0:
            raise ValueError(
                "rgbt_thermal_forward_chunk_size must be non-negative"
            )
        if self.rgbt_train_sdtec_only and not self.rgbt_enabled:
            raise ValueError("rgbt_train_sdtec_only requires rgbt_enabled=True")
        # Despite the historical ``rgbt_`` prefix this is a model-wide
        # normalization-statistics lock.  Keeping it valid for RGB-only arms
        # lets B0/B1 use the exact same BN-state protocol as B2--B5.
        if self.rgbt_sd2_enabled and not self.rgbt_enabled:
            raise ValueError("rgbt_sd2_enabled requires rgbt_enabled=True")
        if self.mote_enabled and not self.rgbt_enabled:
            raise ValueError("mote_enabled requires rgbt_enabled=True")
        if self.mote_enabled and self.rgbt_sd2_enabled:
            raise ValueError("M-OTE2 and M-SD2 cannot be enabled together")
        if self.qcer_enabled and not self.rgbt_enabled:
            raise ValueError("qcer_enabled requires rgbt_enabled=True")
        if self.qdmf_enabled and not self.rgbt_enabled:
            raise ValueError("qdmf_enabled requires rgbt_enabled=True")
        self.rgbt_thermal_dropout = float(rgbt_thermal_dropout)
        self.rgbt_thermal_intervention = str(rgbt_thermal_intervention)
        valid_thermal_interventions = {
            "normal",
            "zero",
            "zero_content_valid",
            "batch_shuffle",
            "feature_permute",
        }
        if self.rgbt_thermal_intervention not in valid_thermal_interventions:
            raise ValueError(
                "rgbt_thermal_intervention must be one of "
                f"{sorted(valid_thermal_interventions)}, got "
                f"{self.rgbt_thermal_intervention!r}"
            )
        if not 0.0 <= self.rgbt_thermal_dropout < 1.0:
            raise ValueError("rgbt_thermal_dropout must be in [0, 1)")
        if self.rgbt_enabled:
            self.thermal_backbone = copy.deepcopy(backbone)
            self.thermal_encoder = copy.deepcopy(encoder)
        else:
            self.thermal_backbone = None
            self.thermal_encoder = None
        if self.rgbt_freeze_thermal_stream:
            if not self.rgbt_enabled:
                raise ValueError(
                    "rgbt_freeze_thermal_stream requires rgbt_enabled=True"
                )
            for parameter in self.thermal_backbone.parameters():
                parameter.requires_grad_(False)
            for parameter in self.thermal_encoder.parameters():
                parameter.requires_grad_(False)
        if self.qcer_enabled:
            self.qcer = QueryConditionedEvidenceReader(
                thermal_channels=tuple(int(value) for value in qcer_thermal_channels),
                retrieval_dim=self.stql_qcer_dim,
                num_classes=int(getattr(self.decoder, "num_classes", 1)),
                topk=int(qcer_topk),
                temperature=float(qcer_temperature),
                uniform_attention=bool(qcer_uniform_attention),
            )
        if self.qdmf_enabled:
            return_idx = list(getattr(self.backbone, "return_idx", ()))
            backbone_channels = list(getattr(self.backbone, "_out_channels", ()))
            encoder_channels = list(getattr(self.encoder, "out_channels", ()))
            if len(return_idx) < 3 or return_idx[0] != 1:
                raise ValueError(
                    "QDMF requires the thermal backbone to expose S8/S16/S32 "
                    "(HGNetv2 return_idx must start with [1, 2, 3])"
                )
            if len(encoder_channels) != 2:
                raise ValueError(
                    "QDMF v1 expects encoded thermal S16/S32 detector levels"
                )
            thermal_channels = (
                int(backbone_channels[return_idx[0]]),
                int(encoder_channels[0]),
                int(encoder_channels[1]),
            )
            self.qdmf = QueryConditionedDynamicModalityFusion(
                query_dim=int(getattr(self.decoder, "hidden_dim", 256)),
                thermal_channels=thermal_channels,
                roi_expand=qdmf_roi_expand,
                roi_grid=qdmf_roi_grid,
                num_heads=qdmf_num_heads,
                gate_dim=qdmf_gate_dim,
                gate_hidden=qdmf_gate_hidden,
                gate_init_bias=qdmf_gate_init_bias,
                output_init_std=qdmf_output_init_std,
                ir_dropout=qdmf_ir_dropout,
                detach_ir_features=qdmf_detach_ir_features,
                detach_base_boxes=qdmf_detach_base_boxes,
                detach_rgb_uncertainty=qdmf_detach_rgb_uncertainty,
                residual_warmup_start=qdmf_residual_warmup_start,
                residual_warmup_end=qdmf_residual_warmup_end,
                residual_init_scale=qdmf_residual_init_scale,
                residual_final_scale=qdmf_residual_final_scale,
                log_diagnostics=qdmf_log_diagnostics,
            )
            self.decoder.return_qdmf_head_state = True
        if self.rgbt_sd2_enabled:
            if self.rgbt_sd2_variant == "global":
                self.sd2_conditioner = SpatiallyDecoupledThermalConditioner(
                    hidden_dim=rgbt_sd2_hidden_dim,
                    num_levels=rgbt_sd2_num_levels,
                    bottleneck_dim=rgbt_sd2_bottleneck_dim,
                    max_rms_ratio=rgbt_sd2_max_rms_ratio,
                )
            elif self.rgbt_sd2_variant == "token":
                self.sd2_conditioner = SpatiallyDecoupledThermalTokenConditioner(
                    hidden_dim=rgbt_sd2_hidden_dim,
                    num_levels=rgbt_sd2_num_levels,
                    bottleneck_dim=rgbt_sd2_bottleneck_dim,
                    num_tokens=rgbt_sd2_num_tokens,
                    num_heads=rgbt_sd2_num_heads,
                    max_rms_ratio=rgbt_sd2_max_rms_ratio,
                    thermal_contrastive=rgbt_sd2_thermal_contrastive,
                    zero_anchored=rgbt_sd2_zero_anchored,
                    zero_anchored_levels=rgbt_sd2_zero_anchored_levels,
                    final_residual_scale=rgbt_sd2_final_residual_scale,
                )
            else:
                raise ValueError(
                    "rgbt_sd2_variant must be 'global' or 'token', got "
                    f"{self.rgbt_sd2_variant!r}"
                )
        else:
            self.sd2_conditioner = None
        if self.mote_enabled:
            if not self.rgbt_freeze_thermal_stream:
                raise ValueError(
                    "M-OTE2 requires the verified thermal backbone/encoder to be frozen"
                )
            return_idx = list(getattr(backbone, "return_idx", ()))
            backbone_channels = list(getattr(backbone, "_out_channels", ()))
            encoder_channels = list(getattr(encoder, "out_channels", ()))
            if len(return_idx) < 3 or return_idx[0] != 1 or len(encoder_channels) != 2:
                raise ValueError(
                    "M-OTE2 requires backbone S8/S16/S32 and encoder S16/S32 outputs"
                )
            self.mote_fusion = TargetEvidenceThermalFusion(
                visible_channels=int(encoder_channels[0]),
                thermal_s8_channels=int(backbone_channels[return_idx[0]]),
                thermal_s16_channels=int(encoder_channels[0]),
                embed_dim=int(mote_embed_dim),
                num_candidates=int(mote_num_candidates),
                max_rms_ratio=float(mote_max_rms_ratio),
                final_residual_scale=float(mote_final_residual_scale),
                objectness_loss_weight=float(mote_objectness_loss_weight),
                layout_fix=bool(mote_layout_fix),
            )
        else:
            self.mote_fusion = None
        self.hbs_enabled = bool(hbs_enabled)
        if self.hbs_enabled:
            if hbs_channels is None or hbs_strides is None:
                raise ValueError("HBS requires hbs_channels and hbs_strides")
            self.hbs = HierarchicalBackgroundSmoothing(
                channels=hbs_channels,
                strides=hbs_strides,
                reduction=hbs_reduction,
                strict_background=hbs_strict_background,
            )
        else:
            self.hbs = None
        self.tflr_enabled = bool(tflr_enabled)
        self.localization_refiner = (
            TargetGuidedFrequencyLocalizationResidual(
                in_channels=tflr_in_channels,
                edge_channels=tflr_edge_channels,
                hidden_dim=tflr_hidden_dim,
                samples_per_side=tflr_samples_per_side,
                max_logit_delta=tflr_max_logit_delta,
            )
            if self.tflr_enabled
            else None
        )
        if self.rgbt_train_sdtec_only:
            sdtec_parameter_count = 0
            for name, parameter in self.named_parameters():
                trainable = name.startswith("decoder.sdtec_")
                if name.startswith(
                    (
                        "decoder.sdtec_candidate_tokenizer.target_proj.",
                        "decoder.sdtec_candidate_tokenizer.target_norm.",
                        "decoder.sdtec_candidate_tokenizer.target_score_head.",
                    )
                ):
                    trainable = False
                parameter.requires_grad_(trainable)
                sdtec_parameter_count += int(trainable)
            if sdtec_parameter_count == 0:
                raise RuntimeError(
                    "rgbt_train_sdtec_only found no decoder.sdtec_* parameters"
                )
        if self.sqmi_train_only:
            if getattr(self.decoder, "sqmi", None) is None:
                raise ValueError("sqmi_train_only requires decoder.sqmi")
            sqmi_parameter_count = 0
            for name, parameter in self.named_parameters():
                trainable = name.startswith("decoder.sqmi.")
                if not self.sqmi_train_gate and name.startswith("decoder.sqmi.gate."):
                    trainable = False
                parameter.requires_grad_(trainable)
                sqmi_parameter_count += int(trainable)
            if sqmi_parameter_count == 0:
                raise RuntimeError("sqmi_train_only found no decoder.sqmi parameters")

    def train(self, mode=True):
        super().train(mode)
        if mode and self.rgbt_train_sdtec_only:
            # Zero learning rates do not freeze BatchNorm running statistics.
            # Keep mature modality streams in their checkpoint state while
            # autograd still traverses the fixed detector heads to SDTEC.
            self.backbone.eval()
            self.encoder.eval()
            self.thermal_backbone.eval()
            self.thermal_encoder.eval()
        if mode and self.sqmi_train_only:
            # The detector supplies fixed queries and S4 features.  Freezing
            # parameters alone is insufficient because backbone BatchNorm
            # buffers would otherwise drift away from the C checkpoint.
            self.backbone.eval()
            self.encoder.eval()
        if mode and self.rgbt_lock_norm_stats:
            # A tiny optimizer LR does not constrain BatchNorm running means and
            # variances.  Keep those checkpoint statistics fixed while leaving
            # affine parameters and the rest of each stream trainable.
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        if mode and self.rgbt_freeze_thermal_stream:
            self.thermal_backbone.eval()
            self.thermal_encoder.eval()
        return self

    def set_training_epoch(self, epoch):
        self.sgc_training_epoch = int(epoch)
        if self.sqer_bypass_until_epoch >= 0:
            self.sqer_bypass = int(epoch) < self.sqer_bypass_until_epoch
        if self.qdmf is not None:
            self.qdmf.set_training_progress(float(epoch))
        if hasattr(self.backbone, "set_training_epoch"):
            self.backbone.set_training_epoch(epoch)
        if self.thermal_backbone is not None and hasattr(
            self.thermal_backbone, "set_training_epoch"
        ):
            self.thermal_backbone.set_training_epoch(epoch)
        if hasattr(self.decoder, "set_training_epoch"):
            self.decoder.set_training_epoch(epoch)

    def _sgc_supervision_scale(self):
        return supervision_scale(
            self.sgc_training_epoch,
            self.sgc_decay_start,
            self.sgc_decay_end,
        )

    def forward(
        self,
        x,
        targets=None,
        qcer_availability=None,
        qdmf_availability=None,
    ):
        self.backbone.sqer_capture_s4 = bool(self.sqer_enabled)
        self.backbone.sqer_s4_source = None
        self.backbone.sgc_capture_s4 = bool(
            self.training and self.sgc_enabled and targets is not None
            and self.sgc_s4_rescue != 'none' and self._sgc_supervision_scale() > 0
        )
        thermal_feature = None
        thermal_features = None
        thermal_qdmf_features = None
        thermal_dropout_mask = None
        thermal_content_mask = None
        hrqs_source = None
        requires_s8_source = bool(
            getattr(self.decoder, "hrqs_enabled", False)
            or getattr(self.decoder, "sqfr_enabled", False)
            or self.mfam is not None
            or self.sgc_enabled
            or self.stql_enabled
            or self.sqer_enabled
            or self.qdmf_enabled
            or self.mote_enabled
        )
        if self.rgbt_enabled:
            if x.ndim != 4 or x.shape[1] != 6:
                raise RuntimeError(
                    "RGB-T DFINE expects a [B, 6, H, W] tensor, got "
                    f"{tuple(x.shape)}"
                )
            visible_input, thermal_input = x[:, :3], x[:, 3:]
            if self.training and self.rgbt_thermal_dropout > 0:
                thermal_dropout_mask = (
                    torch.rand(x.shape[0], device=x.device)
                    < self.rgbt_thermal_dropout
                )
                thermal_input = thermal_input.masked_fill(
                    thermal_dropout_mask[:, None, None, None], 0.0
                )
            else:
                thermal_dropout_mask = torch.zeros(
                    x.shape[0], device=x.device, dtype=torch.bool
                )
            if not self.training:
                if self.rgbt_thermal_intervention == "zero":
                    thermal_input = torch.zeros_like(thermal_input)
                    # M2 guarantees an exact C+D fallback for unavailable or
                    # intentionally removed thermal input.  Mark the entire
                    # batch invalid instead of asking a learned presence head
                    # to recognize the synthetic all-zero image indirectly.
                    thermal_dropout_mask = torch.ones_like(thermal_dropout_mask)
                elif self.rgbt_thermal_intervention == "zero_content_valid":
                    # Causal diagnostic: remove every thermal pixel while
                    # deliberately leaving the calibration path enabled.  If
                    # this matches normal thermal, the learned gain is an
                    # input-independent logit bias rather than useful content.
                    thermal_input = torch.zeros_like(thermal_input)
                elif self.rgbt_thermal_intervention == "batch_shuffle":
                    if thermal_input.shape[0] > 1:
                        thermal_input = torch.roll(thermal_input, 1, dims=0)
            # MA1 uses this raw-input predicate as a structural guard against
            # the fixed-bias shortcut found in M3.  A deliberately all-zero
            # thermal image must produce an exact zero fusion correction even
            # if frozen backbone biases emit non-zero feature tensors.
            thermal_content_mask = (
                thermal_input.detach().abs().amax(dim=(1, 2, 3)) > 1e-8
            )
            if self.rgbt_freeze_thermal_stream:
                # The fixed thermal extractor has no batch-dependent training
                # state.  Run it first and optionally in small chunks so its
                # temporary activations never overlap the trainable RGB
                # backbone activations.  The logical physical batch remains 8.
                with torch.no_grad():
                    chunk_size = self.rgbt_thermal_forward_chunk_size
                    if 0 < chunk_size < thermal_input.shape[0]:
                        thermal_chunks = [
                            self.thermal_backbone(chunk)
                            for chunk in thermal_input.split(
                                chunk_size, dim=0
                            )
                        ]
                        thermal_features = [
                            torch.cat(
                                [chunk[level] for chunk in thermal_chunks],
                                dim=0,
                            )
                            for level in range(len(thermal_chunks[0]))
                        ]
                        del thermal_chunks
                    else:
                        thermal_features = self.thermal_backbone(
                            thermal_input
                        )
                    if thermal_input.is_cuda:
                        # WDDM charges cached thermal workspaces against the
                        # system commit limit even when a physical batch is
                        # smaller than the configured chunk size.  Release the
                        # frozen sub-pass cache before RGB autograd activations.
                        torch.cuda.empty_cache()
                x = self.backbone(visible_input)
            else:
                x = self.backbone(visible_input)
                thermal_features = self.thermal_backbone(thermal_input)
            thermal_s8_source = None
            if requires_s8_source:
                expected_levels = len(self.encoder.in_channels)
                if len(x) != expected_levels + 1:
                    raise RuntimeError(
                        "The enabled high-resolution branch expects one S8 feature "
                        "followed by the ordinary "
                        f"{expected_levels} detector levels, got {len(x)}"
                    )
                if len(thermal_features) != expected_levels + 1:
                    raise RuntimeError(
                        "The high-resolution RGB-T path requires matching backbone "
                        "level counts"
                    )
                hrqs_source = x[0]
                thermal_s8_source = thermal_features[0]
                x = x[1:]
                # HRQS supplements visible query initialization only.  The
                # frozen M-series thermal encoder keeps its mature S16/S32 path.
                thermal_features = thermal_features[1:]
            thermal_encoded = self.thermal_encoder(thermal_features)
            if (
                not self.training
                and self.rgbt_thermal_intervention == "feature_permute"
            ):
                # Deterministic reversal of the HxW descriptor order.  The
                # feature values are unchanged; only their spatial placement
                # is permuted.  A coordinate-free tokenizer must be invariant.
                thermal_encoded = [
                    feature.flatten(2).flip(-1).reshape_as(feature)
                    for feature in thermal_encoded
                ]
            thermal_features = thermal_encoded
            thermal_feature = thermal_encoded[-1]
            if self.qdmf_enabled:
                thermal_qdmf_features = [
                    thermal_s8_source,
                    thermal_encoded[0],
                    thermal_encoded[1],
                ]
        else:
            x = self.backbone(x)
            if requires_s8_source:
                expected_levels = len(self.encoder.in_channels)
                if len(x) != expected_levels + 1:
                    raise RuntimeError(
                        "The enabled high-resolution branch expects one S8 feature "
                        "followed by the ordinary "
                        f"{expected_levels} detector levels, got {len(x)}"
                    )
                hrqs_source = x[0]
                x = x[1:]
        s8_localization_source = (
            hrqs_source
            if self.localization_refiner is not None and hrqs_source is not None
            else (x[0] if self.localization_refiner is not None else None)
        )
        sgc_loss = None
        sbra_loss = None
        if self.sbra_enabled and self.training and targets is not None:
            sbra_loss = self.sbra_aux_weight * self.backbone.stages[2].sbra.supervision(targets, self.sbra_supervision)
        sgc_scale = self._sgc_supervision_scale()
        if (
            self.sgc_enabled
            and self.training
            and targets is not None
            and sgc_scale > 0
        ):
            sgc_loss = (
                self.sgc_aux_weight
                * sgc_scale
                * group_loss(hrqs_source, targets, self.sgc_supervision)
            )
            if self.sgc_s4_rescue != 'none':
                extra, _ = rescue_loss(
                    self.backbone.sgc_s4_source, hrqs_source, targets, self.sgc_s4_rescue
                )
                sgc_loss = sgc_loss + self.sgc_aux_weight * sgc_scale * extra
        self.backbone.sgc_s4_source = None
        qrl_source = getattr(self.backbone, "qrl_source", None)
        mdqa_source = getattr(self.backbone, "mdqa_source", None)
        sqmi_source = getattr(self.backbone, "sqmi_source", None)
        spatial_importance_logits = getattr(self.backbone, "spatial_importance_logits", None)
        pdbr_boundary_logits = getattr(self.backbone, "pdbr_boundary_logits", None)
        tndp_detail_prediction = getattr(
            self.backbone, "tndp_detail_prediction", None
        )
        tndp_detail_target = getattr(self.backbone, "tndp_detail_target", None)
        bpc_boundary_logits = getattr(self.backbone, "bpc_boundary_logits", None)
        spar_fused_features = getattr(self.backbone, "spar_fused_features", None)
        sabr_outputs = getattr(self.backbone, "sabr_outputs", None)
        sbox_extreme_logits = getattr(
            self.backbone, "sbox_extreme_logits", None
        )
        sbrd_stage_features = getattr(self.backbone, "sbrd_stage_features", None)
        qcsr_stage_features = getattr(self.backbone, "qcsr_stage_features", None)
        srtod_reconstruction_loss = getattr(
            self.backbone, "srtod_reconstruction_loss", None
        )
        mfam_loss = None
        if self.mfam is not None:
            x = list(x)
            x[0], mfam_logits = self.mfam(hrqs_source, x[0])
            if self.training and targets is not None and mfam_logits is not None:
                if self.mfam_variant == 'support_shape':
                    mfam_loss = {key: self.mfam_aux_weight * value for key, value in
                                 support_shape_loss(mfam_logits, targets, self.mfam_supervision).items()}
                else:
                    mfam_loss = self.mfam_aux_weight * region_loss(
                        mfam_logits, targets, self.mfam_supervision
                    )
        x = self.encoder(x)
        mote_loss = None
        if self.mote_fusion is not None:
            if thermal_s8_source is None or thermal_features is None:
                raise RuntimeError("M-OTE2 requires IR S8 and encoded S16 features")
            x = list(x)
            x[0], mote_loss = self.mote_fusion(
                x[0],
                thermal_s8_source,
                thermal_features[0],
                content_mask=thermal_content_mask,
                targets=targets if self.training else None,
            )
        if self.sd2_conditioner is not None:
            x = self.sd2_conditioner(
                x,
                thermal_features,
                content_mask=thermal_content_mask,
            )
        hbs_aux_outputs = None
        if self.training and self.hbs is not None and targets is not None:
            hbs_features = self.hbs(x, targets)
            # The original path is evaluated first and remains bitwise
            # untouched.  HBS reuses the same detector as an auxiliary branch.
            main_outputs = self.decoder(
                x,
                targets,
                qrl_source=qrl_source,
                mdqa_source=mdqa_source,
                sqmi_source=sqmi_source,
                qcsr_stage_features=qcsr_stage_features,
                thermal_feature=thermal_feature,
                thermal_features=thermal_features,
                thermal_dropout_mask=thermal_dropout_mask,
                thermal_content_mask=thermal_content_mask,
                hrqs_source=hrqs_source,
            )
            hbs_aux_outputs = self.decoder(
                hbs_features,
                targets,
                qrl_source=qrl_source,
                mdqa_source=mdqa_source,
                sqmi_source=sqmi_source,
                qcsr_stage_features=qcsr_stage_features,
                thermal_feature=thermal_feature,
                thermal_features=thermal_features,
                thermal_dropout_mask=thermal_dropout_mask,
                thermal_content_mask=thermal_content_mask,
                hrqs_source=hrqs_source,
            )
            x = main_outputs
        else:
            if self.training and self.rgbt_checkpoint_decoder_training:
                def checkpointed_decoder(*memory_features):
                    return self.decoder(
                        list(memory_features),
                        targets,
                        qrl_source=qrl_source,
                        mdqa_source=mdqa_source,
                        sqmi_source=sqmi_source,
                        qcsr_stage_features=qcsr_stage_features,
                        thermal_feature=thermal_feature,
                        thermal_features=thermal_features,
                        thermal_dropout_mask=thermal_dropout_mask,
                        thermal_content_mask=thermal_content_mask,
                        hrqs_source=hrqs_source,
                    )

                x = activation_checkpoint(
                    checkpointed_decoder,
                    *x,
                    use_reentrant=False,
                )
            else:
                x = self.decoder(
                    x,
                    targets,
                    qrl_source=qrl_source,
                    mdqa_source=mdqa_source,
                    sqmi_source=sqmi_source,
                    qcsr_stage_features=qcsr_stage_features,
                    thermal_feature=thermal_feature,
                    thermal_features=thermal_features,
                    thermal_dropout_mask=thermal_dropout_mask,
                    thermal_content_mask=thermal_content_mask,
                    hrqs_source=hrqs_source,
                )

        query_features = None
        if self.stql_enabled or self.sqer_enabled or self.qcer_enabled or self.qdmf_enabled:
            if "query_features" not in x:
                raise RuntimeError(
                    "STQL/QCER/QDMF requested final query features, but the decoder "
                    "did not expose them"
                )
            query_features = x.pop("query_features")
            shared_query = (
                self.stql_qcer_query(query_features)
                if self.stql_enabled or self.qcer_enabled
                else None
            )
            if self.qdmf_enabled:
                required_head_state = {
                    "qdmf_head_offset",
                    "qdmf_base_corners",
                    "qdmf_pred_corners",
                    "qdmf_ref_points",
                }
                missing = required_head_state.difference(x)
                if missing:
                    raise RuntimeError(
                        f"QDMF decoder head state is missing {sorted(missing)}"
                    )
                if thermal_qdmf_features is None:
                    raise RuntimeError("QDMF requires thermal S8/S16/S32 features")
                if qdmf_availability is None and targets is not None:
                    availability_values = []
                    for target in targets:
                        value = target.get("infrared_available")
                        availability_values.append(
                            True
                            if value is None
                            else bool(torch.as_tensor(value).reshape(-1)[0])
                        )
                    qdmf_availability = torch.tensor(
                        availability_values,
                        device=query_features.device,
                        dtype=torch.bool,
                    )
                if qdmf_availability is None:
                    qdmf_availability = torch.ones(
                        query_features.shape[0],
                        device=query_features.device,
                        dtype=torch.bool,
                    )
                else:
                    qdmf_availability = torch.as_tensor(
                        qdmf_availability,
                        device=query_features.device,
                        dtype=torch.bool,
                    ).reshape(query_features.shape[0], -1).all(dim=1)
                if thermal_dropout_mask is not None:
                    qdmf_availability = (
                        qdmf_availability & ~thermal_dropout_mask.bool()
                    )

                logits_base = x["pred_logits"]
                boxes_base = x["pred_boxes"]
                fused_queries, qdmf_diagnostics = self.qdmf(
                    query_features,
                    boxes_base,
                    logits_base,
                    thermal_qdmf_features,
                    availability=qdmf_availability,
                    bypass=self.qdmf_bypass,
                )
                x["base_pred_logits"] = logits_base
                x["base_pred_boxes"] = boxes_base
                x["qdmf_matcher_use_base_outputs"] = (
                    self.qdmf_matcher_use_base_outputs
                )
                if self.qdmf_bypass:
                    logits_final, boxes_final = logits_base, boxes_base
                    corners_final = x["qdmf_pred_corners"]
                else:
                    logits_final, boxes_final, corners_final = (
                        self.decoder.qdmf_final_head(
                            query_features,
                            fused_queries,
                            x["qdmf_head_offset"],
                            x["qdmf_pred_corners"],
                            x["qdmf_base_corners"],
                            x["qdmf_ref_points"],
                        )
                    )
                    available = qdmf_diagnostics["availability"].view(-1, 1, 1)
                    logits_final = torch.where(available, logits_final, logits_base)
                    boxes_final = torch.where(available, boxes_final, boxes_base)
                    corners_final = torch.where(
                        available, corners_final, x["qdmf_pred_corners"]
                    )
                x["pred_logits"] = logits_final
                x["pred_boxes"] = boxes_final
                if "pred_corners" in x:
                    x["pred_corners"] = corners_final
                x["qdmf_gate"] = qdmf_diagnostics["gate"]
                x["qdmf_scale_weight"] = qdmf_diagnostics["scale_weight"]
                x["qdmf_residual"] = qdmf_diagnostics["residual"]
                x["qdmf_availability"] = qdmf_diagnostics["availability"]
                x["qdmf_dropout_mask"] = qdmf_diagnostics["dropout_mask"]
                x["qdmf_area"] = qdmf_diagnostics["area"]
                if self.qdmf_log_diagnostics:
                    x.update(
                        qdmf_basic_statistics(
                            qdmf_diagnostics,
                            query_features,
                            logits_base,
                            logits_final,
                            boxes_base,
                            boxes_final,
                        )
                    )
                for key in required_head_state:
                    x.pop(key)
            if self.stql_enabled:
                if hrqs_source is None:
                    raise RuntimeError("STQL requires the native RGB S8 feature")
                x["stql_query_embeddings"] = shared_query
                x["stql_pixel_features"] = self.stql_pixel_projection(hrqs_source)
            if self.qcer_enabled:
                if thermal_features is None:
                    raise RuntimeError("QCER requires encoded thermal S16/S32 features")
                if qcer_availability is None and targets is not None:
                    availability_values = []
                    for target in targets:
                        value = target.get("infrared_available")
                        if value is None:
                            availability_values.append(True)
                        else:
                            availability_values.append(
                                bool(torch.as_tensor(value).reshape(-1)[0])
                            )
                    qcer_availability = torch.tensor(
                        availability_values,
                        device=shared_query.device,
                        dtype=torch.bool,
                    )
                qcer_result = self.qcer(
                    shared_query,
                    thermal_features,
                    availability=qcer_availability,
                    bypass=self.qcer_bypass,
                )
                x["base_pred_logits"] = x["pred_logits"]
                x["pred_logits"] = x["pred_logits"] + qcer_result["delta_logits"]
                for name, value in qcer_result.items():
                    x[f"qcer_{name}"] = value

        if self.sqer_enabled:
            if hrqs_source is None or self.backbone.sqer_s4_source is None:
                raise RuntimeError("S-QER1 requires captured RGB S4 and S8 features")
            if query_features is None:
                raise RuntimeError("S-QER1 decoder query features were not returned")
            base_logits = x["pred_logits"]
            base_boxes = x["pred_boxes"]
            sqer = self.sqer(
                self.backbone.sqer_s4_source,
                hrqs_source,
                query_features,
                base_logits,
                base_boxes,
                bypass=self.sqer_bypass,
                return_aux=self.training and targets is not None,
            )
            x["base_pred_logits"] = base_logits
            x["base_pred_boxes"] = base_boxes
            x["pred_logits"] = base_logits + sqer["delta_logits"]
            x["sqer_delta_logits"] = sqer["delta_logits"]
            x["sqer_query_indices"] = sqer["query_indices"]
            if sqer["roi_grid"] is not None:
                x["sqer_roi_grid"] = sqer["roi_grid"]
            if sqer["attention"] is not None:
                x["sqer_attention"] = sqer["attention"]
            if sqer.get("evidence") is not None:
                x["sqer_evidence"] = sqer["evidence"]
            self.backbone.sqer_s4_source = None

        # S-AUX uses this prediction only for an auxiliary training loss.  It
        if mfam_loss is not None:
            x['mfam_region_loss'] = mfam_loss
        if sgc_loss is not None:
            x['sgc_group_loss'] = sgc_loss
        if sbra_loss is not None:
            x['sbra_relation_loss'] = sbra_loss
        # deliberately does not modify the backbone feature or downsampling.
        if spatial_importance_logits is not None:
            x["spatial_importance_logits"] = spatial_importance_logits
        if pdbr_boundary_logits is not None:
            x["pdbr_boundary_logits"] = pdbr_boundary_logits
        if tndp_detail_prediction is not None and tndp_detail_target is not None:
            x["tndp_detail_prediction"] = tndp_detail_prediction
            x["tndp_detail_target"] = tndp_detail_target
        if bpc_boundary_logits is not None:
            x["bpc_boundary_logits"] = bpc_boundary_logits
        if spar_fused_features is not None:
            x["spar_fused_features"] = spar_fused_features
        if sabr_outputs is not None:
            x["sabr_boundary_logits"] = sabr_outputs["boundary_logits"]
            x["sabr_body_logits"] = sabr_outputs["body_logits"]
        if sbox_extreme_logits is not None:
            x["sbox_extreme_logits"] = sbox_extreme_logits
        if sbrd_stage_features is not None:
            x["sbrd_stage_features"] = sbrd_stage_features
        if srtod_reconstruction_loss is not None:
            x["srtod_reconstruction_loss"] = srtod_reconstruction_loss
        if hbs_aux_outputs is not None:
            x["hbs_aux_outputs"] = hbs_aux_outputs
        if self.sd2_conditioner is not None:
            x["msd2_scale_by_level"] = (
                self.sd2_conditioner.last_scale_by_level
            )
            x["msd2_rms_ratio_by_level"] = (
                self.sd2_conditioner.last_rms_ratio_by_level
            )
            x["msd2_content_ratio"] = self.sd2_conditioner.last_content_ratio
            x["msd2_context_abs_mean"] = (
                self.sd2_conditioner.last_context_abs_mean
            )
            if getattr(
                self.sd2_conditioner, "last_token_diversity", None
            ) is not None:
                x["msd2_token_diversity"] = (
                    self.sd2_conditioner.last_token_diversity
                )
            if getattr(
                self.sd2_conditioner, "last_null_raw_rms_by_level", None
            ) is not None:
                x["msd2_null_raw_rms_by_level"] = (
                    self.sd2_conditioner.last_null_raw_rms_by_level
                )
                x["msd2_contrast_raw_rms_by_level"] = (
                    self.sd2_conditioner.last_contrast_raw_rms_by_level
                )
        if self.mote_fusion is not None:
            if mote_loss is not None:
                x["mote_ir_objectness_loss"] = mote_loss
            x["mote_candidate_scores"] = self.mote_fusion.last_candidate_scores
            x["mote_gate_mean"] = self.mote_fusion.last_gate_mean
            x["mote_object_attention_mean"] = (
                self.mote_fusion.last_object_attention_mean
            )
            x["mote_update_ratio"] = self.mote_fusion.last_update_ratio
        if self.localization_refiner is not None:
            x["base_pred_boxes"] = x["pred_boxes"]
            x["pred_boxes"] = self.localization_refiner(
                s8_localization_source,
                x["pred_boxes"],
                x["pred_logits"],
            )

        return x

    def deploy(
        self,
    ):
        self.eval()
        # HBS is a training-only auxiliary branch in SET.  Physically remove
        # it so deployment parameters and execution are identical to A00.
        self.hbs = None
        self.hbs_enabled = False
        if hasattr(self.backbone, "tndp_head"):
            self.backbone.tndp_head = None
            self.backbone.tndp_stage = -1
            self.backbone.tndp_detail_prediction = None
            self.backbone.tndp_detail_target = None
        if hasattr(self.backbone, "spar_fusion"):
            self.backbone.spar_fusion = None
            self.backbone.spar_enabled = False
            self.backbone.spar_fused_features = None
        if hasattr(self.backbone, "sabr_heads"):
            self.backbone.sabr_heads = None
            self.backbone.sabr_enabled = False
            self.backbone.sabr_outputs = None
        if hasattr(self.backbone, "sbox_head"):
            self.backbone.sbox_head = None
            self.backbone.sbox_enabled = False
            self.backbone.sbox_extreme_logits = None
        if hasattr(self.backbone, "sbrd_enabled"):
            self.backbone.sbrd_enabled = False
            self.backbone.sbrd_stage_features = None
        if hasattr(self.backbone, "qcsr_enabled"):
            self.backbone.qcsr_enabled = False
            self.backbone.qcsr_stage_features = None
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
