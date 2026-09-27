"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
import functools
import math
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ...core import register
from .denoising import get_contrastive_denoising_training_group
from .dfine_utils import distance2bbox, weighting_function
from .sam_query_mask_init import SAMQueryMaskInitializer
from .utils import (
    bias_init_with_prob,
    deformable_attention_core_func_v2,
    get_activation,
    inverse_sigmoid,
)

__all__ = ["DFINETransformer"]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class HighResolutionQueryAdmission(nn.Module):
    """Project an existing S8 backbone feature into D-FINE query space.

    The projected map is used only when choosing the initial object queries.
    Decoder cross-attention keeps the mature S16/S32 memory path unchanged,
    which isolates the value of high-resolution candidate locations without
    rebuilding the complete HybridEncoder feature pyramid.
    """

    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(int(in_channels), hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=False),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=False),
        )

    def forward(self, feature):
        return self.projection(feature)


class SparseQueryHighResolutionRefiner(nn.Module):
    """Refine a sparse set of mature queries with query-aligned S8 regions.

    The module never creates queries and never changes classification logits.
    Query features, scores and coarse boxes are detached when they select and
    describe a region, so the extra localization loss cannot reshape the
    mature decoder through a side path. The refined boxes retain the ordinary
    D-FINE gradient path, while S8 receives the new localization gradient.

    A real-region prediction is contrasted with an all-zero-region prediction.
    Consequently the correction is exactly zero when S8 evidence is removed;
    the head cannot learn a query-only box shortcut. The final layer is also
    zero initialized, making the complete module a bit-exact identity at
    initialization.
    """

    VALID_FEATURE_MODES = {"full", "zero", "shifted"}

    def __init__(
        self,
        in_channels=256,
        query_dim=128,
        hidden_dim=64,
        num_heads=4,
        roi_size=5,
        topk=64,
        context_scale=1.5,
        max_logit_delta=0.25,
    ):
        super().__init__()
        if int(roi_size) != 5:
            raise ValueError("SQFR1 preregisters a 5x5 target-neighbourhood grid")
        if int(topk) <= 0:
            raise ValueError("SQFR1 topk must be positive")
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("SQFR1 hidden_dim must be divisible by num_heads")
        if float(context_scale) <= 1.0:
            raise ValueError("SQFR1 context_scale must be greater than one")
        if not 0.0 < float(max_logit_delta) <= 1.0:
            raise ValueError("SQFR1 max_logit_delta must be in (0, 1]")

        self.roi_size = int(roi_size)
        self.topk = int(topk)
        self.context_scale = float(context_scale)
        self.max_logit_delta = float(max_logit_delta)
        self.feature_mode = "full"

        self.feature_projection = nn.Sequential(
            nn.Conv2d(int(in_channels), int(hidden_dim), kernel_size=1, bias=False),
            nn.SiLU(inplace=False),
        )
        self.query_projection = nn.Linear(
            int(query_dim), int(hidden_dim), bias=False
        )
        self.region_attention = nn.MultiheadAttention(
            int(hidden_dim), int(num_heads), batch_first=True
        )
        self.attention_norm = nn.LayerNorm(int(hidden_dim))
        self.delta_head = nn.Sequential(
            nn.Linear(5 * int(hidden_dim), int(hidden_dim), bias=False),
            nn.SiLU(inplace=False),
            nn.Linear(int(hidden_dim), 4, bias=False),
        )
        nn.init.zeros_(self.delta_head[-1].weight)

        self.last_selected_indices = None
        self.last_residual = None
        self.last_residual_abs_mean = None
        self.last_residual_abs_max = None

    def _intervene(self, feature):
        if self.feature_mode == "full":
            return feature
        if self.feature_mode == "zero":
            return torch.zeros_like(feature)
        if self.feature_mode == "shifted":
            shift = (
                max(1, feature.shape[-2] // 2),
                max(1, feature.shape[-1] // 2),
            )
            return torch.roll(feature, shifts=shift, dims=(-2, -1))
        raise ValueError(
            f"unsupported SQFR1 feature_mode {self.feature_mode!r}; "
            f"expected one of {sorted(self.VALID_FEATURE_MODES)}"
        )

    def _sample_regions(self, feature, boxes):
        # Five deliberately placed samples expose the target centre, the four
        # box sides at +/-0.5, and an outer context ring at context_scale/2.
        outer = 0.5 * self.context_scale
        offsets = boxes.new_tensor((-outer, -0.5, 0.0, 0.5, outer))
        offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
        center_x, center_y, width, height = boxes.unbind(-1)
        grid_x = center_x[..., None, None] + width[..., None, None] * offset_x
        grid_y = center_y[..., None, None] + height[..., None, None] * offset_y
        grid = torch.stack((grid_x, grid_y), dim=-1).clamp(0.0, 1.0)
        grid = grid.mul(2.0).sub(1.0)

        batch, count = boxes.shape[:2]
        sampled = F.grid_sample(
            feature,
            grid.reshape(batch, count * self.roi_size, self.roi_size, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled = sampled.reshape(
            batch, feature.shape[1], count, self.roi_size, self.roi_size
        )
        return sampled.permute(0, 2, 3, 4, 1).contiguous()

    def _describe(self, query, region):
        batch, count, height, width, channels = region.shape
        region_tokens = region.reshape(batch * count, height * width, channels)
        query_tokens = query.reshape(batch * count, 1, channels)
        attended, _ = self.region_attention(
            query_tokens, region_tokens, region_tokens, need_weights=False
        )
        attended = self.attention_norm(query_tokens + attended).reshape(
            batch, count, channels
        )

        target = region[:, :, 2, 2]
        body = region[:, :, 1:4, 1:4].mean(dim=(2, 3))
        inner = region[:, :, 1:4, 1:4].reshape(batch, count, 9, channels)
        boundary = torch.cat((inner[:, :, :4], inner[:, :, 5:]), dim=2).mean(dim=2)
        outer = torch.cat(
            (
                region[:, :, 0].reshape(batch, count, 5, channels),
                region[:, :, 4].reshape(batch, count, 5, channels),
                region[:, :, 1:4, 0].reshape(batch, count, 3, channels),
                region[:, :, 1:4, 4].reshape(batch, count, 3, channels),
            ),
            dim=2,
        ).mean(dim=2)
        return self.delta_head(
            torch.cat((attended, target, body, boundary, outer), dim=-1)
        )

    def forward(self, s8_feature, boxes, logits, query_features):
        if boxes.shape[1] < self.topk:
            raise ValueError(
                f"SQFR1 topk={self.topk} exceeds query count={boxes.shape[1]}"
            )
        if self.training:
            # The dataset-transfer classifier is nearly random at epoch zero.
            # Dense training guarantees that every Hungarian-matched query can
            # supervise the refiner; inference remains sparse and uses top-k.
            selected_indices = torch.arange(
                boxes.shape[1], device=boxes.device, dtype=torch.long
            ).unsqueeze(0).expand(boxes.shape[0], -1)
        else:
            confidence = logits.detach().sigmoid().amax(dim=-1)
            selected_indices = confidence.topk(self.topk, dim=1).indices
        box_index = selected_indices.unsqueeze(-1).expand(-1, -1, 4)
        query_index = selected_indices.unsqueeze(-1).expand(
            -1, -1, query_features.shape[-1]
        )
        selected_boxes = boxes.detach().gather(1, box_index)
        selected_queries = query_features.detach().gather(1, query_index)

        projected = self.feature_projection(self._intervene(s8_feature))
        regions = self._sample_regions(projected, selected_boxes)
        projected_queries = self.query_projection(selected_queries)
        raw_delta = self._describe(projected_queries, regions)
        raw_null_delta = self._describe(
            projected_queries, torch.zeros_like(regions)
        )
        selected_residual = self.max_logit_delta * torch.tanh(
            raw_delta - raw_null_delta
        ).to(boxes.dtype)

        residual = torch.zeros_like(boxes).scatter(1, box_index, selected_residual)
        base_logits = inverse_sigmoid(boxes)
        refined = boxes + (
            (base_logits + residual).sigmoid() - base_logits.sigmoid()
        )

        self.last_selected_indices = selected_indices.detach()
        self.last_residual = residual.detach()
        self.last_residual_abs_mean = residual.detach().abs().mean()
        self.last_residual_abs_max = residual.detach().abs().max()
        return refined


class QueryAlignedRegionLocalizationCarrier(nn.Module):
    """Read detached X8 phase detail at query-aligned box-side locations.

    The carrier never modifies shared backbone or decoder features.  It emits
    only a zero-initialized residual for the final FDR corner distribution.
    SAM supervises ``region_logits`` outside this module during training; the
    predicted regions are the only region signal available at inference.
    """

    def __init__(
        self,
        source_channels,
        hidden_dim,
        reg_max,
        detail_channels=32,
        context_dim=16,
        delta_hidden_dim=64,
        detail_only_delta=False,
    ):
        super().__init__()
        self.source_channels = int(source_channels)
        self.detail_channels = int(detail_channels)
        self.reg_max = int(reg_max)
        self.detail_only_delta = bool(detail_only_delta)
        self.reduce = nn.Conv2d(
            self.source_channels, self.detail_channels, kernel_size=1, bias=False
        )
        self.region_student = nn.Sequential(
            nn.Conv2d(
                self.detail_channels,
                self.detail_channels,
                kernel_size=3,
                padding=1,
                groups=self.detail_channels,
                bias=False,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.detail_channels, 3, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.region_student[-1].bias, -2.0)
        if self.detail_only_delta:
            self.detail_norm = nn.LayerNorm(
                self.detail_channels, elementwise_affine=False
            )
            self.side_projections = nn.ModuleList(
                nn.Linear(
                    self.detail_channels, self.reg_max + 1, bias=False
                )
                for _ in range(4)
            )
            for projection in self.side_projections:
                nn.init.zeros_(projection.weight)
        else:
            self.query_context = nn.Linear(hidden_dim, context_dim, bias=False)
            self.side_embedding = nn.Parameter(torch.zeros(4, 4))
            delta_input_dim = self.detail_channels + context_dim + 4 + 4
            self.delta_head = MLP(
                delta_input_dim,
                delta_hidden_dim,
                self.reg_max + 1,
                3,
                act="silu",
            )
            nn.init.zeros_(self.delta_head.layers[-1].weight)
            nn.init.zeros_(self.delta_head.layers[-1].bias)

        self.region_mode = "learned"
        self.capture_diagnostics = False
        self.last_region_logits = None
        self.last_attention = None
        self.last_side_tokens = None
        self.last_delta = None
        self.last_delta_without_detail = None
        self.last_delta_without_context = None
        self.last_delta_rms = None

    @staticmethod
    def _pad_even(x):
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x

    def prepare(self, source):
        source = source.detach()
        z8 = self.reduce(source)
        region_logits = self.region_student(z8)
        region_probability = region_logits.sigmoid()
        if self.region_mode == "shifted":
            region_probability = torch.roll(
                region_probability,
                shifts=(region_probability.shape[-2] // 2, region_probability.shape[-1] // 2),
                dims=(-2, -1),
            )
        elif self.region_mode == "uniform":
            region_probability = region_probability.mean(
                dim=(-2, -1), keepdim=True
            ).expand_as(region_probability)
        elif self.region_mode not in {"learned", "zero"}:
            raise ValueError(f"unsupported QRL region_mode: {self.region_mode}")

        z8 = self._pad_even(z8)
        batch, channels, height8, width8 = z8.shape
        phase_detail = F.pixel_unshuffle(z8, 2).view(
            batch, channels, 4, height8 // 2, width8 // 2
        )
        phase_detail = phase_detail - phase_detail.mean(dim=2, keepdim=True)
        phase_detail = phase_detail.flatten(1, 2)

        region_probability = self._pad_even(region_probability)
        region_phases = F.pixel_unshuffle(region_probability, 2).view(
            batch, 3, 4, height8 // 2, width8 // 2
        )
        self.last_region_logits = region_logits
        return {
            "phase_detail": phase_detail,
            "region_phases": region_phases.flatten(1, 2),
            "height8": height8,
            "width8": width8,
        }

    @staticmethod
    def _sampling_grid(ref_boxes, height8, width8):
        ref_boxes = ref_boxes.detach().clamp(0.0, 1.0)
        cx, cy, bw, bh = ref_boxes.unbind(-1)
        cell_x = ref_boxes.new_tensor(1.0 / float(width8))
        cell_y = ref_boxes.new_tensor(1.0 / float(height8))
        span_w = torch.maximum(bw, cell_x)
        span_h = torch.maximum(bh, cell_y)
        x1, x2 = cx - bw / 2, cx + bw / 2
        y1, y2 = cy - bh / 2, cy + bh / 2
        tangent = ref_boxes.new_tensor((-0.5, 0.0, 0.5))
        normal = ref_boxes.new_tensor((-0.5, 0.0, 0.5))

        def vertical(x_edge):
            x = x_edge[..., None, None] + normal[None, None, :, None] * cell_x
            y = cy[..., None, None] + tangent[None, None, None, :] * span_h[..., None, None]
            x = x.expand(-1, -1, 3, 3)
            y = y.expand(-1, -1, 3, 3)
            return torch.stack((x, y), dim=-1).flatten(-3, -2)

        def horizontal(y_edge):
            x = cx[..., None, None] + tangent[None, None, :, None] * span_w[..., None, None]
            y = y_edge[..., None, None] + normal[None, None, None, :] * cell_y
            x = x.expand(-1, -1, 3, 3)
            y = y.expand(-1, -1, 3, 3)
            return torch.stack((x, y), dim=-1).flatten(-3, -2)

        grid = torch.stack(
            (vertical(x1), horizontal(y1), vertical(x2), horizontal(y2)), dim=2
        )
        return grid.clamp(0.0, 1.0).mul(2.0).sub(1.0)

    def detail_only_delta_from_tokens(self, side_tokens):
        """Map four side-detail tokens to FDR deltas without any bypass."""
        if not self.detail_only_delta:
            raise RuntimeError("detail-only delta is disabled for this QRL instance")
        if side_tokens.shape[-2:] != (4, self.detail_channels):
            raise ValueError(
                "expected side tokens ending in "
                f"[4, {self.detail_channels}], got {list(side_tokens.shape)}"
            )
        normalized_tokens = self.detail_norm(side_tokens)
        return torch.cat(
            [
                projection(normalized_tokens[..., side_index, :])
                for side_index, projection in enumerate(self.side_projections)
            ],
            dim=-1,
        )

    def forward(self, query_output, ref_boxes, state):
        batch, queries, _ = query_output.shape
        if self.region_mode == "zero":
            delta = query_output.new_zeros(
                batch, queries, 4 * (self.reg_max + 1)
            )
            self.last_side_tokens = None
            self.last_attention = None
            self.last_delta = delta
            self.last_delta_without_detail = None
            self.last_delta_without_context = None
            self.last_delta_rms = delta.detach().float().square().mean().sqrt()
            return delta

        grid = self._sampling_grid(
            ref_boxes, state["height8"], state["width8"]
        )
        flat_grid = grid.flatten(1, 3).unsqueeze(-2)
        sampled_detail = F.grid_sample(
            state["phase_detail"],
            flat_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).squeeze(-1)
        sampled_region = F.grid_sample(
            state["region_phases"],
            flat_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).squeeze(-1)

        sampled_detail = sampled_detail.view(
            batch, self.detail_channels, 4, queries, 4, 9
        ).permute(0, 3, 4, 1, 2, 5)
        sampled_region = sampled_region.view(
            batch, 3, 4, queries, 4, 9
        ).permute(0, 3, 4, 1, 2, 5)
        relevance = sampled_region.mean(dim=3).add(1e-4)
        relevance = relevance / relevance.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        side_tokens = (
            sampled_detail * relevance.unsqueeze(3)
        ).sum(dim=(-2, -1))

        if self.detail_only_delta:
            delta = self.detail_only_delta_from_tokens(side_tokens)
            if self.capture_diagnostics:
                self.last_attention = relevance.detach()
                self.last_delta_without_detail = (
                    self.detail_only_delta_from_tokens(torch.zeros_like(side_tokens))
                    .detach()
                )
                self.last_delta_without_context = None
            else:
                self.last_attention = None
                self.last_delta_without_detail = None
                self.last_delta_without_context = None
        else:
            context = self.query_context(query_output.detach()).unsqueeze(2).expand(
                -1, -1, 4, -1
            )
            geometry = ref_boxes.detach().unsqueeze(2).expand(-1, -1, 4, -1)
            side_embedding = self.side_embedding.view(1, 1, 4, 4).expand(
                batch, queries, -1, -1
            )
            delta_input = torch.cat(
                (side_tokens, context, geometry, side_embedding), dim=-1
            )
            delta = self.delta_head(delta_input).flatten(-2, -1)
            if self.capture_diagnostics:
                without_detail = torch.cat(
                    (
                        torch.zeros_like(side_tokens),
                        context,
                        geometry,
                        side_embedding,
                    ),
                    dim=-1,
                )
                without_context = torch.cat(
                    (
                        side_tokens,
                        torch.zeros_like(context),
                        geometry,
                        side_embedding,
                    ),
                    dim=-1,
                )
                self.last_attention = relevance.detach()
                self.last_delta_without_detail = self.delta_head(
                    without_detail
                ).flatten(-2, -1).detach()
                self.last_delta_without_context = self.delta_head(
                    without_context
                ).flatten(-2, -1).detach()
            else:
                self.last_attention = None
                self.last_delta_without_detail = None
                self.last_delta_without_context = None
        self.last_side_tokens = side_tokens.detach()
        self.last_delta = delta
        self.last_delta_rms = delta.detach().float().square().mean().sqrt()
        return delta


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method="default",
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention"""
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ""
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer(
            "num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32)
        )

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self.ms_deformable_attn_core = functools.partial(
            deformable_attention_core_func_v2, method=self.method
        )

        self._reset_parameters()

        if method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(
            1, -1, 1
        )
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: List[int],
    ):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list), 2
        )

        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list)
        )
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = (
                sampling_offsets
                * num_points_scale
                * reference_points[:, :, None, :, 2:]
                * self.offset_scale
            )
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )

        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list
        )

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
        cross_attn_method="default",
        layer_scale=None,
    ):
        super(TransformerDecoderLayer, self).__init__()
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points, method=cross_attn_method
        )
        self.dropout2 = nn.Dropout(dropout)

        # gate
        self.gateway = Gate(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(
        self, target, reference_points, value, spatial_shapes, attn_mask=None, query_pos_embed=None
    ):
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed), reference_points, value, spatial_shapes
        )

        target = self.gateway(target, self.dropout2(target2))

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))

        return target


class Gate(nn.Module):
    def __init__(self, d_model):
        super(Gate, self).__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class ThermalEvidenceTokenizer(nn.Module):
    """Convert a thermal feature map into a coordinate-free evidence set.

    No positional encoding is used.  Every spatial descriptor is processed by
    the same point MLP and the final aggregation is a symmetric weighted sum,
    so permuting the HxW feature positions leaves the output unchanged (up to
    floating-point reduction noise).
    """

    def __init__(
        self,
        in_channels,
        hidden_dim,
        num_tokens=8,
        num_heads=4,
        num_iterations=2,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("SDTEC hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.num_slot_tokens = self.num_tokens - 1
        self.num_iterations = int(num_iterations)
        if self.num_tokens < 1 or self.num_iterations < 1:
            raise ValueError("SDTEC needs at least one evidence token")

        self.input_proj = nn.Conv2d(int(in_channels), hidden_dim, 1, bias=False)
        self.point_norm = nn.LayerNorm(hidden_dim)
        self.point_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.global_moment_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.token_type = nn.Parameter(
            torch.empty(1, self.num_tokens, hidden_dim)
        )
        if self.num_slot_tokens > 0:
            self.slots = nn.Parameter(
                torch.empty(1, self.num_slot_tokens, hidden_dim)
            )
            self.slot_norm = nn.LayerNorm(hidden_dim)
            self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.slot_update = nn.GRUCell(hidden_dim, hidden_dim)
            self.slot_ffn_norm = nn.LayerNorm(hidden_dim)
            self.slot_ffn = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.GELU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.slot_self_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, batch_first=True
            )
            self.slot_self_norm = nn.LayerNorm(hidden_dim)
        else:
            # K=1 is the strong TAG-style control: one global evidence vector
            # and no competitive local slots.  Omit the unused slot modules so
            # the ablation has no hidden trainable parameters without gradients.
            self.register_parameter("slots", None)
            self.slot_norm = None
            self.query_proj = None
            self.key_proj = None
            self.value_proj = None
            self.slot_update = None
            self.slot_ffn_norm = None
            self.slot_ffn = None
            self.slot_self_attn = None
            self.slot_self_norm = None
        self.evidence_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        if self.slots is not None:
            nn.init.normal_(self.slots, std=0.02)
        nn.init.normal_(self.token_type, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight.flatten(1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, feature):
        points = self.input_proj(feature).flatten(2).transpose(1, 2)
        points = self.point_norm(points)
        points = points + self.point_mlp(points)
        mean = points.mean(dim=1)
        std = points.var(dim=1, unbiased=False).add(1e-6).sqrt()
        global_moment = self.global_moment_proj(torch.cat((mean, std), dim=-1))

        batch_size, num_points, _ = points.shape
        # The global moment used to be added to every slot.  That common term
        # dominated the small learned differences and made all slots collapse.
        # Keep competitive local evidence and the global statistic as explicit
        # different token types instead.
        if self.num_slot_tokens > 0:
            slots = self.slots.expand(batch_size, -1, -1)
            keys = self.key_proj(points)
            values = self.value_proj(points)
            attention = None
            scale = self.hidden_dim**-0.5
            for _ in range(self.num_iterations):
                queries = self.query_proj(self.slot_norm(slots))
                logits = torch.einsum("bkd,bnd->bkn", queries, keys) * scale
                # First let slots compete for every point, then renormalize each
                # slot over points.  Independent per-slot softmax lets all K slots
                # collapse onto the same evidence and empirically reduced SDTEC to
                # K copies of TAG's scalar summary.
                attention = logits.softmax(dim=1).add(1e-8)
                attention = attention / attention.sum(dim=-1, keepdim=True)
                updates = torch.einsum("bkn,bnd->bkd", attention, values)
                slots = self.slot_update(
                    updates.reshape(-1, self.hidden_dim),
                    slots.reshape(-1, self.hidden_dim),
                ).reshape(batch_size, self.num_slot_tokens, self.hidden_dim)
                slots = slots + self.slot_ffn(self.slot_ffn_norm(slots))

            normalized_slots = self.slot_self_norm(slots)
            attended, _ = self.slot_self_attn(
                normalized_slots,
                normalized_slots,
                normalized_slots,
                need_weights=False,
            )
            slot_tokens = slots + attended
        else:
            attention = points.new_empty(batch_size, 0, num_points)
            slot_tokens = points.new_empty(batch_size, 0, self.hidden_dim)
        tokens = torch.cat((slot_tokens, global_moment.unsqueeze(1)), dim=1)
        tokens = self.output_norm(tokens + self.token_type)
        evidence_logits = self.evidence_head(tokens).squeeze(-1)
        token_quality = evidence_logits.sigmoid()
        presence_logits = torch.logsumexp(evidence_logits, dim=1) - math.log(
            self.num_tokens
        )

        if self.num_slot_tokens > 0:
            eps = torch.finfo(attention.dtype).eps
            spatial_entropy = -(
                attention.clamp_min(eps) * attention.clamp_min(eps).log()
            ).sum(dim=-1)
            spatial_entropy = spatial_entropy / max(
                math.log(max(num_points, 2)), 1.0
            )
        else:
            spatial_entropy = points.new_empty(batch_size, 0)
        binary_uncertainty = 4.0 * token_quality * (1.0 - token_quality)
        global_entropy = torch.zeros_like(binary_uncertainty[:, -1:])
        spatial_entropy = torch.cat((spatial_entropy, global_entropy), dim=1)
        uncertainty = 0.5 * (spatial_entropy + binary_uncertainty)
        return {
            "tokens": tokens,
            "token_quality": token_quality,
            "uncertainty": uncertainty.clamp(0.0, 1.0),
            "presence_logits": presence_logits,
            "slot_attention": attention,
        }


class ThermalCandidateTokenizer(nn.Module):
    """Select target-aware thermal evidence and discard its source positions.

    M3 reuses the encoder proposal scorer from an independently supervised
    thermal detector. Spatial indices are used only to select descriptors
    inside the thermal stream; the RGB interface receives only score-sorted
    descriptors and their confidence.
    """

    def __init__(self, hidden_dim, num_tokens=8):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        if self.num_tokens < 1:
            raise ValueError("M3 needs at least one thermal candidate token")

        # Loaded from the independently trained thermal detector and frozen by
        # DFINE while the coordinate-free adapter learns the fusion task.
        self.target_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.target_norm = nn.LayerNorm(self.hidden_dim)
        self.target_score_head = nn.Linear(self.hidden_dim, 1)

        self.token_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        )
        self.score_proj = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.Tanh(),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.target_proj.weight)
        nn.init.zeros_(self.target_proj.bias)
        nn.init.ones_(self.target_norm.weight)
        nn.init.zeros_(self.target_norm.bias)
        nn.init.xavier_uniform_(self.target_score_head.weight)
        nn.init.zeros_(self.target_score_head.bias)
        nn.init.xavier_uniform_(self.score_proj[0].weight)
        nn.init.zeros_(self.score_proj[0].bias)
        for module in self.token_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features):
        if not isinstance(features, (list, tuple)) or not features:
            raise RuntimeError("M3 candidate tokenizer requires encoded levels")
        points = []
        for feature in features:
            if feature.ndim != 4 or feature.shape[1] != self.hidden_dim:
                raise RuntimeError(
                    "M3 expected [B, hidden_dim, H, W] thermal features, got "
                    f"{tuple(feature.shape)}"
                )
            points.append(feature.flatten(2).transpose(1, 2))
        points = torch.cat(points, dim=1)
        if points.shape[1] < self.num_tokens:
            raise RuntimeError(
                f"M3 requested {self.num_tokens} candidates from only "
                f"{points.shape[1]} thermal descriptors"
            )

        target_memory = self.target_norm(self.target_proj(points))
        all_scores = self.target_score_head(target_memory).squeeze(-1)
        top_scores, top_indices = torch.topk(
            all_scores, self.num_tokens, dim=1, sorted=True
        )
        selected = target_memory.gather(
            1, top_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        )
        tokens = self.output_norm(
            selected
            + self.token_proj(selected)
            + self.score_proj(top_scores.unsqueeze(-1))
        )
        quality = top_scores.sigmoid()
        uncertainty = 4.0 * quality * (1.0 - quality)
        return {
            "tokens": tokens,
            "token_quality": quality,
            "uncertainty": uncertainty.clamp(0.0, 1.0),
            "presence_logits": top_scores[:, 0],
            "candidate_scores": top_scores,
            # Kept only for thermal-side diagnostics; no RGB module consumes
            # the source indices or derives a cross-modal coordinate from them.
            "candidate_indices": top_indices,
        }


class ThermalQueryCoupler(nn.Module):
    """Let RGB queries retrieve coordinate-free thermal evidence by content."""

    def __init__(
        self,
        hidden_dim,
        num_heads=4,
        max_residual_scale=0.05,
        use_reliability=True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("SDTEC hidden_dim must be divisible by num_heads")
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        gate_input_dim = 3 * hidden_dim + 3
        self.channel_gate = nn.Sequential(
            nn.Linear(gate_input_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.safety_gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.use_reliability = bool(use_reliability)
        self.max_residual_scale = float(max_residual_scale)
        if not 0.0 < self.max_residual_scale <= 0.25:
            raise ValueError("SDTEC max_residual_scale must be in (0, 0.25]")
        # Exact RGB identity at construction.  Presence supervision trains the
        # tokenizer immediately; detection gradients first open this residual
        # scale and then reach the deeper coupling parameters.
        self.residual_scale = nn.Parameter(torch.zeros(()))
        # Plain float by design: schedules are runtime protocol state and must
        # not become a checkpoint parameter.  Test-only inference defaults to
        # the fully opened branch.
        self.fusion_progress = 1.0

    def set_fusion_progress(self, progress):
        progress = float(progress)
        if not 0.0 <= progress <= 1.0:
            raise ValueError("SDTEC fusion progress must be in [0, 1]")
        self.fusion_progress = progress

    def forward(self, rgb_queries, tokens, quality, uncertainty, presence):
        if self.use_reliability:
            token_strength = (quality * (1.0 - uncertainty)).unsqueeze(-1)
            weighted_tokens = self.token_norm(tokens) * token_strength
        else:
            weighted_tokens = self.token_norm(tokens)
        query = self.query_norm(rgb_queries)
        context, _ = self.cross_attn(
            query, weighted_tokens, weighted_tokens, need_weights=False
        )
        disagreement = (query - self.output_norm(context)).abs()

        query_count = rgb_queries.shape[1]
        if self.use_reliability:
            quality_mean = quality.mean(dim=1, keepdim=True).expand(
                -1, query_count
            )
            uncertainty_mean = uncertainty.mean(dim=1, keepdim=True).expand(
                -1, query_count
            )
            presence_prob = presence.sigmoid().unsqueeze(1).expand(
                -1, query_count
            )
        else:
            quality_mean = rgb_queries.new_ones(
                rgb_queries.shape[0], query_count
            )
            uncertainty_mean = rgb_queries.new_zeros(
                rgb_queries.shape[0], query_count
            )
            presence_prob = rgb_queries.new_ones(
                rgb_queries.shape[0], query_count
            )
        scalars = torch.stack(
            (quality_mean, uncertainty_mean, presence_prob), dim=-1
        )
        gate_input = torch.cat((query, context, disagreement, scalars), dim=-1)
        channel_gate = self.channel_gate(gate_input).sigmoid()
        safety_gate = self.safety_gate(gate_input).sigmoid()
        if self.use_reliability:
            safety_gate = safety_gate * presence_prob.unsqueeze(-1)
            safety_gate = safety_gate * (1.0 - uncertainty_mean.unsqueeze(-1))
        gate = channel_gate * safety_gate
        residual = gate * self.context_proj(context)
        bounded_scale = self.max_residual_scale * torch.tanh(
            self.residual_scale
        )
        output = rgb_queries + self.fusion_progress * bounded_scale * residual
        return output, gate.mean(dim=(1, 2))


class ThermalLogitCalibrator(nn.Module):
    """Use coordinate-free thermal evidence to calibrate class logits only.

    M2 deliberately leaves decoder queries, reference points and box heads
    untouched.  The RGB query retrieves one of several unordered thermal
    evidence tokens, then produces a directly bounded correction to the final
    class logit.  The last ``protected_tail_queries`` (HRQS queries in the
    current C+D model) receive an exact zero correction.
    """

    def __init__(
        self,
        hidden_dim,
        num_classes,
        num_heads=4,
        max_logit_delta=0.25,
        protected_tail_queries=0,
        use_reliability=True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("M2 hidden_dim must be divisible by num_heads")
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        gate_input_dim = 3 * hidden_dim + 4
        self.safety_gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, int(num_classes)),
        )
        self.use_reliability = bool(use_reliability)
        self.max_logit_delta = float(max_logit_delta)
        self.protected_tail_queries = int(protected_tail_queries)
        if not 0.0 < self.max_logit_delta <= 1.0:
            raise ValueError("M2 max_logit_delta must be in (0, 1]")
        if self.protected_tail_queries < 0:
            raise ValueError("M2 protected_tail_queries must be non-negative")
        self.fusion_progress = 1.0

        # Exact C+D identity at construction.  The final projection receives
        # gradients on the first step; the tokenizer and attention receive
        # gradients as soon as that projection moves away from zero.
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.safety_gate[-1].weight)
        nn.init.constant_(self.safety_gate[-1].bias, -2.0)

    def set_fusion_progress(self, progress):
        progress = float(progress)
        if not 0.0 <= progress <= 1.0:
            raise ValueError("M2 fusion progress must be in [0, 1]")
        self.fusion_progress = progress

    def forward(
        self,
        rgb_queries,
        base_logits,
        tokens,
        quality,
        uncertainty,
        presence,
        thermal_valid=None,
    ):
        if tokens.shape[1] < 2:
            raise RuntimeError(
                "M2 logit calibration needs at least two thermal evidence tokens"
            )
        if rgb_queries.shape[:2] != base_logits.shape[:2]:
            raise RuntimeError(
                "M2 query/logit shape mismatch: "
                f"queries={tuple(rgb_queries.shape)} logits={tuple(base_logits.shape)}"
            )

        query = self.query_norm(rgb_queries)
        normalized_tokens = self.token_norm(tokens)
        if self.use_reliability:
            token_strength = (quality * (1.0 - uncertainty)).unsqueeze(-1)
            normalized_tokens = normalized_tokens * token_strength
        context, attention = self.cross_attn(
            query,
            normalized_tokens,
            normalized_tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        context = self.context_norm(context)
        disagreement = (query - context).abs()

        query_count = rgb_queries.shape[1]
        quality_mean = quality.mean(dim=1, keepdim=True).expand(-1, query_count)
        uncertainty_mean = uncertainty.mean(dim=1, keepdim=True).expand(
            -1, query_count
        )
        presence_prob = presence.sigmoid().unsqueeze(1).expand(-1, query_count)
        base_confidence = base_logits.detach().sigmoid().amax(
            dim=-1, keepdim=False
        )
        scalars = torch.stack(
            (
                quality_mean,
                uncertainty_mean,
                presence_prob,
                base_confidence,
            ),
            dim=-1,
        )
        gate_input = torch.cat((query, context, disagreement, scalars), dim=-1)
        gate = self.safety_gate(gate_input).sigmoid()
        if self.use_reliability:
            gate = gate * presence_prob.unsqueeze(-1)
            gate = gate * (1.0 - uncertainty_mean.unsqueeze(-1))

        raw_delta = self.delta_head(
            torch.cat((query, context, disagreement), dim=-1)
        )
        delta = self.max_logit_delta * torch.tanh(raw_delta) * gate
        delta = delta * self.fusion_progress

        if thermal_valid is not None:
            if thermal_valid.ndim != 1 or thermal_valid.shape[0] != delta.shape[0]:
                raise RuntimeError(
                    "M2 thermal_valid must have shape [batch], got "
                    f"{tuple(thermal_valid.shape)}"
                )
            delta = delta * thermal_valid.to(delta).view(-1, 1, 1)

        protected_count = min(self.protected_tail_queries, delta.shape[1])
        if protected_count > 0:
            role_mask = delta.new_ones((1, delta.shape[1], 1))
            role_mask[:, -protected_count:] = 0.0
            delta = delta * role_mask

        calibrated_logits = base_logits + delta
        diagnostics = {
            "delta": delta,
            "gate": gate.squeeze(-1),
            "attention": attention,
        }
        return calibrated_logits, diagnostics


class ThermalSoftAlignedLogitCalibrator(nn.Module):
    """Read local thermal evidence around an RGB-anchored soft alignment.

    MA1 keeps RGB boxes and decoder queries untouched.  A train-fitted affine
    prior maps each detached RGB box centre into thermal coordinates, then a
    bounded deformable reader searches nearby S16/S32 features.  The class
    correction is generated only from a multiplicative RGB/thermal interaction;
    an all-zero thermal image is also masked explicitly, so no query-only or
    bias-only shortcut can reproduce the correction.
    """

    def __init__(
        self,
        hidden_dim,
        num_classes,
        num_heads=4,
        num_levels=2,
        num_points=4,
        search_radius=0.10,
        affine_init=None,
        affine_delta_scale=0.05,
        max_logit_delta=0.15,
        protected_tail_queries=0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_levels = int(num_levels)
        self.num_points = int(num_points)
        self.search_radius = float(search_radius)
        self.affine_delta_scale = float(affine_delta_scale)
        self.max_logit_delta = float(max_logit_delta)
        self.protected_tail_queries = int(protected_tail_queries)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("MA1 hidden_dim must be divisible by num_heads")
        if self.num_levels < 1 or self.num_points < 1:
            raise ValueError("MA1 needs at least one level and sampling point")
        if not 0.0 < self.search_radius <= 0.25:
            raise ValueError("MA1 search_radius must be in (0, 0.25]")
        if not 0.0 <= self.affine_delta_scale <= 0.25:
            raise ValueError("MA1 affine_delta_scale must be in [0, 0.25]")
        if not 0.0 < self.max_logit_delta <= 1.0:
            raise ValueError("MA1 max_logit_delta must be in (0, 1]")

        if affine_init is None:
            affine_init = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        affine_tensor = torch.as_tensor(affine_init, dtype=torch.float32)
        if affine_tensor.numel() != 6:
            raise ValueError("MA1 affine_init must contain six values")
        self.register_buffer("affine_base", affine_tensor.reshape(3, 2))
        self.affine_delta = nn.Parameter(torch.zeros(3, 2))

        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.query_content_proj = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        total_points = (
            self.num_heads * self.num_levels * self.num_points
        )
        self.sampling_offsets = nn.Linear(
            self.hidden_dim, total_points * 2
        )
        self.attention_weights = nn.Linear(self.hidden_dim, total_points)
        self.context_proj = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.gate_head = nn.Linear(self.hidden_dim, 1, bias=False)
        self.delta_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_dim, int(num_classes), bias=False),
        )
        self.fusion_progress = 1.0
        self.last_diagnostics = None
        self._reset_parameters()

    def _reset_parameters(self):
        for module in (
            self.value_proj,
            self.query_content_proj,
            self.context_proj,
            self.gate_head,
            self.delta_head[0],
        ):
            nn.init.xavier_uniform_(module.weight)
        # The C+D start is exactly unchanged.  Detection gradients first train
        # this final projection, then reach the alignment reader.
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)

        # Start with a small radial search rather than a single zero-offset
        # sample.  tanh keeps every learned residual inside search_radius.
        bias = torch.zeros(
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
            dtype=self.sampling_offsets.bias.dtype,
        )
        for head in range(self.num_heads):
            head_angle = 2.0 * math.pi * head / self.num_heads
            for point in range(self.num_points):
                angle = head_angle + 2.0 * math.pi * point / self.num_points
                fraction = 0.5 + 0.5 * (point + 1) / self.num_points
                target = 0.75 * fraction
                raw_radius = math.atanh(min(target, 0.95))
                bias[head, :, point, 0] = raw_radius * math.cos(angle)
                bias[head, :, point, 1] = raw_radius * math.sin(angle)
        self.sampling_offsets.bias.data.copy_(bias.flatten())

    def set_fusion_progress(self, progress):
        progress = float(progress)
        if not 0.0 <= progress <= 1.0:
            raise ValueError("MA1 fusion progress must be in [0, 1]")
        self.fusion_progress = progress

    def _effective_affine(self):
        return self.affine_base + self.affine_delta_scale * torch.tanh(
            self.affine_delta
        )

    def _thermal_value_levels(self, thermal_features):
        if not isinstance(thermal_features, (list, tuple)):
            raise RuntimeError("MA1 thermal_features must be a list or tuple")
        if len(thermal_features) != self.num_levels:
            raise RuntimeError(
                f"MA1 expected {self.num_levels} thermal levels, got "
                f"{len(thermal_features)}"
            )
        value_levels = []
        spatial_shapes = []
        head_dim = self.hidden_dim // self.num_heads
        for feature in thermal_features:
            if feature.ndim != 4 or feature.shape[1] != self.hidden_dim:
                raise RuntimeError(
                    "MA1 expected [B, hidden_dim, H, W] thermal features, "
                    f"got {tuple(feature.shape)}"
                )
            batch, _, height, width = feature.shape
            flattened = feature.flatten(2).transpose(1, 2)
            projected = self.value_proj(flattened)
            value_levels.append(
                projected.reshape(
                    batch, height * width, self.num_heads, head_dim
                ).permute(0, 2, 3, 1)
            )
            spatial_shapes.append((height, width))
        return tuple(value_levels), spatial_shapes

    def forward(
        self,
        rgb_queries,
        base_logits,
        rgb_box_centres,
        thermal_features,
        thermal_content_mask,
    ):
        if rgb_queries.shape[:2] != base_logits.shape[:2]:
            raise RuntimeError("MA1 query/logit shapes do not match")
        if rgb_box_centres.shape != (*rgb_queries.shape[:2], 2):
            raise RuntimeError(
                "MA1 RGB box centres must have shape [B, Q, 2], got "
                f"{tuple(rgb_box_centres.shape)}"
            )
        if thermal_content_mask is None:
            raise RuntimeError("MA1 requires an explicit thermal content mask")
        if (
            thermal_content_mask.ndim != 1
            or thermal_content_mask.shape[0] != rgb_queries.shape[0]
        ):
            raise RuntimeError("MA1 thermal_content_mask must have shape [B]")

        query = self.query_norm(rgb_queries)
        homogeneous = torch.cat(
            (rgb_box_centres.detach(), torch.ones_like(rgb_box_centres[..., :1])),
            dim=-1,
        )
        mapped_centres = torch.matmul(
            homogeneous, self._effective_affine().to(homogeneous)
        )

        batch, query_count = query.shape[:2]
        offsets = self.sampling_offsets(query).reshape(
            batch,
            query_count,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        residual_offsets = torch.tanh(offsets) * self.search_radius
        sampling_locations = (
            mapped_centres[:, :, None, None, None, :] + residual_offsets
        ).flatten(3, 4)

        attention = self.attention_weights(query).reshape(
            batch,
            query_count,
            self.num_heads,
            self.num_levels * self.num_points,
        )
        attention = attention.softmax(dim=-1)
        value_levels, spatial_shapes = self._thermal_value_levels(
            thermal_features
        )
        context = deformable_attention_core_func_v2(
            value_levels,
            spatial_shapes,
            sampling_locations,
            attention,
            [self.num_points] * self.num_levels,
        )
        context = self.context_proj(context)
        interaction = self.query_content_proj(query) * context
        gate_logits = self.gate_head(interaction)
        gate = gate_logits.sigmoid()
        raw_delta = self.delta_head(interaction)
        delta = self.max_logit_delta * torch.tanh(raw_delta) * gate
        content_mask = thermal_content_mask.to(delta).view(-1, 1, 1)
        delta = delta * content_mask * self.fusion_progress

        protected_count = min(self.protected_tail_queries, delta.shape[1])
        if protected_count > 0:
            role_mask = delta.new_ones((1, delta.shape[1], 1))
            role_mask[:, -protected_count:] = 0.0
            delta = delta * role_mask

        # Average the actually used residual over points and heads.  This is a
        # diagnostic of learned alignment, not an input to the detector.
        flat_residual = residual_offsets.flatten(3, 4)
        effective_offset = (
            flat_residual * attention.unsqueeze(-1)
        ).sum(dim=3).mean(dim=2)
        eps = torch.finfo(attention.dtype).eps
        attention_entropy = -(
            attention.clamp_min(eps) * attention.clamp_min(eps).log()
        ).sum(dim=-1) / math.log(max(self.num_levels * self.num_points, 2))
        diagnostics = {
            "delta": delta,
            "gate": gate.squeeze(-1),
            "gate_logits": gate_logits.squeeze(-1),
            "mapped_centres": mapped_centres,
            "effective_offset": effective_offset,
            "aligned_centres": mapped_centres + effective_offset,
            "max_abs_residual_offset": residual_offsets.abs().amax(),
            "attention_entropy": attention_entropy,
            "thermal_content_mask": thermal_content_mask,
            "effective_affine": self._effective_affine(),
        }
        self.last_diagnostics = {
            key: value.detach() for key, value in diagnostics.items()
        }
        return base_logits + delta, diagnostics


class SharedK1ThermalQueryCoupler(nn.Module):
    """Shared lightweight coupler for a single coordinate-free thermal token.

    With exactly one key/value token, cross-attention has no competition: the
    softmax weight is always one for every RGB query.  The value and output
    projections can therefore be folded into one linear transformation.  The
    comparatively expensive query-aware reliability gates are retained, but
    shared by all decoder layers.  Each layer keeps an independent bounded
    residual scale so it can decide how strongly to use the shared evidence.
    """

    def __init__(
        self,
        hidden_dim,
        num_layers,
        max_residual_scale=0.05,
        use_reliability=True,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.context_linear = nn.Linear(hidden_dim, hidden_dim)
        gate_input_dim = 3 * hidden_dim + 3
        self.channel_gate = nn.Sequential(
            nn.Linear(gate_input_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.safety_gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.use_reliability = bool(use_reliability)
        self.max_residual_scale = float(max_residual_scale)
        if not 0.0 < self.max_residual_scale <= 0.25:
            raise ValueError("SDTEC max_residual_scale must be in (0, 0.25]")
        num_layers = int(num_layers)
        if num_layers < 1:
            raise ValueError("Shared K=1 SDTEC needs at least one decoder layer")
        self.layer_residual_scales = nn.Parameter(torch.zeros(num_layers))
        self.fusion_progress = 1.0

    def set_fusion_progress(self, progress):
        progress = float(progress)
        if not 0.0 <= progress <= 1.0:
            raise ValueError("SDTEC fusion progress must be in [0, 1]")
        self.fusion_progress = progress

    def effective_scales(self):
        return (
            self.layer_residual_scales.tanh()
            * self.max_residual_scale
            * self.fusion_progress
        )

    def forward_layer(
        self,
        layer_index,
        rgb_queries,
        tokens,
        quality,
        uncertainty,
        presence,
    ):
        if tokens.shape[1] != 1:
            raise RuntimeError(
                "Shared K=1 SDTEC requires exactly one thermal evidence token, "
                f"got {tokens.shape[1]}"
            )
        if not 0 <= int(layer_index) < self.layer_residual_scales.numel():
            raise IndexError(f"invalid SDTEC decoder layer index {layer_index}")

        if self.use_reliability:
            token_strength = (quality * (1.0 - uncertainty)).unsqueeze(-1)
            weighted_token = self.token_norm(tokens) * token_strength
        else:
            weighted_token = self.token_norm(tokens)

        query = self.query_norm(rgb_queries)
        # A single key/value receives attention probability one for every
        # query.  Expand its transformed value instead of computing Q/K and a
        # one-element softmax on every decoder layer.
        context = self.context_linear(weighted_token).expand(
            -1, rgb_queries.shape[1], -1
        )
        disagreement = (query - self.output_norm(context)).abs()

        query_count = rgb_queries.shape[1]
        if self.use_reliability:
            quality_mean = quality.mean(dim=1, keepdim=True).expand(
                -1, query_count
            )
            uncertainty_mean = uncertainty.mean(dim=1, keepdim=True).expand(
                -1, query_count
            )
            presence_prob = presence.sigmoid().unsqueeze(1).expand(
                -1, query_count
            )
        else:
            quality_mean = rgb_queries.new_ones(
                rgb_queries.shape[0], query_count
            )
            uncertainty_mean = rgb_queries.new_zeros(
                rgb_queries.shape[0], query_count
            )
            presence_prob = rgb_queries.new_ones(
                rgb_queries.shape[0], query_count
            )
        scalars = torch.stack(
            (quality_mean, uncertainty_mean, presence_prob), dim=-1
        )
        gate_input = torch.cat((query, context, disagreement, scalars), dim=-1)
        channel_gate = self.channel_gate(gate_input).sigmoid()
        safety_gate = self.safety_gate(gate_input).sigmoid()
        if self.use_reliability:
            safety_gate = safety_gate * presence_prob.unsqueeze(-1)
            safety_gate = safety_gate * (1.0 - uncertainty_mean.unsqueeze(-1))
        gate = channel_gate * safety_gate
        residual = gate * self.context_proj(context)
        bounded_scale = self.max_residual_scale * torch.tanh(
            self.layer_residual_scales[int(layer_index)]
        )
        output = rgb_queries + self.fusion_progress * bounded_scale * residual
        return output, gate.mean(dim=(1, 2))


class ThermalSamePositionCoupler(nn.Module):
    """Negative control that assumes RGB and thermal S32 coordinates align.

    This branch deliberately transfers a thermal descriptor only to the RGB
    descriptor at the same flattened S32 index.  It is not an M1 candidate;
    it exists to test whether removing cross-modal coordinates is necessary on
    the current unregistered data.
    """

    def __init__(self, hidden_dim, max_residual_scale=0.05):
        super().__init__()
        gate_input_dim = 3 * hidden_dim
        self.rgb_norm = nn.LayerNorm(hidden_dim)
        self.thermal_norm = nn.LayerNorm(hidden_dim)
        self.channel_gate = nn.Sequential(
            nn.Linear(gate_input_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))
        self.max_residual_scale = float(max_residual_scale)
        if not 0.0 < self.max_residual_scale <= 0.25:
            raise ValueError(
                "SDTEC same-position max_residual_scale must be in (0, 0.25]"
            )
        self.fusion_progress = 1.0

    def set_fusion_progress(self, progress):
        progress = float(progress)
        if not 0.0 <= progress <= 1.0:
            raise ValueError("SDTEC fusion progress must be in [0, 1]")
        self.fusion_progress = progress

    def forward(self, rgb_memory, thermal_memory):
        if rgb_memory.shape != thermal_memory.shape:
            raise RuntimeError(
                "Same-position control requires identical S32 tensor shapes, got "
                f"RGB={tuple(rgb_memory.shape)} thermal={tuple(thermal_memory.shape)}"
            )
        rgb = self.rgb_norm(rgb_memory)
        thermal = self.thermal_norm(thermal_memory)
        disagreement = (rgb - thermal).abs()
        gate = self.channel_gate(
            torch.cat((rgb, thermal, disagreement), dim=-1)
        ).sigmoid()
        residual = gate * self.context_proj(thermal)
        bounded_scale = self.max_residual_scale * torch.tanh(
            self.residual_scale
        )
        output = rgb_memory + self.fusion_progress * bounded_scale * residual
        return output, gate.mean(dim=(1, 2))


class Integral(nn.Module):
    """
    A static layer that calculates integral results from a distribution.

    This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`,
    where Pr(n) is the softmax probability vector representing the discrete
    distribution, and W(n) is the non-uniform Weighting Function.

    Args:
        reg_max (int): Max number of the discrete bins. Default is 32.
                       It can be adjusted based on the dataset or task requirements.
    """

    def __init__(self, reg_max=32):
        super(Integral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class QueryRankAdaptiveScoreBias(nn.Module):
    """Calibrate decoder scores with the encoder Top-K rank of each query.

    Denoising queries, when present, are prepended to the ordinary detector
    queries.  The rank table is therefore applied only to the final
    ``num_queries`` slots.  A zero table is an exact numerical identity.
    """

    def __init__(self, num_layers, num_queries, num_classes):
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)
        if self.num_layers <= 0 or self.num_queries <= 0 or self.num_classes <= 0:
            raise ValueError("Q-Rank1 dimensions must be positive")
        self.rank_bias = nn.Parameter(
            torch.zeros(self.num_layers, self.num_queries, self.num_classes)
        )
        self.intervention_mode = "learned"
        self.last_normal_query_count = None
        self.last_denoising_query_count = None
        self.last_abs_delta = None

    def forward(self, scores, layer_index):
        if scores.ndim != 3 or scores.shape[-1] != self.num_classes:
            raise ValueError(
                "Q-Rank1 expects [batch, queries, classes] scores, got "
                f"{tuple(scores.shape)}"
            )
        if scores.shape[1] < self.num_queries:
            raise ValueError(
                "Q-Rank1 received fewer queries than its rank table: "
                f"scores={scores.shape[1]}, table={self.num_queries}"
            )
        layer_index = int(layer_index)
        if not 0 <= layer_index < self.num_layers:
            raise ValueError(f"Q-Rank1 layer index is out of range: {layer_index}")
        if self.intervention_mode not in {"learned", "zero"}:
            raise ValueError(
                "Q-Rank1 intervention_mode must be 'learned' or 'zero'"
            )

        denoising_count = scores.shape[1] - self.num_queries
        self.last_normal_query_count = self.num_queries
        self.last_denoising_query_count = denoising_count
        if self.intervention_mode == "zero":
            self.last_abs_delta = scores.new_zeros(())
            return scores

        # Keep the tiny learnable calibration in FP32.  Down-casting a fresh
        # 2e-4-scale update to FP16 before adding it to a roughly unit-scale
        # logit makes the update numerically disappear during AMP training.
        rank_delta = self.rank_bias[layer_index]
        rank_delta = rank_delta.unsqueeze(0).expand(scores.shape[0], -1, -1)
        self.last_abs_delta = rank_delta.detach().abs().mean()
        calibrated = scores[:, denoising_count:] + rank_delta
        if denoising_count == 0:
            return calibrated
        return torch.cat((scores[:, :denoising_count], calibrated), dim=1)


class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max):
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers,
    utilizing attention mechanisms, location quality estimators, and distribution refinement techniques
    to improve bounding box accuracy and robustness.
    """

    def __init__(
        self,
        hidden_dim,
        decoder_layer,
        decoder_layer_wide,
        num_layers,
        num_head,
        reg_max,
        reg_scale,
        up,
        eval_idx=-1,
        layer_scale=2,
    ):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.lqe_layers = nn.ModuleList(
            [copy.deepcopy(LQE(4, 64, 2, reg_max)) for _ in range(num_layers)]
        )

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """
        Preprocess values for MSDeformableAttention.
        """
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList(
            [nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]]
        )

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
        attn_mask=None,
        memory_mask=None,
        dn_meta=None,
        qrl_carrier=None,
        qrl_state=None,
        thermal_tokens=None,
        thermal_quality=None,
        thermal_uncertainty=None,
        thermal_presence=None,
        thermal_couplers=None,
        query_rank_score_bias=None,
    ):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_base_pred_corners = []
        dec_out_refs = []
        dec_out_queries = []
        dec_out_head_offsets = []
        sdtec_gate_by_layer = []
        if not hasattr(self, "project"):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            # TODO Adjust scale if needed for detachable wider layers
            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(
                    memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes
                )
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(
                output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed
            )
            if thermal_couplers is not None:
                if hasattr(thermal_couplers, "forward_layer"):
                    output, layer_gate = thermal_couplers.forward_layer(
                        i,
                        output,
                        thermal_tokens,
                        thermal_quality,
                        thermal_uncertainty,
                        thermal_presence,
                    )
                else:
                    output, layer_gate = thermal_couplers[i](
                        output,
                        thermal_tokens,
                        thermal_quality,
                        thermal_uncertainty,
                        thermal_presence,
                    )
                sdtec_gate_by_layer.append(layer_gate)

            if i == 0:
                # Initial bounding box predictions with inverse sigmoid refinement
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            # Refine bounding box corners using FDR, integrating previous layer's corrections
            base_pred_corners = (
                bbox_head[i](output + output_detach) + pred_corners_undetach
            )
            if qrl_carrier is not None and qrl_state is not None and i == self.eval_idx:
                detail_delta = qrl_carrier(output, ref_points_detach, qrl_state)
                pred_corners = base_pred_corners + detail_delta
            else:
                pred_corners = base_pred_corners
            inter_ref_bbox = distance2bbox(
                ref_points_initial, integral(pred_corners, project), reg_scale
            )

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                # QRL may change the forward localization-quality score, but
                # VFL must not train the localization carrier through LQE.
                lqe_corners = base_pred_corners + (
                    pred_corners - base_pred_corners
                ).detach()
                scores = self.lqe_layers[i](scores, lqe_corners)
                if query_rank_score_bias is not None:
                    scores = query_rank_score_bias(scores, layer_index=i)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_base_pred_corners.append(base_pred_corners)
                dec_out_refs.append(ref_points_initial)
                dec_out_queries.append(output)
                dec_out_head_offsets.append(
                    output_detach
                    if isinstance(output_detach, torch.Tensor)
                    else torch.zeros_like(output)
                )

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        self.last_sdtec_gate_by_layer = (
            torch.stack(sdtec_gate_by_layer)
            if sdtec_gate_by_layer
            else None
        )

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
            torch.stack(dec_out_queries),
            torch.stack(dec_out_base_pred_corners),
            torch.stack(dec_out_head_offsets),
        )


@register()
class DFINETransformer(nn.Module):
    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        qrl_enabled=False,
        qrl_source_channels=256,
        qrl_detail_channels=32,
        qrl_context_dim=16,
        qrl_delta_hidden_dim=64,
        qrl_detail_only_delta=False,
        mdqa_enabled=False,
        mdqa_source_channels=256,
        mdqa_mask_dim=64,
        mdqa_detach_query=False,
        sqmi_enabled=False,
        sqmi_source_channels=256,
        sqmi_mask_dim=64,
        sqmi_topk=64,
        sqmi_max_mix=0.25,
        sqmi_max_box_delta=0.25,
        sqmi_gate_bias=-4.0,
        sqmi_apply_initialization=True,
        qcsr_enabled=False,
        qcsr_s8_channels=256,
        qcsr_s16_channels=512,
        qcsr_dim=64,
        sdtec_enabled=False,
        sdtec_thermal_channels=128,
        sdtec_num_tokens=8,
        sdtec_num_heads=4,
        sdtec_slot_iterations=2,
        sdtec_use_reliability=True,
        sdtec_fusion_mode="coordinate_free",
        sdtec_coupler_mode="independent",
        sdtec_max_residual_scale=0.05,
        sdtec_max_logit_delta=0.25,
        sdtec_alignment_affine_init=None,
        sdtec_alignment_affine_delta_scale=0.05,
        sdtec_alignment_search_radius=0.10,
        sdtec_alignment_levels=2,
        sdtec_alignment_points=4,
        sdtec_protect_hrqs_queries=True,
        sdtec_train_mismatch=True,
        sdtec_warmup_epochs=1,
        sdtec_ramp_epochs=5,
        hrqs_enabled=False,
        hrqs_source_channels=256,
        hrqs_anchor_size=0.025,
        hrqs_num_queries=50,
        sqfr_enabled=False,
        sqfr_source_channels=256,
        sqfr_hidden_dim=64,
        sqfr_num_heads=4,
        sqfr_roi_size=5,
        sqfr_topk=64,
        sqfr_context_scale=1.5,
        sqfr_max_logit_delta=0.25,
        qrank_enabled=False,
    ):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)

        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        self.qrl_enabled = bool(qrl_enabled)
        self.mdqa_enabled = bool(mdqa_enabled)
        self.mdqa_mask_dim = int(mdqa_mask_dim)
        self.mdqa_detach_query = bool(mdqa_detach_query)
        self.sqmi_enabled = bool(sqmi_enabled)
        self.sqmi_apply_initialization = bool(sqmi_apply_initialization)
        self.qcsr_enabled = bool(qcsr_enabled)
        self.qcsr_dim = int(qcsr_dim)
        self.sdtec_enabled = bool(sdtec_enabled)
        self.sdtec_fusion_mode = str(sdtec_fusion_mode)
        self.sdtec_coupler_mode = str(sdtec_coupler_mode)
        self.hrqs_enabled = bool(hrqs_enabled)
        self.hrqs_anchor_size = float(hrqs_anchor_size)
        self.hrqs_num_queries = int(hrqs_num_queries)
        self.sqfr_enabled = bool(sqfr_enabled)
        self.qrank_enabled = bool(qrank_enabled)
        if self.hrqs_enabled and self.sqfr_enabled:
            raise ValueError("SQFR1 replaces HRQS1; the two branches cannot coexist")
        if not 0.0 < self.hrqs_anchor_size < 1.0:
            raise ValueError("hrqs_anchor_size must be between zero and one")
        if self.hrqs_enabled and not 0 < self.hrqs_num_queries < self.num_queries:
            raise ValueError("hrqs_num_queries must be between zero and num_queries")
        if self.sdtec_fusion_mode not in {
            "coordinate_free",
            "same_position",
            "soft_aligned",
        }:
            raise ValueError(
                "sdtec_fusion_mode must be coordinate_free, same_position or "
                "soft_aligned"
            )
        if self.sdtec_coupler_mode not in {
            "independent",
            "shared_k1",
            "logit_calibration",
            "candidate_logit_calibration",
            "soft_aligned_logit",
        }:
            raise ValueError(
                "sdtec_coupler_mode must be 'independent', 'shared_k1' or a "
                "protected logit calibration mode"
            )
        self.sdtec_max_logit_delta = float(sdtec_max_logit_delta)
        self.sdtec_alignment_affine_init = sdtec_alignment_affine_init
        self.sdtec_alignment_affine_delta_scale = float(
            sdtec_alignment_affine_delta_scale
        )
        self.sdtec_alignment_search_radius = float(
            sdtec_alignment_search_radius
        )
        self.sdtec_alignment_levels = int(sdtec_alignment_levels)
        self.sdtec_alignment_points = int(sdtec_alignment_points)
        self.sdtec_protect_hrqs_queries = bool(sdtec_protect_hrqs_queries)
        self.sdtec_train_mismatch = bool(sdtec_train_mismatch)
        if not 0.0 < self.sdtec_max_logit_delta <= 1.0:
            raise ValueError("sdtec_max_logit_delta must be in (0, 1]")
        if self.sdtec_coupler_mode in {
            "logit_calibration",
            "candidate_logit_calibration",
        }:
            if self.sdtec_fusion_mode != "coordinate_free":
                raise ValueError("M2 logit calibration requires coordinate_free tokens")
            if int(sdtec_num_tokens) < 2:
                raise ValueError("M2 logit calibration requires sdtec_num_tokens >= 2")
            if self.sdtec_protect_hrqs_queries and not self.hrqs_enabled:
                raise ValueError(
                    "sdtec_protect_hrqs_queries requires hrqs_enabled=True"
                )
        if self.sdtec_coupler_mode == "soft_aligned_logit":
            if self.sdtec_fusion_mode != "soft_aligned":
                raise ValueError(
                    "soft_aligned_logit requires sdtec_fusion_mode=soft_aligned"
                )
            if self.sdtec_protect_hrqs_queries and not self.hrqs_enabled:
                raise ValueError(
                    "sdtec_protect_hrqs_queries requires hrqs_enabled=True"
                )
        self.sdtec_warmup_epochs = int(sdtec_warmup_epochs)
        self.sdtec_ramp_epochs = int(sdtec_ramp_epochs)
        if self.sdtec_warmup_epochs < 0 or self.sdtec_ramp_epochs < 1:
            raise ValueError(
                "SDTEC warmup_epochs must be non-negative and ramp_epochs positive"
            )
        self.sdtec_fusion_progress = 1.0
        if self.mdqa_mask_dim <= 0:
            raise ValueError("mdqa_mask_dim must be positive")
        if self.qcsr_dim <= 0:
            raise ValueError("qcsr_dim must be positive")

        assert query_select_method in ("default", "one2many", "agnostic"), ""
        assert cross_attn_method in ("default", "discrete"), ""
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
        )
        decoder_layer_wide = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
        )
        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
        )
        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)

        # if num_select_queries != self.num_queries:
        #     layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, activation='gelu')
        #     self.encoder = TransformerEncoder(layer, 1)

        self.enc_output = nn.Sequential(
            OrderedDict(
                [
                    ("proj", nn.Linear(hidden_dim, hidden_dim)),
                    (
                        "norm",
                        nn.LayerNorm(
                            hidden_dim,
                        ),
                    ),
                ]
            )
        )

        if query_select_method == "agnostic":
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        # decoder head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
            + [nn.Linear(scaled_dim, num_classes) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.dec_bbox_head = nn.ModuleList(
            [
                MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3)
                for _ in range(self.eval_idx + 1)
            ]
            + [
                MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )
        self.integral = Integral(self.reg_max)
        self.qrl = (
            QueryAlignedRegionLocalizationCarrier(
                source_channels=qrl_source_channels,
                hidden_dim=hidden_dim,
                reg_max=reg_max,
                detail_channels=qrl_detail_channels,
                context_dim=qrl_context_dim,
                delta_hidden_dim=qrl_delta_hidden_dim,
                detail_only_delta=qrl_detail_only_delta,
            )
            if self.qrl_enabled
            else None
        )

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)
        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters(feat_channels)

        # S-QMI1 is created after the ordinary detector reset so toggling the
        # optional route cannot alter any shared detector initialization.
        self.sqmi = (
            SAMQueryMaskInitializer(
                hidden_dim=hidden_dim,
                source_channels=int(sqmi_source_channels),
                mask_dim=int(sqmi_mask_dim),
                topk=int(sqmi_topk),
                max_mix=float(sqmi_max_mix),
                max_box_delta=float(sqmi_max_box_delta),
                gate_bias=float(sqmi_gate_bias),
                apply_initialization=self.sqmi_apply_initialization,
            )
            if self.sqmi_enabled
            else None
        )
        self.last_sqmi_diagnostics = None
        self.last_sqmi_query_embeddings = None
        self.last_sqmi_pixel_features = None

        # Build this optional head after every ordinary detector parameter has
        # been initialized so enabling MDQA cannot perturb A00's shared
        # initialization under the same random seed.
        if self.mdqa_enabled:
            self.mdqa_query_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_dim, self.mdqa_mask_dim),
            )
            groups = min(8, self.mdqa_mask_dim)
            while self.mdqa_mask_dim % groups != 0:
                groups -= 1
            self.mdqa_pixel_proj = nn.Sequential(
                nn.Conv2d(
                    int(mdqa_source_channels),
                    self.mdqa_mask_dim,
                    kernel_size=1,
                    bias=False,
                ),
                nn.GroupNorm(groups, self.mdqa_mask_dim),
            )
        else:
            self.mdqa_query_proj = None
            self.mdqa_pixel_proj = None

        # QCSR heads are training-only.  They are deliberately created after
        # the detector reset so enabling REP3 preserves every shared A00
        # initialization under the same random seed.
        if self.qcsr_enabled:
            self.qcsr_query_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_dim, self.qcsr_dim),
            )
            groups = min(8, self.qcsr_dim)
            while self.qcsr_dim % groups != 0:
                groups -= 1
            self.qcsr_teacher_proj = nn.Sequential(
                nn.Conv2d(int(qcsr_s8_channels), self.qcsr_dim, 1, bias=False),
                nn.GroupNorm(groups, self.qcsr_dim),
            )
            self.qcsr_student_proj = nn.Sequential(
                nn.Conv2d(int(qcsr_s16_channels), self.qcsr_dim, 1, bias=False),
                nn.GroupNorm(groups, self.qcsr_dim),
            )
        else:
            self.qcsr_query_proj = None
            self.qcsr_teacher_proj = None
            self.qcsr_student_proj = None

        # SDTEC is created after the ordinary detector reset so enabling the
        # branch cannot perturb any shared RGB parameter initialization.
        if self.sdtec_enabled:
            if layer_scale != 1:
                raise ValueError("SDTEC currently requires layer_scale=1")
            if self.sdtec_fusion_mode == "soft_aligned":
                self.sdtec_tokenizer = None
                self.sdtec_candidate_tokenizer = None
                self.sdtec_couplers = None
                self.sdtec_logit_calibrator = None
                self.sdtec_spatial_coupler = None
                self.sdtec_aligned_calibrator = (
                    ThermalSoftAlignedLogitCalibrator(
                        hidden_dim=hidden_dim,
                        num_classes=self.num_classes,
                        num_heads=int(sdtec_num_heads),
                        num_levels=self.sdtec_alignment_levels,
                        num_points=self.sdtec_alignment_points,
                        search_radius=self.sdtec_alignment_search_radius,
                        affine_init=self.sdtec_alignment_affine_init,
                        affine_delta_scale=self.sdtec_alignment_affine_delta_scale,
                        max_logit_delta=self.sdtec_max_logit_delta,
                        protected_tail_queries=(
                            self.hrqs_num_queries
                            if self.sdtec_protect_hrqs_queries
                            else 0
                        ),
                    )
                )
            elif self.sdtec_fusion_mode == "coordinate_free":
                if self.sdtec_coupler_mode == "candidate_logit_calibration":
                    self.sdtec_tokenizer = None
                    self.sdtec_candidate_tokenizer = ThermalCandidateTokenizer(
                        hidden_dim=hidden_dim,
                        num_tokens=int(sdtec_num_tokens),
                    )
                else:
                    self.sdtec_tokenizer = ThermalEvidenceTokenizer(
                        in_channels=int(sdtec_thermal_channels),
                        hidden_dim=hidden_dim,
                        num_tokens=int(sdtec_num_tokens),
                        num_heads=int(sdtec_num_heads),
                        num_iterations=int(sdtec_slot_iterations),
                    )
                    self.sdtec_candidate_tokenizer = None
                if self.sdtec_coupler_mode == "shared_k1":
                    if int(sdtec_num_tokens) != 1:
                        raise ValueError(
                            "shared_k1 SDTEC requires sdtec_num_tokens=1"
                        )
                    self.sdtec_couplers = SharedK1ThermalQueryCoupler(
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        max_residual_scale=float(sdtec_max_residual_scale),
                        use_reliability=bool(sdtec_use_reliability),
                    )
                    self.sdtec_logit_calibrator = None
                elif self.sdtec_coupler_mode in {
                    "logit_calibration",
                    "candidate_logit_calibration",
                }:
                    self.sdtec_couplers = None
                    self.sdtec_logit_calibrator = ThermalLogitCalibrator(
                        hidden_dim=hidden_dim,
                        num_classes=self.num_classes,
                        num_heads=int(sdtec_num_heads),
                        max_logit_delta=self.sdtec_max_logit_delta,
                        protected_tail_queries=(
                            self.hrqs_num_queries
                            if self.sdtec_protect_hrqs_queries
                            else 0
                        ),
                        use_reliability=bool(sdtec_use_reliability),
                    )
                else:
                    self.sdtec_couplers = nn.ModuleList(
                        [
                            ThermalQueryCoupler(
                                hidden_dim,
                                int(sdtec_num_heads),
                                float(sdtec_max_residual_scale),
                                bool(sdtec_use_reliability),
                            )
                            for _ in range(num_layers)
                        ]
                    )
                    self.sdtec_logit_calibrator = None
                self.sdtec_spatial_coupler = None
                self.sdtec_aligned_calibrator = None
            else:
                self.sdtec_tokenizer = None
                self.sdtec_candidate_tokenizer = None
                self.sdtec_couplers = None
                self.sdtec_logit_calibrator = None
                self.sdtec_spatial_coupler = ThermalSamePositionCoupler(
                    hidden_dim=hidden_dim,
                    max_residual_scale=float(sdtec_max_residual_scale),
                )
                self.sdtec_aligned_calibrator = None
        else:
            self.sdtec_tokenizer = None
            self.sdtec_candidate_tokenizer = None
            self.sdtec_couplers = None
            self.sdtec_logit_calibrator = None
            self.sdtec_spatial_coupler = None
            self.sdtec_aligned_calibrator = None

        # Build HRQS after the ordinary detector reset.  Enabling the optional
        # branch therefore preserves every shared D-FINE/GQ1 initialization
        # under the same random seed.
        self.hrqs_adapter = (
            HighResolutionQueryAdmission(
                in_channels=int(hrqs_source_channels),
                hidden_dim=hidden_dim,
            )
            if self.hrqs_enabled
            else None
        )
        self.last_hrqs_selected_count = None
        self.last_hrqs_selected_ratio = None
        self.sqfr_refiner = (
            SparseQueryHighResolutionRefiner(
                in_channels=int(sqfr_source_channels),
                query_dim=hidden_dim,
                hidden_dim=int(sqfr_hidden_dim),
                num_heads=int(sqfr_num_heads),
                roi_size=int(sqfr_roi_size),
                topk=int(sqfr_topk),
                context_scale=float(sqfr_context_scale),
                max_logit_delta=float(sqfr_max_logit_delta),
            )
            if self.sqfr_enabled
            else None
        )
        # Q-Rank1 is built after the shared detector reset.  Its strictly zero
        # initialization preserves every existing C-only output and consumes
        # no random numbers that could perturb the control initialization.
        self.qrank_score_bias = (
            QueryRankAdaptiveScoreBias(
                num_layers=num_layers,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
            )
            if self.qrank_enabled
            else None
        )

    def set_training_epoch(self, epoch):
        """Protect the pretrained RGB detector while the new reader settles.

        Epochs before ``sdtec_warmup_epochs`` are an exact RGB-only forward
        path.  The bounded residual is then opened linearly.  Test-only loads
        do not call this method and therefore use the fully opened branch.
        """
        if not self.sdtec_enabled:
            return
        epoch = int(epoch)
        if epoch < self.sdtec_warmup_epochs:
            progress = 0.0
        else:
            progress = min(
                1.0,
                (epoch - self.sdtec_warmup_epochs + 1)
                / self.sdtec_ramp_epochs,
            )
        self.sdtec_fusion_progress = float(progress)
        if self.sdtec_couplers is not None:
            if hasattr(self.sdtec_couplers, "set_fusion_progress"):
                self.sdtec_couplers.set_fusion_progress(progress)
            else:
                for coupler in self.sdtec_couplers:
                    coupler.set_fusion_progress(progress)
        if self.sdtec_spatial_coupler is not None:
            self.sdtec_spatial_coupler.set_fusion_progress(progress)
        if self.sdtec_logit_calibrator is not None:
            self.sdtec_logit_calibrator.set_fusion_progress(progress)
        if self.sdtec_aligned_calibrator is not None:
            self.sdtec_aligned_calibrator.set_fusion_progress(progress)

    def convert_to_deploy(self):
        self.mdqa_query_proj = None
        self.mdqa_pixel_proj = None
        self.mdqa_enabled = False
        self.qcsr_query_proj = None
        self.qcsr_teacher_proj = None
        self.qcsr_student_proj = None
        self.qcsr_enabled = False
        self.dec_score_head = nn.ModuleList(
            [nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]]
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity()
                for i in range(len(self.dec_bbox_head))
            ]
        )

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, "layers"):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("conv", nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                                (
                                    "norm",
                                    nn.BatchNorm2d(
                                        self.hidden_dim,
                                    ),
                                ),
                            ]
                        )
                    )
                )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            if in_channels == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(
                                        in_channels, self.hidden_dim, 3, 2, padding=1, bias=False
                                    ),
                                ),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )
                in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(
        self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"
    ):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(
        self,
        memory: torch.Tensor,
        spatial_shapes,
        denoising_logits=None,
        denoising_bbox_unact=None,
        hrqs_source=None,
        sqmi_source=None,
    ):
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        # memory = torch.where(valid_mask, memory, 0)
        # TODO fix type error for onnx export
        memory = valid_mask.to(memory.dtype) * memory

        output_memory: torch.Tensor = self.enc_output(memory)
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)

        ordinary_query_count = self.num_queries
        high_resolution_memory = None
        high_resolution_logits = None
        high_resolution_anchors = None
        if self.hrqs_enabled:
            ordinary_query_count -= self.hrqs_num_queries
            if hrqs_source is None:
                raise RuntimeError("HRQS is enabled but no S8 source was provided")
            high_resolution = self.hrqs_adapter(hrqs_source)
            high_resolution_memory = high_resolution.flatten(2).permute(0, 2, 1)
            high_resolution_memory = self.enc_output(high_resolution_memory)
            high_resolution_logits = self.enc_score_head(high_resolution_memory)
            high_height, high_width = high_resolution.shape[-2:]
            high_resolution_anchors, high_resolution_valid = self._generate_anchors(
                [[high_height, high_width]],
                grid_size=self.hrqs_anchor_size,
                dtype=memory.dtype,
                device=memory.device,
            )
            if memory.shape[0] > 1:
                high_resolution_anchors = high_resolution_anchors.repeat(
                    memory.shape[0], 1, 1
                )
                high_resolution_valid = high_resolution_valid.repeat(
                    memory.shape[0], 1, 1
                )
            high_resolution_memory = (
                high_resolution_valid.to(high_resolution_memory.dtype)
                * high_resolution_memory
            )

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.sqmi_enabled:
            enc_topk_memory, enc_topk_logits, enc_topk_anchors, _ = self._select_topk(
                output_memory,
                enc_outputs_logits,
                anchors,
                ordinary_query_count,
                return_indices=True,
            )
        else:
            enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
                output_memory,
                enc_outputs_logits,
                anchors,
                ordinary_query_count,
            )
        if self.hrqs_enabled:
            if self.sqmi_enabled:
                hrqs_memory, hrqs_logits, hrqs_anchors, _ = self._select_topk(
                    high_resolution_memory,
                    high_resolution_logits,
                    high_resolution_anchors,
                    self.hrqs_num_queries,
                    return_indices=True,
                )
            else:
                hrqs_memory, hrqs_logits, hrqs_anchors = self._select_topk(
                    high_resolution_memory,
                    high_resolution_logits,
                    high_resolution_anchors,
                    self.hrqs_num_queries,
                )
            enc_topk_memory = torch.cat((enc_topk_memory, hrqs_memory), dim=1)
            if enc_topk_logits is not None:
                enc_topk_logits = torch.cat((enc_topk_logits, hrqs_logits), dim=1)
            enc_topk_anchors = torch.cat((enc_topk_anchors, hrqs_anchors), dim=1)
            selected_count = torch.full(
                (memory.shape[0],),
                self.hrqs_num_queries,
                device=memory.device,
                dtype=torch.long,
            )
            self.last_hrqs_selected_count = selected_count
            self.last_hrqs_selected_ratio = selected_count.float() / float(
                self.num_queries
            )
        else:
            self.last_hrqs_selected_count = None
            self.last_hrqs_selected_ratio = None

        enc_topk_bbox_unact: torch.Tensor = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors
        enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)

        if self.sqmi is not None:
            if sqmi_source is None:
                raise RuntimeError("S-QMI1 is enabled but no RGB S8 source was provided")
            (
                enc_topk_bboxes,
                self.last_sqmi_query_embeddings,
                self.last_sqmi_pixel_features,
                self.last_sqmi_diagnostics,
            ) = self.sqmi(
                enc_topk_memory,
                sqmi_source,
                enc_topk_bboxes,
                detector_scores=enc_topk_logits,
            )
            if self.sqmi_apply_initialization:
                enc_topk_bbox_unact = inverse_sigmoid(enc_topk_bboxes)
        else:
            self.last_sqmi_query_embeddings = None
            self.last_sqmi_pixel_features = None
            self.last_sqmi_diagnostics = None

        if self.training:
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        # if self.num_select_queries != self.num_queries:
        #     raise NotImplementedError('')

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(
        self,
        memory: torch.Tensor,
        outputs_logits: torch.Tensor,
        outputs_anchors_unact: torch.Tensor,
        topk: int,
        return_indices: bool = False,
    ):
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == "agnostic":
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        topk_ind: torch.Tensor

        topk_anchors = outputs_anchors_unact.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1])
        )

        topk_logits = (
            outputs_logits.gather(
                dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])
            )
            if self.training or return_indices
            else None
        )

        topk_memory = memory.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1])
        )

        if return_indices:
            return topk_memory, topk_logits, topk_anchors, topk_ind
        return topk_memory, topk_logits, topk_anchors

    def forward(
        self,
        feats,
        targets=None,
        qrl_source=None,
        mdqa_source=None,
        sqmi_source=None,
        qcsr_stage_features=None,
        thermal_feature=None,
        thermal_features=None,
        thermal_dropout_mask=None,
        thermal_content_mask=None,
        hrqs_source=None,
    ):
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats)
        thermal_evidence = None
        thermal_spatial_gate = None
        thermal_alignment_features = None
        if self.sdtec_enabled:
            if thermal_feature is None:
                raise RuntimeError(
                    "SDTEC is enabled but DFINE did not provide a thermal feature"
                )
            if self.sdtec_fusion_mode == "soft_aligned":
                if thermal_features is None:
                    raise RuntimeError(
                        "MA1 is enabled but DFINE omitted thermal feature levels"
                    )
                if len(thermal_features) < self.sdtec_alignment_levels:
                    raise RuntimeError(
                        "MA1 received fewer thermal levels than requested"
                    )
                thermal_alignment_features = thermal_features[
                    -self.sdtec_alignment_levels :
                ]
            elif self.sdtec_fusion_mode == "coordinate_free":
                if self.sdtec_candidate_tokenizer is not None:
                    if thermal_features is None:
                        raise RuntimeError(
                            "M3 is enabled but DFINE omitted thermal feature levels"
                        )
                    thermal_evidence = self.sdtec_candidate_tokenizer(
                        thermal_features
                    )
                else:
                    thermal_evidence = self.sdtec_tokenizer(thermal_feature)
            else:
                last_height, last_width = spatial_shapes[-1]
                last_count = int(last_height * last_width)
                thermal_memory = thermal_feature.flatten(2).transpose(1, 2)
                if thermal_memory.shape[1] != last_count:
                    raise RuntimeError(
                        "Same-position control received mismatched S32 spatial size: "
                        f"decoder={last_height}x{last_width}, "
                        f"thermal={thermal_feature.shape[-2:]}"
                    )
                rgb_memory = memory[:, -last_count:]
                fused_memory, thermal_spatial_gate = self.sdtec_spatial_coupler(
                    rgb_memory, thermal_memory
                )
                memory = torch.cat((memory[:, :-last_count], fused_memory), dim=1)
        if self.qrl is not None:
            if qrl_source is None:
                raise RuntimeError(
                    "QRL is enabled but HGNetv2 did not expose qrl_source"
                )
            qrl_state = self.qrl.prepare(qrl_source)
        else:
            qrl_state = None

        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                )
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = (
            self._get_decoder_input(
                memory,
                spatial_shapes,
                denoising_logits,
                denoising_bbox_unact,
                hrqs_source=hrqs_source,
                sqmi_source=sqmi_source,
            )
        )

        # decoder
        (
            out_bboxes,
            out_logits,
            out_corners,
            out_refs,
            pre_bboxes,
            pre_logits,
            out_queries,
            out_base_corners,
            out_head_offsets,
        ) = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            dn_meta=dn_meta,
            qrl_carrier=self.qrl,
            qrl_state=qrl_state,
            thermal_tokens=(
                thermal_evidence["tokens"] if thermal_evidence is not None else None
            ),
            thermal_quality=(
                thermal_evidence["token_quality"]
                if thermal_evidence is not None
                else None
            ),
            thermal_uncertainty=(
                thermal_evidence["uncertainty"]
                if thermal_evidence is not None
                else None
            ),
            thermal_presence=(
                thermal_evidence["presence_logits"]
                if thermal_evidence is not None
                else None
            ),
            thermal_couplers=self.sdtec_couplers,
            query_rank_score_bias=self.qrank_score_bias,
        )

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)

            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)
            dn_out_queries, out_queries = torch.split(
                out_queries, dn_meta["dn_num_split"], dim=2
            )
            dn_out_base_corners, out_base_corners = torch.split(
                out_base_corners, dn_meta["dn_num_split"], dim=2
            )
            dn_out_head_offsets, out_head_offsets = torch.split(
                out_head_offsets, dn_meta["dn_num_split"], dim=2
            )

        ma1_diagnostics = None
        ma1_mismatch_diagnostics = None
        ma1_pair_valid_mask = None
        if self.sdtec_aligned_calibrator is not None:
            if thermal_alignment_features is None:
                raise RuntimeError("MA1 alignment features were not prepared")
            base_final_logits = out_logits[-1]
            calibrated_logits, ma1_diagnostics = self.sdtec_aligned_calibrator(
                out_queries[-1],
                base_final_logits,
                out_bboxes[-1][..., :2].detach(),
                thermal_alignment_features,
                thermal_content_mask,
            )
            out_logits = torch.cat(
                (out_logits[:-1], calibrated_logits.unsqueeze(0)), dim=0
            )
            if (
                self.training
                and self.sdtec_train_mismatch
                and base_final_logits.shape[0] > 1
            ):
                mismatch_content_mask = thermal_content_mask.roll(1, dims=0)
                _mismatch_logits, ma1_mismatch_diagnostics = (
                    self.sdtec_aligned_calibrator(
                        out_queries[-1],
                        base_final_logits,
                        out_bboxes[-1][..., :2].detach(),
                        [feature.roll(1, dims=0) for feature in thermal_alignment_features],
                        mismatch_content_mask,
                    )
                )
                ma1_pair_valid_mask = (
                    thermal_content_mask.bool() & mismatch_content_mask.bool()
                )

        m2_diagnostics = None
        m2_mismatch_logits = None
        m2_mismatch_diagnostics = None
        m2_pair_valid_mask = None
        if thermal_evidence is not None and self.sdtec_logit_calibrator is not None:
            if thermal_dropout_mask is None:
                thermal_valid = torch.ones(
                    out_logits.shape[1],
                    device=out_logits.device,
                    dtype=torch.bool,
                )
            else:
                thermal_valid = ~thermal_dropout_mask.bool()
            base_final_logits = out_logits[-1]
            calibrated_logits, m2_diagnostics = self.sdtec_logit_calibrator(
                out_queries[-1],
                base_final_logits,
                thermal_evidence["tokens"],
                thermal_evidence["token_quality"],
                thermal_evidence["uncertainty"],
                thermal_evidence["presence_logits"],
                thermal_valid=thermal_valid,
            )
            out_logits = torch.cat(
                (out_logits[:-1], calibrated_logits.unsqueeze(0)), dim=0
            )

            if (
                self.training
                and self.sdtec_train_mismatch
                and base_final_logits.shape[0] > 1
            ):
                # A deterministic within-batch derangement provides a cheap
                # paired-content negative without a second thermal backbone
                # pass.  The coordinate-free tokenizer makes this a pure
                # image-pair intervention rather than a spatial alignment test.
                mismatch_valid = thermal_valid.roll(1, dims=0)
                m2_mismatch_logits, m2_mismatch_diagnostics = (
                    self.sdtec_logit_calibrator(
                        out_queries[-1],
                        base_final_logits,
                        thermal_evidence["tokens"].roll(1, dims=0),
                        thermal_evidence["token_quality"].roll(1, dims=0),
                        thermal_evidence["uncertainty"].roll(1, dims=0),
                        thermal_evidence["presence_logits"].roll(1, dims=0),
                        thermal_valid=mismatch_valid,
                    )
                )
                m2_pair_valid_mask = thermal_valid & mismatch_valid

        if self.sqfr_refiner is not None:
            if hrqs_source is None:
                raise RuntimeError("SQFR1 is enabled but no RGB S8 source was provided")
            refined_boxes = self.sqfr_refiner(
                hrqs_source,
                out_bboxes[-1],
                out_logits[-1],
                out_queries[-1],
            )
            out_bboxes = torch.cat(
                (out_bboxes[:-1], refined_boxes.unsqueeze(0)), dim=0
            )

        if self.training:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale,
            }
            if self.qrl is not None:
                out["qrl_region_logits"] = self.qrl.last_region_logits
            if self.mdqa_enabled:
                if mdqa_source is None:
                    raise RuntimeError(
                        "MDQA is enabled but HGNetv2 did not expose mdqa_source"
                    )
                mdqa_query_source = (
                    out_queries[-1].detach()
                    if self.mdqa_detach_query
                    else out_queries[-1]
                )
                out["mdqa_query_embeddings"] = self.mdqa_query_proj(
                    mdqa_query_source
                )
                out["mdqa_pixel_features"] = self.mdqa_pixel_proj(mdqa_source)
            if self.sqmi_enabled:
                if (
                    self.last_sqmi_query_embeddings is None
                    or self.last_sqmi_pixel_features is None
                ):
                    raise RuntimeError("S-QMI1 did not produce mask embeddings")
                out["sqmi_query_embeddings"] = self.last_sqmi_query_embeddings
                out["sqmi_pixel_features"] = self.last_sqmi_pixel_features
            if self.qcsr_enabled:
                if qcsr_stage_features is None or len(qcsr_stage_features) != 2:
                    raise RuntimeError(
                        "QCSR is enabled but HGNetv2 did not expose real S8/S16 features"
                    )
                qcsr_s8, qcsr_s16 = qcsr_stage_features
                # The SAM-anchored teacher may train its private projections,
                # but it cannot reshape S8 or the detector query.  Only the
                # student projection remains connected to the real S16 path.
                out["qcsr_query_embeddings"] = self.qcsr_query_proj(
                    out_queries[-1].detach()
                )
                out["qcsr_teacher_pixels"] = self.qcsr_teacher_proj(
                    qcsr_s8.detach()
                )
                out["qcsr_student_pixels"] = self.qcsr_student_proj(qcsr_s16)
            if ma1_diagnostics is not None:
                out["ma1_logit_delta"] = ma1_diagnostics["delta"]
                out["ma1_gate"] = ma1_diagnostics["gate"]
                out["ma1_gate_logits"] = ma1_diagnostics["gate_logits"]
                out["ma1_mapped_centres"] = ma1_diagnostics[
                    "mapped_centres"
                ]
                out["ma1_effective_offset"] = ma1_diagnostics[
                    "effective_offset"
                ]
                out["ma1_aligned_centres"] = ma1_diagnostics[
                    "aligned_centres"
                ]
                out["ma1_max_abs_residual_offset"] = ma1_diagnostics[
                    "max_abs_residual_offset"
                ]
                out["ma1_attention_entropy"] = ma1_diagnostics[
                    "attention_entropy"
                ]
                out["ma1_thermal_content_mask"] = ma1_diagnostics[
                    "thermal_content_mask"
                ]
                out["ma1_effective_affine"] = ma1_diagnostics[
                    "effective_affine"
                ]
                out["ma1_fusion_progress"] = memory.new_tensor(
                    self.sdtec_fusion_progress
                )
                if ma1_mismatch_diagnostics is not None:
                    out["ma1_mismatch_gate_logits"] = ma1_mismatch_diagnostics[
                        "gate_logits"
                    ]
                    out["ma1_mismatch_delta"] = ma1_mismatch_diagnostics[
                        "delta"
                    ]
                    out["ma1_pair_valid_mask"] = ma1_pair_valid_mask
            if thermal_evidence is not None:
                out["sdtec_presence_logits"] = thermal_evidence[
                    "presence_logits"
                ]
                out["sdtec_tokens"] = thermal_evidence["tokens"]
                out["sdtec_token_quality"] = thermal_evidence["token_quality"]
                out["sdtec_uncertainty"] = thermal_evidence["uncertainty"]
                if "candidate_scores" in thermal_evidence:
                    out["sdtec_candidate_scores"] = thermal_evidence[
                        "candidate_scores"
                    ]
                if self.sdtec_logit_calibrator is not None:
                    if m2_diagnostics is None:
                        raise RuntimeError("M2 calibration diagnostics were not produced")
                    out["sdtec_logit_delta"] = m2_diagnostics["delta"]
                    out["sdtec_logit_gate"] = m2_diagnostics["gate"]
                    out["sdtec_gate_by_layer"] = m2_diagnostics["gate"].mean(
                        dim=1
                    ).unsqueeze(0)
                    out["sdtec_effective_scale_by_layer"] = (
                        thermal_evidence["presence_logits"].new_tensor(
                            [
                                self.sdtec_logit_calibrator.max_logit_delta
                                * self.sdtec_logit_calibrator.fusion_progress
                            ]
                        )
                    )
                    if m2_mismatch_logits is not None:
                        out["sdtec_mismatch_logits"] = m2_mismatch_logits
                        out["sdtec_mismatch_delta"] = m2_mismatch_diagnostics[
                            "delta"
                        ]
                        out["sdtec_pair_valid_mask"] = m2_pair_valid_mask
                elif hasattr(self.sdtec_couplers, "effective_scales"):
                    out["sdtec_gate_by_layer"] = (
                        self.decoder.last_sdtec_gate_by_layer
                    )
                    out["sdtec_effective_scale_by_layer"] = (
                        self.sdtec_couplers.effective_scales()
                    )
                else:
                    out["sdtec_gate_by_layer"] = (
                        self.decoder.last_sdtec_gate_by_layer
                    )
                    out["sdtec_effective_scale_by_layer"] = torch.stack(
                        [
                            coupler.residual_scale.tanh()
                            * coupler.max_residual_scale
                            * coupler.fusion_progress
                            for coupler in self.sdtec_couplers
                        ]
                    )
                out["sdtec_fusion_progress"] = thermal_evidence[
                    "presence_logits"
                ].new_tensor(self.sdtec_fusion_progress)
                if thermal_dropout_mask is None:
                    thermal_dropout_mask = torch.zeros(
                        thermal_evidence["presence_logits"].shape[0],
                        device=thermal_evidence["presence_logits"].device,
                        dtype=torch.bool,
                    )
                out["sdtec_dropout_mask"] = thermal_dropout_mask
            elif thermal_spatial_gate is not None:
                out["sdtec_spatial_gate"] = thermal_spatial_gate
                out["sdtec_effective_scale_by_layer"] = torch.stack(
                    [
                        self.sdtec_spatial_coupler.residual_scale.tanh()
                        * self.sdtec_spatial_coupler.max_residual_scale
                        * self.sdtec_spatial_coupler.fusion_progress
                    ]
                )
                out["sdtec_fusion_progress"] = memory.new_tensor(
                    self.sdtec_fusion_progress
                )
                if thermal_dropout_mask is None:
                    thermal_dropout_mask = torch.zeros(
                        memory.shape[0],
                        device=memory.device,
                        dtype=torch.bool,
                    )
                out["sdtec_dropout_mask"] = thermal_dropout_mask
        else:
            out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}

        # STQL/QCER consume the same final non-DN query tensor that produced
        # pred_logits/pred_boxes.  DFINE.forward enables this flag only for
        # the new research path, so legacy output dictionaries are unchanged.
        if getattr(self, "return_query_features", False):
            out["query_features"] = out_queries[-1]
        if getattr(self, "return_qdmf_head_state", False):
            out["qdmf_head_offset"] = out_head_offsets[-1]
            out["qdmf_base_corners"] = out_base_corners[-1]
            out["qdmf_pred_corners"] = out_corners[-1]
            out["qdmf_ref_points"] = out_refs[-1]

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_corners[-1],
                out_logits[-1],
            )
            out["enc_aux_outputs"] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}

            if dn_meta is not None:
                out["dn_outputs"] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_corners[-1],
                    dn_out_logits[-1],
                )
                out["dn_pre_outputs"] = {"pred_logits": dn_pre_logits, "pred_boxes": dn_pre_bboxes}
                out["dn_meta"] = dn_meta

        return out

    def qdmf_final_head(
        self,
        base_queries,
        fused_queries,
        head_offset,
        base_pred_corners,
        base_corners,
        ref_points,
    ):
        """Re-run the actual final D-FINE heads for QDMF-updated queries.

        ``head_offset`` and ``base_corners`` preserve the decoder's iterative
        distribution-refinement state.  The difference form makes this a
        final-query update rather than a new box head.
        """
        layer_index = self.num_layers - 1 if self.training else self.eval_idx
        bbox_head = self.dec_bbox_head[layer_index]
        old_delta = bbox_head(base_queries + head_offset)
        new_delta = bbox_head(fused_queries + head_offset)
        refined_base_corners = base_corners + (new_delta - old_delta)
        qrl_delta = base_pred_corners - base_corners
        refined_corners = refined_base_corners + qrl_delta
        if not hasattr(self.decoder, "project"):
            project = weighting_function(
                self.reg_max, self.up, self.reg_scale
            )
        else:
            project = self.decoder.project
        refined_boxes = distance2bbox(
            ref_points,
            self.integral(refined_corners, project),
            self.reg_scale,
        )
        refined_logits = self.dec_score_head[layer_index](fused_queries)
        lqe_corners = refined_base_corners + qrl_delta.detach()
        refined_logits = self.decoder.lqe_layers[layer_index](
            refined_logits, lqe_corners
        )
        if self.qrank_score_bias is not None:
            refined_logits = self.qrank_score_bias(
                refined_logits, layer_index=layer_index
            )
        return refined_logits, refined_boxes, refined_corners

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class,
        outputs_coord,
        outputs_corners,
        outputs_ref,
        teacher_corners=None,
        teacher_logits=None,
    ):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_corners": c,
                "ref_points": d,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            for a, b, c, d in zip(outputs_class, outputs_coord, outputs_corners, outputs_ref)
        ]
