"""
reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register
from .common import FrozenBatchNorm2d
from .partialnet_pat_sf import (
    PartialConvOnly,
    PartialGlobalMeanConv,
    PartialGlobalQueryConv,
    PartialSelfAttentionConv,
)
from .srtod_modules import DGFE, RH, DeficiencyPredictor

# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_

__all__ = ["HGNetv2"]

def safe_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    else:
        pass

def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size,
        stride=1,
        groups=1,
        padding="",
        use_act=True,
        use_lab=False,
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == "same":
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(in_chs, out_chs, kernel_size, stride, groups=groups, bias=False),
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            )
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x


class PartialSelfAttentionBNAct(nn.Module):
    """Host adapter that replaces a spatial ConvBNAct with official PAT_sf.

    PAT_sf itself is unchanged.  Batch normalization, ReLU, and the optional
    learnable affine block are retained as HGNetv2's original post-operation
    wrapper so that the experiment changes only the spatial mixing operator.
    """

    def __init__(self, channels, n_div=4, use_lab=False, variant="self"):
        super().__init__()
        variants = {
            "self": PartialSelfAttentionConv,
            "partial": PartialConvOnly,
            "channel": PartialChannelAttentionConv,
            "global_mean": PartialGlobalMeanConv,
            "global_query": PartialGlobalQueryConv,
        }
        if variant not in variants:
            raise ValueError(
                f"Unsupported PAT spatial-mixer variant {variant!r}; "
                f"expected one of {sorted(variants)}"
            )
        self.pat_sf = variants[variant](channels, n_div=n_div)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()
        self.lab = LearnableAffineBlock() if use_lab else nn.Identity()

    def forward(self, x):
        x = self.pat_sf(x)
        x = self.bn(x)
        x = self.act(x)
        return self.lab(x)


class LightConvBNAct(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size,
        groups=1,
        use_lab=False,
        use_pat_sf=False,
        pat_sf_n_div=4,
        pat_sf_variant="self",
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
        )
        if use_pat_sf:
            self.conv2 = PartialSelfAttentionBNAct(
                out_chs,
                n_div=pat_sf_n_div,
                use_lab=use_lab,
                variant=pat_sf_variant,
            )
        else:
            self.conv2 = ConvBNAct(
                out_chs,
                out_chs,
                kernel_size=kernel_size,
                groups=out_chs,
                use_act=True,
                use_lab=use_lab,
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    # for HGNetv2
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_chs,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem2a = ConvBNAct(
            mid_chs,
            mid_chs // 2,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem2b = ConvBNAct(
            mid_chs // 2,
            mid_chs,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem3 = ConvBNAct(
            mid_chs * 2,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem4 = ConvBNAct(
            mid_chs,
            out_chs,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class EseModule(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(
            chs,
            chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)


class GaussianChannelAttention(nn.Module):
    """Gaussian-SE branch used by PartialNet's PAT_ch block.

    This is a dependency-free transcription of the official AAAI 2026
    PartialNet implementation.  It uses both channel mean and standard
    deviation instead of discarding the second-order statistic.
    """

    def __init__(self, channels):
        super().__init__()
        self.stat_fusion = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 2),
            bias=False,
        )
        self.gate_norm = nn.BatchNorm2d(channels)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.shape
        flattened = x.reshape(batch, channels, -1)
        mean = flattened.mean(-1).view(batch, channels, 1, 1)
        std = flattened.std(-1).view(batch, channels, 1, 1)
        statistics = torch.cat((mean, std), dim=-1)
        weight = self.gate(self.gate_norm(self.stat_fusion(statistics)))
        return x * weight


class PartialChannelAttentionConv(nn.Module):
    """PartialNet PAT_ch: partial 3x3 convolution + Gaussian channel attention.

    Only the first ``1 / n_div`` channels receive the spatial convolution.
    The remaining channels retain their own feature values and are modulated
    by the mean/std channel-attention branch before concatenation.
    """

    def __init__(self, channels, n_div=4):
        super().__init__()
        channels = int(channels)
        n_div = int(n_div)
        if n_div <= 1 or channels % n_div != 0:
            raise ValueError(
                f"PAT_ch requires n_div > 1 and channels divisible by n_div; "
                f"got channels={channels}, n_div={n_div}"
            )
        self.conv_channels = channels // n_div
        self.attention_channels = channels - self.conv_channels
        self.partial_conv = nn.Conv2d(
            self.conv_channels,
            self.conv_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.channel_attention = GaussianChannelAttention(self.attention_channels)
        self.attention_norm = nn.BatchNorm2d(self.attention_channels)

    def forward(self, x):
        conv_x, attention_x = torch.split(
            x,
            [self.conv_channels, self.attention_channels],
            dim=1,
        )
        conv_x = self.partial_conv(conv_x)
        attention_x = self.attention_norm(self.channel_attention(attention_x))
        return torch.cat((conv_x, attention_x), dim=1)


class HG_Block(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        layer_num,
        kernel_size=3,
        residual=False,
        light_block=False,
        use_lab=False,
        agg="ese",
        drop_path=0.0,
        use_pat_sf=False,
        pat_sf_n_div=4,
        pat_sf_variant="self",
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            if light_block:
                self.layers.append(
                    LightConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                        use_pat_sf=use_pat_sf,
                        pat_sf_n_div=pat_sf_n_div,
                        pat_sf_variant=pat_sf_variant,
                    )
                )
            else:
                self.layers.append(
                    ConvBNAct(
                        in_chs if i == 0 else mid_chs,
                        mid_chs,
                        kernel_size=kernel_size,
                        stride=1,
                        use_lab=use_lab,
                    )
                )

        # feature aggregation
        total_chs = in_chs + layer_num * mid_chs
        if agg == "se":
            aggregation_squeeze_conv = ConvBNAct(
                total_chs,
                out_chs // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        else:
            aggregation_conv = ConvBNAct(
                total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x


class PartialInformationPreservingDownsample(nn.Module):
    """Add a cheap sub-pixel detail residual to an existing downsample.

    A fixed fraction of input channels is rearranged with PixelUnshuffle before
    projection.  The original HGNetv2 depthwise downsample remains untouched,
    which preserves checkpoint compatibility and provides the semantic branch.
    The ``pooled`` mode has exactly the same learned projection as ``pixel`` but
    removes the four sub-pixel phases, making it a parameter-matched control.
    """

    def __init__(
        self,
        channels,
        ratio=0.25,
        groups=4,
        mode="pixel",
        init_scale=0.1,
    ):
        super().__init__()
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"preserve ratio must be in (0, 1], got {ratio}")
        if mode not in {"pixel", "pooled"}:
            raise ValueError(f"unsupported preserve mode: {mode}")

        detail_channels = max(1, int(round(channels * ratio)))
        projection_groups = math.gcd(math.gcd(4 * detail_channels, channels), groups)

        self.detail_channels = detail_channels
        self.mode = mode
        self.proj = nn.Conv2d(
            4 * detail_channels,
            channels,
            kernel_size=1,
            groups=max(1, projection_groups),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x, semantic):
        detail = x[:, : self.detail_channels]
        pad_h = detail.shape[-2] % 2
        pad_w = detail.shape[-1] % 2
        if pad_h or pad_w:
            detail = F.pad(detail, (0, pad_w, 0, pad_h))

        if self.mode == "pixel":
            detail = F.pixel_unshuffle(detail, 2)
        else:
            detail = F.avg_pool2d(detail, kernel_size=2, stride=2)
            detail = detail.repeat_interleave(4, dim=1)

        detail = self.bn(self.proj(detail))
        return semantic + self.scale * detail


class DeficiencyGuidedHaarDetailResidual(nn.Module):
    """Low-cost pre-downsample directional-detail residual for S2.

    The standard HGNet path remains untouched.  This branch takes the feature
    before S8->S16 compression, extracts only the three Haar high-frequency
    subbands (LH/HL/HH), projects them to the returned S16 width, and exposes a
    residual that is spatially gated after the standard stage has completed.

    A zero-initialized scalar makes the detection path exactly identical to
    A00 at construction while retaining a non-zero gradient for the scalar.
    Once the scalar moves away from zero, gradients also reach the projection.
    """

    def __init__(self, in_channels, out_channels, groups=16, init_scale=0.0):
        super().__init__()
        projection_groups = math.gcd(math.gcd(3 * in_channels, out_channels), groups)
        projection_groups = max(1, projection_groups)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.projection_groups = projection_groups
        self.orientation_mix = nn.Sequential(
            nn.Conv2d(
                3 * in_channels,
                out_channels,
                kernel_size=1,
                groups=projection_groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.local_refine = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))
        self.gate_mode = "learned"
        self.gate_override = None
        self.shuffle_seed = 20260813
        self.last_gate = None
        self.last_scale = None
        self.last_residual_rms = None

    @staticmethod
    def haar_detail(x):
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]
        # Orthonormal 2x2 Haar signs. LL is deliberately excluded so the
        # residual cannot duplicate the standard low-frequency main path.
        lh = 0.5 * (x00 - x01 + x10 - x11)
        hl = 0.5 * (x00 + x01 - x10 - x11)
        hh = 0.5 * (x00 - x01 - x10 + x11)
        return torch.cat((lh, hl, hh), dim=1)

    def extract(self, x):
        detail = self.haar_detail(x)
        residual = self.local_refine(self.orientation_mix(detail))
        self.last_residual_rms = residual.detach().float().square().mean().sqrt()
        return residual

    def resolve_gate(self, gate, size):
        gate = F.interpolate(gate.float(), size=size, mode="bilinear", align_corners=False)
        if self.gate_mode == "one":
            gate = torch.ones_like(gate)
        elif self.gate_mode == "zero":
            gate = torch.zeros_like(gate)
        elif self.gate_mode == "shuffled":
            generator = torch.Generator(device="cpu").manual_seed(
                self.shuffle_seed + size[0] * 1009 + size[1]
            )
            permutation = torch.randperm(size[0] * size[1], generator=generator).to(gate.device)
            gate = gate.flatten(-2).index_select(-1, permutation).reshape_as(gate)
        elif self.gate_mode == "external":
            if self.gate_override is None:
                raise RuntimeError("external Haar-detail gate requires gate_override")
            gate = F.interpolate(
                self.gate_override.float(), size=size, mode="bilinear", align_corners=False
            )
        elif self.gate_mode != "learned":
            raise ValueError(f"unsupported Haar-detail gate mode: {self.gate_mode}")
        self.last_gate = gate.detach()
        return gate

    def apply_gate(self, residual, gate):
        gate = self.resolve_gate(gate, residual.shape[-2:]).to(residual.dtype)
        self.last_scale = self.scale.detach()
        return self.scale.to(residual.dtype) * gate * residual


class PhaseCandidateResidualMLP(nn.Module):
    """Per-channel nonlinear mixing of the four explicit 2x2 phases.

    PixelUnshuffle stores four consecutive phase channels for every original
    channel. Grouped 1x1 convolutions with ``groups=C`` therefore implement an
    independent 4->4 MLP for each original channel. The second projection is
    zero-initialized, making the whole block an exact identity at startup.
    """

    def __init__(self, channels):
        super().__init__()
        phase_channels = 4 * channels
        self.expand = nn.Conv2d(
            phase_channels, phase_channels, kernel_size=1, groups=channels, bias=True
        )
        self.act = nn.SiLU(inplace=True)
        self.project = nn.Conv2d(
            phase_channels, phase_channels, kernel_size=1, groups=channels, bias=True
        )
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        return x + self.project(self.act(self.expand(x)))


class PhaseCandidateDirectMLP(nn.Module):
    """Direct nonlinear 4->4 phase transform without an identity bypass."""

    def __init__(self, channels):
        super().__init__()
        phase_channels = 4 * channels
        self.expand = nn.Conv2d(
            phase_channels, phase_channels, kernel_size=1, groups=channels, bias=True
        )
        self.act = nn.SiLU(inplace=True)
        self.project = nn.Conv2d(
            phase_channels, phase_channels, kernel_size=1, groups=channels, bias=True
        )

    def forward(self, x):
        return self.project(self.act(self.expand(x)))


class LightAdaptiveWeightDownsample(nn.Module):
    """Paper-faithful LAD operator from LRDS-YOLO (Han et al., 2025).

    The attention path predicts four phase weights per output channel from a
    non-overlapping 2x2 average-pooled feature.  The local path uses a 3x3,
    stride-2, eight-group convolution to produce four candidate values per
    output channel.  Softmax-weighted phase candidates are summed back to C.

    LAD was lightweight relative to YOLO's dense stride convolution.  HGNetv2
    already uses a depthwise stride convolution, so its cost must be measured
    rather than assumed to be lightweight in this backbone.
    """

    def __init__(self, channels, groups=8, target_gate=False, candidate_mode="learned_conv"):
        super().__init__()
        if channels % groups != 0:
            raise ValueError(f"LAD channels={channels} must be divisible by groups={groups}")
        self.channels = channels
        self.target_gate = bool(target_gate)
        self.candidate_mode = candidate_mode
        self.weight_projection = nn.Conv2d(channels, 4 * channels, kernel_size=1, bias=True)
        if candidate_mode == "learned_conv":
            self.local_projection = nn.Conv2d(
                channels,
                4 * channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=groups,
                bias=True,
            )
            self.phase_transform = None
        elif candidate_mode in ("polyphase", "polyphase_mlp", "polyphase_direct_mlp"):
            # Exact 2x2 space-to-depth rearrangement.  Channel ordering from
            # pixel_unshuffle is [channel, phase], matching [B,C,4,H/2,W/2].
            self.local_projection = None
            if candidate_mode == "polyphase_mlp":
                self.phase_transform = PhaseCandidateResidualMLP(channels)
            elif candidate_mode == "polyphase_direct_mlp":
                self.phase_transform = PhaseCandidateDirectMLP(channels)
            else:
                self.phase_transform = None
        else:
            raise ValueError(f"unsupported LAD candidate_mode: {candidate_mode}")
        # S-LAD2-TG predicts targetness at the original S8 resolution.  The
        # predictor can be initialized from the independently validated S-AUX
        # head because both consume the same 256-channel S8 representation.
        self.target_projection = (
            nn.Conv2d(channels, 1, kernel_size=1, bias=True) if self.target_gate else None
        )
        # Evaluation-only causal controls.  Training uses ``learned`` unless a
        # diagnostic script explicitly changes this attribute after loading.
        self.phase_weight_mode = "learned"
        self.phase_shuffle_seed = 20260808
        # Evaluation-only intervention on the correspondence between the
        # four encoded candidates and their predicted phase weights.
        self.candidate_phase_mode = "learned"
        # Evaluation-only intervention on targetness q.  This is independent
        # from ``phase_weight_mode`` so causal audits can isolate whether the
        # learned target locations, rather than phase weights alone, matter.
        self.target_gate_mode = "learned"
        self.target_gate_override = None
        self.last_selectivity_map = None
        self.last_phase_entropy = None
        self.last_candidate_variance = None
        self.last_weight_deviation_from_uniform = None
        self.last_target_logits = None
        self.last_target_gate = None

    def forward(self, x):
        self.last_target_logits = self.target_projection(x) if self.target_projection is not None else None
        pooled = F.avg_pool2d(x, kernel_size=2, stride=2)
        batch, _, height, width = pooled.shape
        weights = self.weight_projection(pooled).reshape(
            batch, self.channels, 4, height, width
        )
        weights = weights.softmax(dim=2)
        if self.candidate_mode == "learned_conv":
            candidates = self.local_projection(x).reshape(
                batch, self.channels, 4, height, width
            )
        else:
            phase_features = F.pixel_unshuffle(x, downscale_factor=2)
            if self.phase_transform is not None:
                phase_features = self.phase_transform(phase_features)
            candidates = phase_features.reshape(
                batch, self.channels, 4, height, width
            )
        if self.candidate_phase_mode in ("permuted", "paired_permuted"):
            # Preserve every candidate value and its spatial position while
            # breaking only the candidate-index/weight-index correspondence.
            candidates = candidates[:, :, [2, 0, 3, 1], :, :]
        elif self.candidate_phase_mode != "learned":
            raise ValueError(
                f"unsupported LAD candidate_phase_mode: {self.candidate_phase_mode}"
            )
        if self.target_gate:
            # If any S8 cell inside a 2x2 compression window is target-like,
            # retain the adaptive phase selector for that S16 output cell.
            target_gate = F.max_pool2d(self.last_target_logits.sigmoid(), kernel_size=2, stride=2)
            if self.target_gate_mode == "oracle":
                if self.target_gate_override is None:
                    raise RuntimeError("oracle target gate requires target_gate_override")
                target_gate = self.target_gate_override.to(
                    device=target_gate.device, dtype=target_gate.dtype
                )
            elif self.target_gate_mode == "one":
                target_gate = torch.ones_like(target_gate)
            elif self.target_gate_mode == "zero":
                target_gate = torch.zeros_like(target_gate)
            elif self.target_gate_mode == "shuffled":
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self.phase_shuffle_seed + height * 917 + width)
                permutation = torch.randperm(height * width, generator=generator).to(target_gate.device)
                target_gate = target_gate.flatten(-2).index_select(-1, permutation).reshape_as(target_gate)
            elif self.target_gate_mode != "learned":
                raise ValueError(f"unsupported LAD target_gate_mode: {self.target_gate_mode}")
            self.last_target_gate = target_gate.detach()
            weights = 0.25 + target_gate.unsqueeze(2) * (weights - 0.25)
        else:
            self.last_target_gate = None

        if self.phase_weight_mode == "uniform":
            weights = torch.full_like(weights, 0.25)
        elif self.phase_weight_mode == "shuffled":
            # Use one fixed spatial permutation for every sample/channel.  It
            # preserves the learned weight distribution while breaking its
            # correspondence to the current image location.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.phase_shuffle_seed + height * 1009 + width)
            permutation = torch.randperm(height * width, generator=generator).to(weights.device)
            weights = weights.flatten(-2).index_select(-1, permutation).reshape_as(weights)
        elif self.phase_weight_mode != "learned":
            raise ValueError(f"unsupported LAD phase_weight_mode: {self.phase_weight_mode}")

        if self.candidate_phase_mode == "paired_permuted":
            # Algebraic sanity control: the same candidate/weight permutation
            # must leave their weighted sum unchanged.
            weights = weights[:, :, [2, 0, 3, 1], :, :]

        # Mean maximum phase probability is a non-constant, single-channel
        # diagnostic map.  It is detached so it never retains a training graph.
        self.last_selectivity_map = weights.max(dim=2).values.mean(dim=1, keepdim=True).detach()
        entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=2)
        self.last_phase_entropy = (entropy / math.log(4.0)).mean().detach()
        self.last_candidate_variance = candidates.var(dim=2, unbiased=False).mean().detach()
        self.last_weight_deviation_from_uniform = (weights - 0.25).abs().mean().detach()
        return (weights * candidates).sum(dim=2)


class TargetConditionalLADDetail(nn.Module):
    """Route between a stable standard downsample and a learned LAD detail path.

    The standard branch is owned by ``HG_Stage`` so its parameter names remain
    checkpoint-compatible with the official D-FINE model.  This module only
    predicts targetness, LAD candidates and their four-way weights.  A bounded
    scalar ``alpha`` controls the global intervention strength:

        output = (1 - alpha * q) * standard + alpha * q * detail

    Consequently q=0 (or alpha=0) is an exact structural fallback to the
    original downsample.  PixelUnshuffle is intentionally absent: this first
    experiment tests conditional compression rather than a candidate operator.
    """

    def __init__(self, channels, groups=8, route_init=0.1, target_gate_mode="learned"):
        super().__init__()
        if channels % groups != 0:
            raise ValueError(
                f"conditional LAD channels={channels} must be divisible by groups={groups}"
            )
        if not 0.0 < route_init < 1.0:
            raise ValueError(f"route_init must be in (0, 1), got {route_init}")
        self.channels = channels
        self.weight_projection = nn.Conv2d(channels, 4 * channels, kernel_size=1, bias=True)
        self.local_projection = nn.Conv2d(
            channels,
            4 * channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=groups,
            bias=True,
        )
        self.phase_transform = None
        self.target_projection = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        route_logit = math.log(route_init / (1.0 - route_init))
        self.route_scale_logit = nn.Parameter(torch.tensor(route_logit, dtype=torch.float32))

        if target_gate_mode not in {"learned", "one", "budget_centered"}:
            raise ValueError(
                "training target_gate_mode must be 'learned', 'one', or "
                "'budget_centered', "
                f"got {target_gate_mode}"
            )

        # ``one`` is also a training configuration for the ALLON-AUX
        # single-variable control. Other values remain evaluation-only causal
        # interventions assigned after model construction.
        self.phase_weight_mode = "learned"
        self.target_gate_mode = target_gate_mode
        self.route_scale_mode = "learned"
        self.phase_shuffle_seed = 20260809
        self.target_gate_override = None

        # Detached diagnostics populated on every forward.
        self.last_target_logits = None
        self.last_target_gate = None
        self.last_route_gate = None
        self.last_route_alpha = None
        self.last_selectivity_map = None
        self.last_phase_entropy = None
        self.last_candidate_variance = None
        self.last_weight_deviation_from_uniform = None
        self.last_base_detail_rms_ratio = None

    def forward(self, x, standard):
        self.last_target_logits = self.target_projection(x)
        pooled = F.avg_pool2d(x, kernel_size=2, stride=2)
        batch, _, height, width = pooled.shape
        weights = self.weight_projection(pooled).reshape(
            batch, self.channels, 4, height, width
        ).softmax(dim=2)
        candidates = self.local_projection(x).reshape(
            batch, self.channels, 4, height, width
        )

        target_gate = F.max_pool2d(
            self.last_target_logits.sigmoid(), kernel_size=2, stride=2
        )
        if self.target_gate_mode == "oracle":
            if self.target_gate_override is None:
                raise RuntimeError("oracle target gate requires target_gate_override")
            target_gate = self.target_gate_override.to(
                device=target_gate.device, dtype=target_gate.dtype
            )
        elif self.target_gate_mode == "one":
            target_gate = torch.ones_like(target_gate)
        elif self.target_gate_mode == "budget_centered":
            # Preserve the ALLON residual budget exactly while allowing q to
            # redistribute it spatially. Since q is in [0, 1], the centered
            # allocation remains bounded in [0, 2] and starts near ALLON when
            # q is initially close to uniform.
            spatial_mean = target_gate.mean(dim=(-2, -1), keepdim=True)
            target_gate = 1.0 + target_gate - spatial_mean
        elif self.target_gate_mode == "budget_centered_shuffled":
            # Evaluation-only causal control for CDS2. Preserve both the
            # per-image mean allocation (=1) and the learned allocation-value
            # distribution, while breaking correspondence to image locations.
            spatial_mean = target_gate.mean(dim=(-2, -1), keepdim=True)
            target_gate = 1.0 + target_gate - spatial_mean
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.phase_shuffle_seed + height * 917 + width)
            permutation = torch.randperm(height * width, generator=generator).to(target_gate.device)
            target_gate = target_gate.flatten(-2).index_select(-1, permutation).reshape_as(target_gate)
        elif self.target_gate_mode == "zero":
            target_gate = torch.zeros_like(target_gate)
        elif self.target_gate_mode == "shuffled":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.phase_shuffle_seed + height * 917 + width)
            permutation = torch.randperm(height * width, generator=generator).to(target_gate.device)
            target_gate = target_gate.flatten(-2).index_select(-1, permutation).reshape_as(target_gate)
        elif self.target_gate_mode != "learned":
            raise ValueError(f"unsupported target gate mode: {self.target_gate_mode}")

        if self.phase_weight_mode == "uniform":
            weights = torch.full_like(weights, 0.25)
        elif self.phase_weight_mode == "shuffled":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.phase_shuffle_seed + height * 1009 + width)
            permutation = torch.randperm(height * width, generator=generator).to(weights.device)
            weights = weights.flatten(-2).index_select(-1, permutation).reshape_as(weights)
        elif self.phase_weight_mode != "learned":
            raise ValueError(f"unsupported phase weight mode: {self.phase_weight_mode}")

        detail = (weights * candidates).sum(dim=2)
        alpha = self.route_scale_logit.sigmoid()
        if self.route_scale_mode == "zero":
            alpha = torch.zeros_like(alpha)
        elif self.route_scale_mode == "one":
            alpha = torch.ones_like(alpha)
        elif self.route_scale_mode != "learned":
            raise ValueError(f"unsupported route scale mode: {self.route_scale_mode}")
        route_gate = alpha * target_gate
        output = standard + route_gate * (detail - standard)

        self.last_target_gate = target_gate.detach()
        self.last_route_gate = route_gate.detach()
        self.last_route_alpha = alpha.detach()
        self.last_selectivity_map = weights.max(dim=2).values.mean(dim=1, keepdim=True).detach()
        entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=2)
        self.last_phase_entropy = (entropy / math.log(4.0)).mean().detach()
        self.last_candidate_variance = candidates.var(dim=2, unbiased=False).mean().detach()
        self.last_weight_deviation_from_uniform = (weights - 0.25).abs().mean().detach()
        base_rms = standard.detach().float().square().mean().sqrt().clamp_min(1e-12)
        detail_rms = detail.detach().float().square().mean().sqrt()
        self.last_base_detail_rms_ratio = (detail_rms / base_rms).detach()
        return output


class FixedBlurFilter(nn.Module):
    """Checkpoint-compatible anti-alias prefilter for a stride-2 downsample.

    The fixed 3x3 binomial kernel is applied depthwise at stride 1.  Keeping
    the original learned downsample as a separate following operation retains
    all of its pretrained parameters and isolates the anti-aliasing variable.
    """

    def __init__(self, channels):
        super().__init__()
        kernel_1d = torch.tensor([1.0, 2.0, 1.0])
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        self.channels = channels
        self.register_buffer("kernel", kernel_2d.reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding=1, groups=self.channels)


class BTRDConvBNAct(nn.Module):
    """Conv-BN-SiLU wrapper used by the source BTRD implementation."""

    def __init__(self, in_chs, out_chs, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chs)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BTRDAddCoords(nn.Module):
    """Append normalized x/y coordinate planes without changing semantics."""

    def forward(self, x):
        batch, _, height, width = x.shape
        dtype = x.dtype
        device = x.device
        y = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
        x_coord = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
        y = y.view(1, 1, height, 1).expand(batch, 1, height, width)
        x_coord = x_coord.view(1, 1, 1, width).expand(batch, 1, height, width)
        return torch.cat((x, y, x_coord), dim=1)


class BTRDSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError(f"kernel_size must be 3 or 7, got {kernel_size}")
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x):
        average = x.mean(dim=1, keepdim=True)
        maximum = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat((average, maximum), dim=1)))


class BTRDEfficientMultiScaleAttention(nn.Module):
    """EMA post-fusion block copied from RAFDE's official BTRD path."""

    def __init__(self, channels, factor=32):
        super().__init__()
        if channels % factor != 0:
            raise ValueError(
                f"BTRD EMA requires channels divisible by factor, got {channels} and {factor}"
            )
        channels_per_group = channels // factor
        self.groups = factor
        self.channels_per_group = channels_per_group
        self.softmax = nn.Softmax(dim=-1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.norm = nn.GroupNorm(channels_per_group, channels_per_group)
        self.conv1 = nn.Conv2d(channels_per_group, channels_per_group, 1)
        self.conv3 = nn.Conv2d(channels_per_group, channels_per_group, 3, padding=1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        grouped = x.reshape(batch * self.groups, self.channels_per_group, height, width)
        pooled_h = self.pool_h(grouped)
        pooled_w = self.pool_w(grouped).permute(0, 1, 3, 2)
        fused = self.conv1(torch.cat((pooled_h, pooled_w), dim=2))
        fused_h, fused_w = torch.split(fused, (height, width), dim=2)
        branch1 = self.norm(
            grouped
            * fused_h.sigmoid()
            * fused_w.permute(0, 1, 3, 2).sigmoid()
        )
        branch2 = self.conv3(grouped)
        weight1 = self.softmax(
            self.global_pool(branch1).reshape(batch * self.groups, 1, -1)
        )
        value1 = branch2.reshape(batch * self.groups, self.channels_per_group, -1)
        weight2 = self.softmax(
            self.global_pool(branch2).reshape(batch * self.groups, 1, -1)
        )
        value2 = branch1.reshape(batch * self.groups, self.channels_per_group, -1)
        weights = (
            torch.matmul(weight1, value1) + torch.matmul(weight2, value2)
        ).reshape(batch * self.groups, 1, height, width)
        return (grouped * weights.sigmoid()).reshape(batch, channels, height, width)


class FrequencySpatialDynamicDownsample(nn.Module):
    """Frequency-spatial dynamic downsampling for one HGNetv2 transition.

    This is a formula-level migration of FSD-Down rather than an official-code
    reproduction. A fixed orthonormal Haar transform preserves the four 2x2
    frequency bands, while a learned depthwise stride-2 branch supplies a
    complementary spatial representation. Global channel gates are computed
    from both branches, the frequency branch is scaled by a learnable ``rho``,
    and a final 1x1 projection restores the original C -> C stage interface.

    ``frequency_mode`` is evaluation-only causal instrumentation. Every mode
    uses the exact same learned checkpoint; it only changes which frequency
    evidence reaches the trained fusion operator.
    """

    VALID_FREQUENCY_MODES = {
        "full",
        "ll_only",
        "shift_hf",
        "phase_permute",
        "spatial_only",
        "target_hf_zero",
        "matched_background_hf_zero",
    }

    def __init__(self, channels):
        super().__init__()
        self.channels = int(channels)
        expanded_channels = 4 * self.channels
        self.spatial_depthwise = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=2,
            stride=2,
            groups=self.channels,
            bias=False,
        )
        self.spatial_expand = nn.Conv2d(
            self.channels, expanded_channels, kernel_size=1, bias=False
        )
        self.rho = nn.Parameter(torch.ones((), dtype=torch.float32))
        self.output_projection = nn.Conv2d(
            expanded_channels, self.channels, kernel_size=1, bias=False
        )
        # HGNetv2's replaced downsample ends in BN and has no activation.
        self.output_norm = nn.BatchNorm2d(self.channels)

        self.frequency_mode = "full"
        # Evaluation-only GT intervention used to distinguish target-edge
        # frequency dependence from equally energetic background texture.
        self.frequency_intervention_mask = None
        self.last_frequency_gate_mean = None
        self.last_spatial_gate_mean = None
        self.last_rho = None
        self.last_frequency_rms = None
        self.last_spatial_rms = None
        self.last_intervention_target_energy = None
        self.last_intervention_removed_energy = None
        self.last_intervention_removed_energy_ratio = None
        self.last_intervention_cell_ratio = None

    @staticmethod
    def haar_dwt(x):
        """Return orthonormal LL/LH/HL/HH bands at half resolution."""
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(
                "FSD-Down requires even input height/width, "
                f"got {tuple(x.shape[-2:])}"
            )
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 + x01 - x10 + x11) * 0.5
        hl = (-x00 - x01 + x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    def _intervene_frequency(self, bands):
        ll, lh, hl, hh = bands
        mode = self.frequency_mode
        self.last_intervention_target_energy = None
        self.last_intervention_removed_energy = None
        self.last_intervention_removed_energy_ratio = None
        self.last_intervention_cell_ratio = None
        if mode == "full":
            return torch.cat((ll, lh, hl, hh), dim=1)
        if mode == "ll_only":
            zeros = torch.zeros_like(ll)
            return torch.cat((ll, zeros, zeros, zeros), dim=1)
        if mode == "shift_hf":
            shift = (max(1, ll.shape[-2] // 2), max(1, ll.shape[-1] // 2))
            high = [torch.roll(band, shifts=shift, dims=(-2, -1)) for band in (lh, hl, hh)]
            return torch.cat((ll, *high), dim=1)
        if mode == "phase_permute":
            return torch.cat((ll, hl, hh, lh), dim=1)
        if mode == "spatial_only":
            return torch.zeros(
                ll.shape[0],
                4 * ll.shape[1],
                ll.shape[2],
                ll.shape[3],
                device=ll.device,
                dtype=ll.dtype,
            )
        if mode in {"target_hf_zero", "matched_background_hf_zero"}:
            if self.frequency_intervention_mask is None:
                raise RuntimeError(
                    f"{mode} requires frequency_intervention_mask from the evaluator"
                )
            target_mask = self.frequency_intervention_mask
            if target_mask.shape[-2:] != ll.shape[-2:]:
                target_mask = F.interpolate(
                    target_mask.float(), size=ll.shape[-2:], mode="nearest"
                ).bool()
            else:
                target_mask = target_mask.bool()
            if target_mask.shape[0] != ll.shape[0]:
                raise RuntimeError(
                    "frequency intervention batch mismatch: "
                    f"mask={target_mask.shape[0]}, feature={ll.shape[0]}"
                )
            high_energy = (
                lh.float().square()
                + hl.float().square()
                + hh.float().square()
            ).sum(dim=1, keepdim=True)
            if mode == "target_hf_zero":
                intervention_mask = target_mask
            else:
                # Select the strongest non-adjacent background cells until the
                # removed high-frequency energy matches the target-region
                # budget for each image. This is stricter than equal area.
                excluded = F.max_pool2d(target_mask.float(), 3, 1, 1).bool()
                intervention_mask = torch.zeros_like(target_mask)
                for batch_index in range(ll.shape[0]):
                    target_energy = high_energy[batch_index][target_mask[batch_index]].sum()
                    candidate = (~excluded[batch_index]).flatten()
                    candidate_indices = candidate.nonzero(as_tuple=False).flatten()
                    if candidate_indices.numel() == 0 or float(target_energy) <= 0.0:
                        continue
                    candidate_energy = high_energy[batch_index, 0].flatten()[candidate_indices]
                    order = torch.argsort(candidate_energy, descending=True)
                    ordered_energy = candidate_energy[order]
                    cumulative = ordered_energy.cumsum(0)
                    selected_count = int(
                        torch.searchsorted(cumulative, target_energy).clamp(
                            max=max(0, cumulative.numel() - 1)
                        ).item()
                    ) + 1
                    selected = candidate_indices[order[:selected_count]]
                    intervention_mask[batch_index, 0].flatten()[selected] = True
            mask_float = intervention_mask.to(lh.dtype)
            keep = 1.0 - mask_float
            target_energy = (high_energy * target_mask.float()).sum()
            removed_energy = (high_energy * intervention_mask.float()).sum()
            total_energy = high_energy.sum().clamp_min(1e-12)
            self.last_intervention_target_energy = target_energy.detach()
            self.last_intervention_removed_energy = removed_energy.detach()
            self.last_intervention_removed_energy_ratio = (
                removed_energy / total_energy
            ).detach()
            self.last_intervention_cell_ratio = intervention_mask.float().mean().detach()
            return torch.cat((ll, lh * keep, hl * keep, hh * keep), dim=1)
        raise ValueError(
            f"unsupported FSD-Down frequency_mode {mode!r}; "
            f"expected one of {sorted(self.VALID_FREQUENCY_MODES)}"
        )

    def forward(self, x):
        frequency = self._intervene_frequency(self.haar_dwt(x))
        spatial = self.spatial_expand(self.spatial_depthwise(x))
        gates = torch.sigmoid(
            F.adaptive_avg_pool2d(torch.cat((frequency, spatial), dim=1), 1)
        )
        frequency_gate, spatial_gate = gates.chunk(2, dim=1)
        fused = (
            self.rho.to(frequency.dtype) * frequency_gate * frequency
            + spatial_gate * spatial
        )
        output = self.output_norm(self.output_projection(fused))

        self.last_frequency_gate_mean = frequency_gate.detach().mean()
        self.last_spatial_gate_mean = spatial_gate.detach().mean()
        self.last_rho = self.rho.detach()
        self.last_frequency_rms = frequency.detach().float().square().mean().sqrt()
        self.last_spatial_rms = spatial.detach().float().square().mean().sqrt()
        return output


class ReinitializedStandardDownsample(nn.Module):
    """Standard operator under a deliberately new checkpoint-key namespace.

    This control has the same operator as A00, but its parameters cannot match
    the COCO checkpoint's original ``downsample.conv/bn`` keys. It therefore
    isolates transition reinitialization from FSD-Down's architectural effect.
    """

    def __init__(self, channels, use_lab=False):
        super().__init__()
        self.reinitialized_path = ConvBNAct(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            groups=channels,
            use_act=False,
            use_lab=use_lab,
        )

    def forward(self, x):
        return self.reinitialized_path(x)


class BoundaryTransitionRegionDownsample(nn.Module):
    """Source-equivalent RAFDE BTRD adapted to HGNetv2's C -> C interface.

    The learned max-vs-average activation difference estimates boundary
    transition locations.  Stable and transitional responses are routed to a
    learned convolution path and a max-pooling path respectively, before the
    official EMA fusion.  This module replaces only the stride-2 operation;
    the following HG stage remains unchanged.
    """

    def __init__(self, channels, ema_factor=32):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(f"BTRD requires an even channel count, got {channels}")
        self.add_coords = BTRDAddCoords()
        self.input_projection = BTRDConvBNAct(channels + 2, channels, 1)
        self.activation_projection = nn.Conv2d(channels, 1, kernel_size=1)
        self.spatial_attention = BTRDSpatialAttention(kernel_size=7)
        self.stable_downsample = BTRDConvBNAct(channels // 2, channels // 2, 3, stride=2)
        self.transition_downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fusion = BTRDEfficientMultiScaleAttention(channels, factor=ema_factor)
        self.transition_mode = "learned"
        self.last_learned_transition_map = None
        self.last_transition_map = None

    def forward(self, x):
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(
                "BTRD's source geometry requires even input height/width, "
                f"got {tuple(x.shape[-2:])}"
            )
        x = self.input_projection(self.add_coords(x))
        activation = self.activation_projection(x)
        activation = activation * self.spatial_attention(activation)

        activation_average = F.avg_pool2d(
            activation, kernel_size=2, stride=1, count_include_pad=True
        )
        activation_maximum = F.max_pool2d(activation, kernel_size=2, stride=1)
        transition = torch.sigmoid(
            F.relu(torch.sigmoid(activation_maximum) - torch.sigmoid(activation_average))
        )
        self.last_learned_transition_map = transition.detach()
        if self.transition_mode == "constant":
            transition = transition.mean(dim=(-2, -1), keepdim=True).expand_as(transition)
        elif self.transition_mode == "shuffled":
            # A deterministic half-map translation preserves every value and
            # the global transition budget while breaking image correspondence.
            transition = torch.roll(
                transition,
                shifts=(transition.shape[-2] // 2, transition.shape[-1] // 2),
                dims=(-2, -1),
            )
        elif self.transition_mode != "learned":
            raise ValueError(f"unsupported BTRD transition_mode: {self.transition_mode}")

        feature_average = F.avg_pool2d(
            x, kernel_size=2, stride=1, count_include_pad=True
        )
        feature_maximum = F.max_pool2d(x, kernel_size=2, stride=1)
        routed = feature_average + feature_maximum * transition
        stable, transitional = routed.chunk(2, dim=1)
        stable = self.stable_downsample(stable * (1.0 - transition))
        transitional = self.transition_downsample(transitional * transition)
        self.last_transition_map = transition.detach()
        return self.fusion(torch.cat((stable, transitional), dim=1))


class PositionDirectionalBoundaryResidual(nn.Module):
    """Preserve boundary position and orientation beside a stable downsample.

    The standard HGNetv2 stride-2 depthwise convolution remains the semantic
    path.  A narrow residual path extracts horizontal/vertical changes at the
    input resolution, locates the two corresponding box-side families, and
    uses PixelUnshuffle only to retain the 2x2 sub-pixel phase before a 1x1
    projection.  This is deliberately different from replacing the complete
    downsample with SPD: phase is retained only for boundary evidence.

    ``mask_mode`` and ``detail_mode`` are causal-audit controls.  Normal
    training uses ``learned`` and ``directional`` respectively.
    """

    def __init__(
        self,
        in_channels,
        detail_channels=16,
        max_scale=0.5,
        init_scale=0.05,
    ):
        super().__init__()
        if detail_channels <= 0:
            raise ValueError("PDBR detail_channels must be positive")
        if not 0.0 < init_scale < max_scale:
            raise ValueError("PDBR requires 0 < init_scale < max_scale")

        self.in_channels = int(in_channels)
        self.detail_channels = int(detail_channels)
        self.max_scale = float(max_scale)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, detail_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(detail_channels),
            nn.SiLU(inplace=True),
        )

        def make_locator():
            return nn.Sequential(
                nn.Conv2d(
                    detail_channels,
                    detail_channels,
                    kernel_size=3,
                    padding=1,
                    groups=detail_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(detail_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(detail_channels, 1, kernel_size=1, bias=True),
            )

        self.locator_x = make_locator()
        self.locator_y = make_locator()
        self.project = nn.Sequential(
            nn.Conv2d(8 * detail_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        init_ratio = init_scale / max_scale
        self.scale_logit = nn.Parameter(
            torch.tensor(math.atanh(init_ratio), dtype=torch.float32)
        )

        self.mask_mode = "learned"
        self.detail_mode = "directional"
        self.mask_override = None
        self.shift_fraction = 0.5
        self.last_boundary_logits = None
        self.last_masks = None
        self.last_scale = None
        self.last_residual_rms_ratio = None

    @staticmethod
    def _pad_even(x):
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x

    def _resolve_masks(self, logits_x, logits_y):
        mask_x, mask_y = logits_x.sigmoid(), logits_y.sigmoid()
        if self.mask_mode == "one":
            mask_x, mask_y = torch.ones_like(mask_x), torch.ones_like(mask_y)
        elif self.mask_mode == "zero":
            mask_x, mask_y = torch.zeros_like(mask_x), torch.zeros_like(mask_y)
        elif self.mask_mode == "crossed":
            mask_x, mask_y = mask_y, mask_x
        elif self.mask_mode == "shifted":
            shift_y = max(1, int(round(mask_x.shape[-2] * self.shift_fraction)))
            shift_x = max(1, int(round(mask_x.shape[-1] * self.shift_fraction)))
            mask_x = torch.roll(mask_x, shifts=shift_x, dims=-1)
            mask_y = torch.roll(mask_y, shifts=shift_y, dims=-2)
        elif self.mask_mode == "oracle":
            if self.mask_override is None:
                raise RuntimeError("PDBR oracle mode requires mask_override=(Mx, My)")
            mask_x, mask_y = self.mask_override
            mask_x = F.interpolate(
                mask_x.float(), size=logits_x.shape[-2:], mode="bilinear", align_corners=False
            ).to(device=logits_x.device, dtype=logits_x.dtype)
            mask_y = F.interpolate(
                mask_y.float(), size=logits_y.shape[-2:], mode="bilinear", align_corners=False
            ).to(device=logits_y.device, dtype=logits_y.dtype)
        elif self.mask_mode != "learned":
            raise ValueError(f"unsupported PDBR mask_mode: {self.mask_mode}")
        return mask_x, mask_y

    def forward(self, x, standard):
        z = self.reduce(x)
        dx = z - F.avg_pool2d(z, kernel_size=(1, 3), stride=1, padding=(0, 1))
        dy = z - F.avg_pool2d(z, kernel_size=(3, 1), stride=1, padding=(1, 0))
        logits_x = self.locator_x(dx.abs())
        logits_y = self.locator_y(dy.abs())
        mask_x, mask_y = self._resolve_masks(logits_x, logits_y)

        if self.detail_mode == "directional":
            detail_x, detail_y = dx, dy
        elif self.detail_mode == "isotropic":
            isotropic = z - F.avg_pool2d(z, kernel_size=3, stride=1, padding=1)
            detail_x = detail_y = isotropic
        else:
            raise ValueError(f"unsupported PDBR detail_mode: {self.detail_mode}")

        boundary_x = self._pad_even(mask_x * detail_x)
        boundary_y = self._pad_even(mask_y * detail_y)
        phases = torch.cat(
            (
                F.pixel_unshuffle(boundary_x, downscale_factor=2),
                F.pixel_unshuffle(boundary_y, downscale_factor=2),
            ),
            dim=1,
        )
        residual = self.project(phases)
        if residual.shape[-2:] != standard.shape[-2:]:
            residual = residual[..., : standard.shape[-2], : standard.shape[-1]]
        scale = self.max_scale * torch.tanh(self.scale_logit)
        output = standard + scale.to(residual.dtype) * torch.tanh(residual)

        self.last_boundary_logits = (logits_x, logits_y)
        self.last_masks = (mask_x.detach(), mask_y.detach())
        self.last_scale = scale.detach()
        base_rms = standard.detach().float().square().mean().sqrt().clamp_min(1e-12)
        residual_rms = (scale * torch.tanh(residual)).detach().float().square().mean().sqrt()
        self.last_residual_rms_ratio = (residual_rms / base_rms).detach()
        return output


class TargetNeighborhoodDetailReconstruction(nn.Module):
    """Training-only decoder for S8 detail retention at the S16 bottleneck.

    This head never changes the detector feature.  It asks the complete S16
    stage output to reconstruct three non-negative relative-detail maps from
    the pre-downsample S8 feature.  The 2x upsampling is implemented by a
    sub-pixel decoder, and the head is physically removable for deployment.
    """

    def __init__(self, out_channels, target_max=3.0):
        super().__init__()
        if target_max <= 0:
            raise ValueError("TNDP target_max must be positive")
        self.target_max = float(target_max)
        # Avoid the name ``decoder``: the repository optimizer uses a regex
        # for the detector decoder, and that name would put these backbone
        # parameters into two optimizer groups.
        self.reconstructor = nn.Conv2d(out_channels, 3 * 4, kernel_size=1, bias=True)
        nn.init.normal_(self.reconstructor.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.reconstructor.bias)

    @staticmethod
    def _relative_detail(x, kernel_size):
        if kernel_size == (3, 3):
            smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        elif kernel_size == (1, 3):
            smooth = F.avg_pool2d(x, kernel_size=(1, 3), stride=1, padding=(0, 1))
        elif kernel_size == (3, 1):
            smooth = F.avg_pool2d(x, kernel_size=(3, 1), stride=1, padding=(1, 0))
        else:
            raise ValueError(f"unsupported TNDP kernel: {kernel_size}")
        numerator = (x - smooth).abs().mean(dim=1, keepdim=True)
        denominator = x.abs().mean(dim=1, keepdim=True).clamp_min(1e-4)
        return torch.log1p(numerator / denominator)

    def make_target(self, x):
        with torch.no_grad():
            target = torch.cat(
                (
                    self._relative_detail(x.float(), (3, 3)),
                    self._relative_detail(x.float(), (1, 3)),
                    self._relative_detail(x.float(), (3, 1)),
                ),
                dim=1,
            )
            return target.clamp_(0.0, self.target_max)

    def predict(self, y):
        prediction = F.pixel_shuffle(self.reconstructor(y), upscale_factor=2)
        return F.softplus(prediction.float())


class BoundaryPolyphaseCarrier(nn.Module):
    """Carry SAM-supervised native S8 phase residuals across S8->S16.

    SAM masks never enter this module.  A lightweight student predicts a soft
    S8 boundary gate.  Detection gradients are stopped at that gate, while the
    transported values remain native pre-downsample features.  The grouped
    output projection is zero initialized so construction is exactly the
    standard HGNetv2 detector.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        detail_channels=32,
        project_groups=8,
        max_scale=0.25,
        init_scale=0.05,
    ):
        super().__init__()
        if detail_channels <= 0:
            raise ValueError("BPC detail_channels must be positive")
        if not 0.0 < init_scale < max_scale:
            raise ValueError("BPC requires 0 < init_scale < max_scale")
        projection_groups = math.gcd(
            math.gcd(4 * int(detail_channels), int(out_channels)),
            int(project_groups),
        )
        if projection_groups <= 0:
            raise ValueError("BPC project_groups must be positive")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.detail_channels = int(detail_channels)
        self.project_groups = int(projection_groups)
        self.max_scale = float(max_scale)
        self.reduce = nn.Conv2d(
            self.in_channels, self.detail_channels, kernel_size=1, bias=False
        )
        self.locator = nn.Sequential(
            nn.Conv2d(
                self.detail_channels,
                self.detail_channels,
                kernel_size=3,
                padding=1,
                groups=self.detail_channels,
                bias=False,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.detail_channels, 1, kernel_size=1, bias=True),
        )
        nn.init.constant_(self.locator[-1].bias, -4.0)
        self.project = nn.Conv2d(
            4 * self.detail_channels,
            self.out_channels,
            kernel_size=1,
            groups=self.project_groups,
            bias=False,
        )
        nn.init.zeros_(self.project.weight)
        init_ratio = float(init_scale) / self.max_scale
        self.scale_logit = nn.Parameter(
            torch.tensor(math.atanh(init_ratio), dtype=torch.float32)
        )

        self.gate_mode = "learned"
        # Evaluation-only controls used by the registered BPC1 amplitude
        # probe.  Their defaults preserve the trained model exactly.
        self.intervention_gate_logit_bias = 0.0
        self.intervention_output_gain = 1.0
        self.intervention_linear_residual = False
        self.last_boundary_logits = None
        self.last_gate = None
        self.last_scale = None
        self.last_residual_rms_ratio = None

    @staticmethod
    def _pad_even(x):
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x

    @staticmethod
    def phase_residual(z):
        phases = F.pixel_unshuffle(BoundaryPolyphaseCarrier._pad_even(z), 2)
        batch, _, height, width = phases.shape
        phases = phases.view(batch, z.shape[1], 4, height, width)
        return phases - phases.mean(dim=2, keepdim=True)

    def _resolve_gate(self, logits):
        gate = (logits + float(self.intervention_gate_logit_bias)).sigmoid()
        if self.gate_mode == "zero":
            gate = torch.zeros_like(gate)
        elif self.gate_mode == "one":
            gate = torch.ones_like(gate)
        elif self.gate_mode == "shifted":
            gate = torch.roll(
                gate,
                shifts=(gate.shape[-2] // 2, gate.shape[-1] // 2),
                dims=(-2, -1),
            )
        elif self.gate_mode == "constant":
            gate = gate.mean(dim=(-2, -1), keepdim=True).expand_as(gate)
        elif self.gate_mode != "learned":
            raise ValueError(f"unsupported BPC gate_mode: {self.gate_mode}")
        return gate

    def forward(self, x, standard):
        # The duplicated reduction exists only in training: SAM supervision may
        # train the adapter weights but is not allowed to modify the shared X8
        # backbone representation.  Inference computes the reduction once.
        z = self.reduce(x)
        locator_z = self.reduce(x.detach()) if self.training else z
        logits = self.locator(locator_z)
        gate = self._resolve_gate(logits).detach()

        phase_detail = self.phase_residual(z)
        gate_phases = F.pixel_unshuffle(self._pad_even(gate), 2).unsqueeze(1)
        gated = (phase_detail * gate_phases).flatten(1, 2)
        residual = self.project(gated)
        if residual.shape[-2:] != standard.shape[-2:]:
            residual = residual[..., : standard.shape[-2], : standard.shape[-1]]
        scale = self.max_scale * torch.tanh(self.scale_logit)
        carrier = residual if self.intervention_linear_residual else torch.tanh(residual)
        enhancement = (
            scale.to(residual.dtype)
            * float(self.intervention_output_gain)
            * carrier
        )
        output = standard + enhancement

        self.last_boundary_logits = logits
        self.last_gate = gate
        self.last_scale = scale.detach()
        base_rms = standard.detach().float().square().mean().sqrt().clamp_min(1e-12)
        residual_rms = enhancement.detach().float().square().mean().sqrt()
        self.last_residual_rms_ratio = (residual_rms / base_rms).detach()
        return output


class HG_Stage(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        block_num,
        layer_num,
        downsample=True,
        light_block=False,
        kernel_size=3,
        use_lab=False,
        agg="se",
        drop_path=0.0,
        preserve_mode="none",
        preserve_ratio=0.0,
        preserve_groups=4,
        preserve_init_scale=0.1,
        downsample_mode="standard",
        lad_groups=8,
        lad_target_gate=False,
        lad_candidate_mode="learned_conv",
        lad_fusion_mode="replace",
        lad_route_init=0.1,
        lad_target_gate_mode="learned",
        preblur=False,
        use_pat_sf=False,
        pat_sf_n_div=4,
        pat_sf_variant="self",
        pdbr_enabled=False,
        pdbr_detail_channels=16,
        pdbr_max_scale=0.5,
        pdbr_init_scale=0.05,
    ):
        super().__init__()
        self.downsample = downsample
        self.conditional_detail = None
        if downsample and downsample_mode == "fsd":
            # Preserve the RNG stream used by unchanged downstream blocks.
            cpu_rng_state = torch.random.get_rng_state()
            try:
                self.downsample = FrequencySpatialDynamicDownsample(in_chs)
            finally:
                torch.random.set_rng_state(cpu_rng_state)
        elif downsample and downsample_mode == "standard_reinit":
            cpu_rng_state = torch.random.get_rng_state()
            try:
                self.downsample = ReinitializedStandardDownsample(in_chs, use_lab=use_lab)
            finally:
                torch.random.set_rng_state(cpu_rng_state)
        elif downsample and downsample_mode == "btrd":
            self.downsample = BoundaryTransitionRegionDownsample(in_chs)
        elif downsample and downsample_mode == "lad":
            self.downsample = LightAdaptiveWeightDownsample(
                in_chs,
                groups=lad_groups,
                target_gate=lad_target_gate,
                candidate_mode=lad_candidate_mode,
            )
        elif downsample and downsample_mode == "conditional_lad":
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
            )
            self.conditional_detail = TargetConditionalLADDetail(
                in_chs,
                groups=lad_groups,
                route_init=lad_route_init,
                target_gate_mode=lad_target_gate_mode,
            )
        elif downsample and downsample_mode == "standard":
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
            )
        elif not downsample:
            self.downsample = nn.Identity()
        else:
            raise ValueError(f"unsupported downsample_mode: {downsample_mode}")

        self.preblur = FixedBlurFilter(in_chs) if downsample and preblur else nn.Identity()

        if downsample and preserve_mode != "none":
            self.preserve_downsample = PartialInformationPreservingDownsample(
                in_chs,
                ratio=preserve_ratio,
                groups=preserve_groups,
                mode=preserve_mode,
                init_scale=preserve_init_scale,
            )
        else:
            self.preserve_downsample = None

        self.pdbr = (
            PositionDirectionalBoundaryResidual(
                in_channels=in_chs,
                detail_channels=pdbr_detail_channels,
                max_scale=pdbr_max_scale,
                init_scale=pdbr_init_scale,
            )
            if downsample and pdbr_enabled
            else None
        )

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                    use_pat_sf=use_pat_sf,
                    pat_sf_n_div=pat_sf_n_div,
                    pat_sf_variant=pat_sf_variant,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        identity = x
        x = self.preblur(x)
        x = self.downsample(x)
        if getattr(self, 'sbra', None) is not None:
            x = self.sbra(identity, x)
        if self.conditional_detail is not None:
            x = self.conditional_detail(identity, x)
        if self.preserve_downsample is not None:
            x = self.preserve_downsample(identity, x)
        if self.pdbr is not None:
            x = self.pdbr(identity, x)
        x = self.blocks(x)
        return x


class LowRankComplementaryDownsample(nn.Module):
    """Low-cost S8-to-S16 residual used after physical S16 slimming.

    The depthwise operation performs the spatial reduction before the two
    pointwise projections.  Consequently, the expensive channel mixing runs
    on the smaller S16 grid.  The branch keeps the main feature width and is
    fused by addition, so it does not restore the removed 512-channel path.
    """

    def __init__(
        self,
        in_chs,
        out_chs,
        hidden_chs=32,
        init_scale=0.0,
        use_lab=False,
    ):
        super().__init__()
        hidden_chs = int(hidden_chs)
        if hidden_chs <= 0:
            raise ValueError(f"hidden_chs must be positive, got {hidden_chs}")
        self.spatial = ConvBNAct(
            in_chs,
            in_chs,
            kernel_size=3,
            stride=2,
            groups=in_chs,
            use_act=True,
            use_lab=use_lab,
        )
        self.reduce = ConvBNAct(
            in_chs,
            hidden_chs,
            kernel_size=1,
            use_act=True,
            use_lab=use_lab,
        )
        self.expand = ConvBNAct(
            hidden_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
        )
        # Scalar routing avoids introducing a per-channel dynamic gate.  A
        # zero start makes the inserted branch function-preserving at step 0.
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        x = self.spatial(x)
        x = self.reduce(x)
        x = self.expand(x)
        return self.scale * x


class SPARBasicConv2d(nn.Module):
    """Conv-BN block used by the official FALCON-SPAR feature fusion.

    The released FALCON code intentionally returns the BatchNorm output
    without a ReLU.  REP1 preserves that behavior instead of silently
    replacing the published activation map.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class SPARFuseFeatures(nn.Module):
    """Scale-matched reproduction of FALCON-SFOD ``FuseFeatures``.

    ``deep``, ``middle`` and ``shallow`` correspond to S32, S16 and S8.
    Only the channel width is configurable so the published fusion topology
    can be used with the much smaller HGNetv2-B0 detector.
    """

    def __init__(self, deep_channels, middle_channels, shallow_channels, channels):
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError("SPAR fusion channels must be positive")
        self.conv1 = SPARBasicConv2d(deep_channels, channels, 1)
        self.conv2 = SPARBasicConv2d(middle_channels, channels, 1)
        self.conv3 = SPARBasicConv2d(shallow_channels, channels, 1)

        self.conv_upsample1 = SPARBasicConv2d(channels, channels, 3, padding=1)
        self.conv_upsample2 = SPARBasicConv2d(channels, channels, 3, padding=1)
        self.conv_upsample3 = SPARBasicConv2d(channels, channels, 3, padding=1)
        self.conv_upsample4 = SPARBasicConv2d(
            2 * channels, 2 * channels, 3, padding=1
        )

        self.conv_concat2 = SPARBasicConv2d(
            2 * channels, 2 * channels, 3, padding=1
        )
        self.conv_concat3 = SPARBasicConv2d(
            3 * channels, 3 * channels, 3, padding=1
        )
        self.conv4 = SPARBasicConv2d(3 * channels, 3 * channels, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channels, channels, kernel_size=1)

    def forward(self, deep, middle, shallow):
        x1 = self.conv1(deep)
        x2 = self.conv2(middle)
        x3 = self.conv3(shallow)

        x1_up = F.interpolate(
            x1, size=x2.shape[-2:], mode="bilinear", align_corners=True
        )
        x2_1 = self.conv_upsample1(x1_up) * x2

        x2_up = F.interpolate(
            x2_1, size=x3.shape[-2:], mode="bilinear", align_corners=True
        )
        x3_1 = self.conv_upsample2(x2_up) * x3

        x1_up2 = F.interpolate(
            x1, size=x2.shape[-2:], mode="bilinear", align_corners=True
        )
        x2_2 = torch.cat((x2_1, self.conv_upsample3(x1_up2)), dim=1)
        x2_2 = self.conv_concat2(x2_2)

        x2_2_up = F.interpolate(
            x2_2, size=x3.shape[-2:], mode="bilinear", align_corners=True
        )
        x3_2 = torch.cat((x3_1, self.conv_upsample4(x2_2_up)), dim=1)
        x3_2 = self.conv_concat3(x3_2)
        return self.conv5(self.conv4(x3_2))


class SABRScaleHeads(nn.Module):
    """Training-only scale-allocated body/boundary readouts.

    The real S8 stage predicts a soft boundary band while S16 predicts soft
    foreground occupancy.  These heads never alter detector features and are
    absent from the inference outputs.
    """

    def __init__(self, s8_channels, s16_channels, hidden_channels=32):
        super().__init__()
        hidden_channels = int(hidden_channels)
        if hidden_channels <= 0:
            raise ValueError("SABR hidden channels must be positive")
        self.boundary_head = nn.Sequential(
            SPARBasicConv2d(s8_channels, hidden_channels, 3, padding=1),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        self.body_head = nn.Sequential(
            SPARBasicConv2d(s16_channels, hidden_channels, 3, padding=1),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, s8, s16):
        return {
            "boundary_logits": self.boundary_head(s8),
            "body_logits": self.body_head(s16),
        }


class SBOXExtremePointHead(nn.Module):
    """Training-only four-direction extreme-point predictor on real S8."""

    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()
        hidden_channels = int(hidden_channels)
        if hidden_channels <= 0:
            raise ValueError("S-BOX hidden channels must be positive")
        groups = min(8, hidden_channels)
        while hidden_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 4, kernel_size=1, bias=True),
        )
        # Sparse heatmaps start with a conservative foreground prior.  This
        # head never enters the detector forward path, so the initialization
        # cannot alter A00 predictions.
        nn.init.constant_(self.block[-1].bias, -2.1972246)

    def forward(self, s8):
        return self.block(s8)


@register()
class HGNetv2(nn.Module):
    """
    HGNetV2
    Args:
        stem_channels: list. Number of channels for the stem block.
        stage_type: str. The stage configuration of HGNet. such as the number of channels, stride, etc.
        use_lab: boolean. Whether to use LearnableAffineBlock in network.
        lr_mult_list: list. Control the learning rate of different stages.
    Returns:
        model: nn.Layer. Specific HGNetV2 model depends on args.
    """

    arch_configs = {
        "B0": {
            "stem_channels": [3, 16, 16],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth",
        },
        "B1": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth",
        },
        "B2": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth",
        },
        "B3": {
            "stem_channels": [3, 24, 32],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth",
        },
        "B4": {
            "stem_channels": [3, 32, 48],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth",
        },
        "B5": {
            "stem_channels": [3, 32, 64],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth",
        },
        "B6": {
            "stem_channels": [3, 48, 96],
            "stage_config": {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            "url": "https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth",
        },
    }

    def __init__(
        self,
        name,
        use_lab=False,
        return_idx=[1, 2, 3],
        freeze_stem_only=True,
        freeze_at=0,
        freeze_norm=True,
        pretrained=True,
        local_model_dir="weight/hgnetv2/",
        preserve_stage=-1,
        preserve_mode="none",
        preserve_ratio=0.0,
        preserve_groups=4,
        preserve_init_scale=0.1,
        spatial_aux_stage=-1,
        btrd_stage=-1,
        fsd_stage=-1,
        reinit_downsample_stage=-1,
        lad_stage=-1,
        lad_groups=8,
        lad_target_gate=False,
        lad_candidate_mode="learned_conv",
        lad_fusion_mode="replace",
        lad_route_init=0.1,
        lad_target_gate_mode="learned",
        blur_stage=-1,
        srtod_stage=-1,
        srtod_threshold=0.0156862,
        srtod_student_only=False,
        srtod_student_hidden_channels=32,
        srtod_enhancer="dgfe",
        srtod_detail_groups=16,
        srtod_detail_init_scale=0.0,
        rpc_stage=-1,
        rpc_hidden_channels=32,
        rpc_init_scale=0.0,
        pat_stage=-1,
        pat_n_div=4,
        pat_sf_stage=-1,
        pat_sf_n_div=4,
        pat_sf_variant="self",
        pdbr_stage=-1,
        pdbr_detail_channels=16,
        pdbr_max_scale=0.5,
        pdbr_init_scale=0.05,
        tndp_stage=-1,
        tndp_target_max=3.0,
        bpc_stage=-1,
        bpc_detail_channels=32,
        bpc_project_groups=8,
        bpc_max_scale=0.25,
        bpc_init_scale=0.05,
        qrl_source_stage=-1,
        mdqa_source_stage=-1,
        sqmi_source_stage=-1,
        spar_enabled=False,
        spar_fusion_channels=64,
        spar_gradient_mode="full",
        spar_gradient_ramp_start=5,
        spar_gradient_ramp_end=10,
        spar_gradient_max_scale=1.0,
        sabr_enabled=False,
        sabr_hidden_channels=32,
        sbrd_enabled=False,
        qcsr_enabled=False,
        sbox_enabled=False,
        sbox_hidden_channels=32,
        sbox_gradient_ramp_start=5,
        sbox_gradient_ramp_end=10,
        sbox_gradient_max_scale=1.0,
    ):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx

        stem_channels = self.arch_configs[name]["stem_channels"]
        stage_config = self.arch_configs[name]["stage_config"]
        download_url = self.arch_configs[name]["url"]

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]
        self.spatial_aux_stage = spatial_aux_stage
        self.btrd_stage = int(btrd_stage)
        self.fsd_stage = int(fsd_stage)
        self.reinit_downsample_stage = int(reinit_downsample_stage)
        self.lad_stage = lad_stage
        self.srtod_stage = srtod_stage
        self.srtod_student_only = bool(srtod_student_only)
        self.srtod_enhancer = str(srtod_enhancer)
        self.rpc_stage = int(rpc_stage)
        self.pat_stage = int(pat_stage)
        self.pat_sf_stage = int(pat_sf_stage)
        self.pat_sf_variant = str(pat_sf_variant)
        self.pdbr_stage = int(pdbr_stage)
        self.tndp_stage = int(tndp_stage)
        self.bpc_stage = int(bpc_stage)
        self.qrl_source_stage = int(qrl_source_stage)
        self.mdqa_source_stage = int(mdqa_source_stage)
        self.sqmi_source_stage = int(sqmi_source_stage)
        self.spar_enabled = bool(spar_enabled)
        self.sabr_enabled = bool(sabr_enabled)
        self.sbox_enabled = bool(sbox_enabled)
        self.sbox_gradient_ramp_start = int(sbox_gradient_ramp_start)
        self.sbox_gradient_ramp_end = int(sbox_gradient_ramp_end)
        self.sbox_gradient_max_scale = float(sbox_gradient_max_scale)
        self.spar_gradient_mode = str(spar_gradient_mode)
        self.spar_gradient_ramp_start = int(spar_gradient_ramp_start)
        self.spar_gradient_ramp_end = int(spar_gradient_ramp_end)
        self.spar_gradient_max_scale = float(spar_gradient_max_scale)
        self.training_epoch = 0
        if self.spar_gradient_mode not in {"full", "progressive"}:
            raise ValueError("spar_gradient_mode must be full or progressive")
        if (
            self.spar_gradient_mode == "progressive"
            and not 0 <= self.spar_gradient_ramp_start < self.spar_gradient_ramp_end
        ):
            raise ValueError(
                "Progressive SPAR gradient requires 0 <= ramp_start < ramp_end"
            )
        if not 0.0 < self.spar_gradient_max_scale <= 1.0:
            raise ValueError("spar_gradient_max_scale must be in (0, 1]")
        if not 0 <= self.sbox_gradient_ramp_start < self.sbox_gradient_ramp_end:
            raise ValueError(
                "S-BOX gradient ramp requires 0 <= ramp_start < ramp_end"
            )
        if not 0.0 < self.sbox_gradient_max_scale <= 1.0:
            raise ValueError("sbox_gradient_max_scale must be in (0, 1]")
        self.sbrd_enabled = bool(sbrd_enabled)
        self.qcsr_enabled = bool(qcsr_enabled)
        self.spatial_importance_logits = None
        # S-QER1 requests an ephemeral reference to the native RGB S4 tensor.
        # DFINE enables this only on the RGB backbone and clears it after use.
        self.sqer_capture_s4 = False
        self.sqer_s4_source = None
        self.pdbr_boundary_logits = None
        self.tndp_detail_prediction = None
        self.tndp_detail_target = None
        self.bpc_boundary_logits = None
        self.qrl_source = None
        self.mdqa_source = None
        self.sqmi_source = None
        self.spar_fused_features = None
        self.sabr_outputs = None
        self.sbox_extreme_logits = None
        self.sbrd_stage_features = None
        self.qcsr_stage_features = None
        self.srtod_reconstruction_loss = None
        self.srtod_difference_map = None
        self.srtod_difference_mask = None
        self.spatial_aux_head = (
            nn.Conv2d(self._out_channels[spatial_aux_stage], 1, kernel_size=1, bias=True)
            if 0 <= spatial_aux_stage < len(self._out_channels)
            else None
        )
        if self.qrl_source_stage < -1 or self.qrl_source_stage >= len(self._out_channels):
            raise ValueError(
                "qrl_source_stage must be -1 or a valid stage index, "
                f"got {self.qrl_source_stage}"
            )
        if self.mdqa_source_stage < -1 or self.mdqa_source_stage >= len(self._out_channels):
            raise ValueError(
                "mdqa_source_stage must be -1 or a valid stage index, "
                f"got {self.mdqa_source_stage}"
            )
        if self.sqmi_source_stage < -1 or self.sqmi_source_stage >= len(self._out_channels):
            raise ValueError(
                "sqmi_source_stage must be -1 or a valid stage index, "
                f"got {self.sqmi_source_stage}"
            )
        if srtod_stage < -1 or srtod_stage >= len(self._out_channels):
            raise ValueError(
                f"srtod_stage must be -1 or a valid stage index, got {srtod_stage}"
            )
        if srtod_stage >= 0 and srtod_stage not in return_idx:
            raise ValueError(
                "SR-TOD DGFE must enhance a returned detection feature; "
                f"stage {srtod_stage} is not in return_idx={return_idx}"
            )
        if self.srtod_student_only and srtod_stage < 0:
            raise ValueError("srtod_student_only requires a valid srtod_stage")
        if self.srtod_enhancer not in {"dgfe", "haar_detail"}:
            raise ValueError(
                "srtod_enhancer must be 'dgfe' or 'haar_detail', "
                f"got {self.srtod_enhancer!r}"
            )
        self.srtod_rh = (
            RH(
                in_channels=self._out_channels[srtod_stage],
                out_channels=stem_channels[0],
            )
            if srtod_stage >= 0 and not self.srtod_student_only
            else None
        )
        self.srtod_deficiency_head = (
            DeficiencyPredictor(
                in_channels=self._out_channels[srtod_stage],
                hidden_channels=int(srtod_student_hidden_channels),
            )
            if srtod_stage >= 0 and self.srtod_student_only
            else None
        )
        self.srtod_dgfe = (
            DGFE(gate_channels=self._out_channels[srtod_stage])
            if srtod_stage >= 0 and self.srtod_enhancer == "dgfe"
            else None
        )
        self.srtod_detail_residual = None
        if srtod_stage >= 0 and self.srtod_enhancer == "haar_detail":
            stage_key = list(stage_config.keys())[srtod_stage]
            stage_in_channels = stage_config[stage_key][0]
            self.srtod_detail_residual = DeficiencyGuidedHaarDetailResidual(
                in_channels=stage_in_channels,
                out_channels=self._out_channels[srtod_stage],
                groups=int(srtod_detail_groups),
                init_scale=float(srtod_detail_init_scale),
            )
        self.srtod_learnable_thresh = (
            nn.Parameter(torch.tensor(float(srtod_threshold), dtype=torch.float32))
            if srtod_stage >= 0 and not self.srtod_student_only
            else None
        )
        if self.tndp_stage < -1 or self.tndp_stage >= len(stage_config):
            raise ValueError(
                f"tndp_stage must be -1 or a valid stage index, got {self.tndp_stage}"
            )
        if self.tndp_stage >= 0:
            tndp_key = list(stage_config.keys())[self.tndp_stage]
            if not bool(stage_config[tndp_key][4]):
                raise ValueError("TNDP must supervise a downsampling stage")
            self.tndp_head = TargetNeighborhoodDetailReconstruction(
                out_channels=stage_config[tndp_key][2],
                target_max=float(tndp_target_max),
            )
        else:
            self.tndp_head = None

        # stem
        self.stem = StemBlock(
            in_chs=stem_channels[0],
            mid_chs=stem_channels[1],
            out_chs=stem_channels[2],
            use_lab=use_lab,
        )

        # stages
        self.stages = nn.ModuleList()
        if self.pat_sf_stage < -1 or self.pat_sf_stage >= len(stage_config):
            raise ValueError(
                "pat_sf_stage must be -1 or a valid stage index, "
                f"got {self.pat_sf_stage}"
            )
        if self.pat_sf_stage >= 0 and self.pat_sf_stage != len(stage_config) - 1:
            raise ValueError(
                "The official PAT_sf design restricts self-attention to the "
                "last stage; set pat_sf_stage to the final stage index"
            )
        if self.pat_sf_variant not in {
            "self",
            "partial",
            "channel",
            "global_mean",
            "global_query",
        }:
            raise ValueError(
                "pat_sf_variant must be 'self', 'partial', 'channel', "
                "'global_mean', or 'global_query'; "
                f"got {self.pat_sf_variant!r}"
            )
        if self.pat_stage >= 0 and self.pat_sf_stage >= 0:
            raise ValueError(
                "PAT_ch and PAT_sf direct-transfer screens must be isolated"
            )
        if self.btrd_stage < -1 or self.btrd_stage >= len(stage_config):
            raise ValueError(
                f"btrd_stage must be -1 or a valid stage index, got {self.btrd_stage}"
            )
        if self.btrd_stage >= 0:
            btrd_key = list(stage_config.keys())[self.btrd_stage]
            if not bool(stage_config[btrd_key][4]):
                raise ValueError("BTRD must replace a stage that performs downsampling")
            conflicting = {
                "lad_stage": lad_stage,
                "preserve_stage": preserve_stage,
                "blur_stage": blur_stage,
            }
            conflicts = [name for name, stage in conflicting.items() if stage == self.btrd_stage]
            if conflicts:
                raise ValueError(
                    "BTRD1 is an isolated downsampling test and cannot share its stage with "
                    + ", ".join(conflicts)
                )
        for option_name, stage in {
            "fsd_stage": self.fsd_stage,
            "reinit_downsample_stage": self.reinit_downsample_stage,
        }.items():
            if stage < -1 or stage >= len(stage_config):
                raise ValueError(
                    f"{option_name} must be -1 or a valid stage index, got {stage}"
                )
            if stage >= 0:
                stage_key = list(stage_config.keys())[stage]
                if not bool(stage_config[stage_key][4]):
                    raise ValueError(f"{option_name} must replace a downsampling stage")
        active_replacements = {
            "btrd_stage": self.btrd_stage,
            "fsd_stage": self.fsd_stage,
            "reinit_downsample_stage": self.reinit_downsample_stage,
        }
        active_replacements = {
            name: stage for name, stage in active_replacements.items() if stage >= 0
        }
        if len(active_replacements) > 1:
            raise ValueError(
                "isolated downsampling replacements cannot be enabled together: "
                + ", ".join(f"{name}={stage}" for name, stage in active_replacements.items())
            )
        isolated_stage = self.fsd_stage if self.fsd_stage >= 0 else self.reinit_downsample_stage
        if isolated_stage >= 0:
            conflicting = {
                "lad_stage": lad_stage,
                "preserve_stage": preserve_stage,
                "blur_stage": blur_stage,
                "pdbr_stage": self.pdbr_stage,
            }
            conflicts = [name for name, stage in conflicting.items() if stage == isolated_stage]
            if conflicts:
                raise ValueError(
                    "FSD/control must remain an isolated downsampling test and cannot share "
                    "its stage with " + ", ".join(conflicts)
                )
        if self.pdbr_stage < -1 or self.pdbr_stage >= len(stage_config):
            raise ValueError(
                f"pdbr_stage must be -1 or a valid stage index, got {self.pdbr_stage}"
            )
        if self.pdbr_stage >= 0:
            pdbr_key = list(stage_config.keys())[self.pdbr_stage]
            if not bool(stage_config[pdbr_key][4]):
                raise ValueError("PDBR must be attached to a downsampling stage")
            conflicts = []
            for conflict_name, stage in {
                "btrd_stage": self.btrd_stage,
                "lad_stage": lad_stage,
                "preserve_stage": preserve_stage,
                "blur_stage": blur_stage,
            }.items():
                if stage == self.pdbr_stage:
                    conflicts.append(conflict_name)
            if conflicts:
                raise ValueError(
                    "PDBR is an isolated residual test and cannot share its stage with "
                    + ", ".join(conflicts)
                )
        for i, k in enumerate(stage_config):
            (
                in_channels,
                mid_channels,
                out_channels,
                block_num,
                downsample,
                light_block,
                kernel_size,
                layer_num,
            ) = stage_config[k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                    preserve_mode=(preserve_mode if i == preserve_stage else "none"),
                    preserve_ratio=preserve_ratio,
                    preserve_groups=preserve_groups,
                    preserve_init_scale=preserve_init_scale,
                    downsample_mode=(
                        "fsd"
                        if i == self.fsd_stage
                        else (
                            "standard_reinit"
                            if i == self.reinit_downsample_stage
                            else (
                                "btrd"
                                if i == self.btrd_stage
                                else (
                                    "conditional_lad"
                                    if i == lad_stage and lad_fusion_mode == "stable_route"
                                    else "lad" if i == lad_stage else "standard"
                                )
                            )
                        )
                    ),
                    lad_groups=lad_groups,
                    lad_target_gate=(lad_target_gate and i == lad_stage),
                    lad_candidate_mode=lad_candidate_mode,
                    lad_fusion_mode=lad_fusion_mode,
                    lad_route_init=lad_route_init,
                    lad_target_gate_mode=lad_target_gate_mode,
                    preblur=(i == blur_stage),
                    use_pat_sf=(i == self.pat_sf_stage),
                    pat_sf_n_div=pat_sf_n_div,
                    pat_sf_variant=self.pat_sf_variant,
                    pdbr_enabled=(i == self.pdbr_stage),
                    pdbr_detail_channels=pdbr_detail_channels,
                    pdbr_max_scale=pdbr_max_scale,
                    pdbr_init_scale=pdbr_init_scale,
                )
            )

        if self.spar_enabled:
            if len(self._out_channels) < 4:
                raise ValueError("SPAR requires S8/S16/S32 backbone stages")
            # Adding a training-only branch must not consume the CPU RNG that
            # initializes the unchanged encoder and decoder later.  This keeps
            # all shared parameters identical to A00 under the same seed.
            cpu_rng_state = torch.random.get_rng_state()
            try:
                self.spar_fusion = SPARFuseFeatures(
                    deep_channels=self._out_channels[3],
                    middle_channels=self._out_channels[2],
                    shallow_channels=self._out_channels[1],
                    channels=int(spar_fusion_channels),
                )
            finally:
                torch.random.set_rng_state(cpu_rng_state)
        else:
            self.spar_fusion = None

        if self.sabr_enabled:
            if len(self._out_channels) < 3:
                raise ValueError("SABR requires real S8 and S16 backbone stages")
            # As with SPAR, the training-only heads must not perturb the RNG
            # stream used by the unchanged encoder and decoder initialization.
            cpu_rng_state = torch.random.get_rng_state()
            try:
                self.sabr_heads = SABRScaleHeads(
                    s8_channels=self._out_channels[1],
                    s16_channels=self._out_channels[2],
                    hidden_channels=sabr_hidden_channels,
                )
            finally:
                torch.random.set_rng_state(cpu_rng_state)
        else:
            self.sabr_heads = None

        if self.sbox_enabled:
            if len(self._out_channels) < 2:
                raise ValueError("S-BOX requires the real S8 backbone stage")
            # Preserve the initialization stream of every shared A00 module.
            cpu_rng_state = torch.random.get_rng_state()
            try:
                self.sbox_head = SBOXExtremePointHead(
                    in_channels=self._out_channels[1],
                    hidden_channels=sbox_hidden_channels,
                )
            finally:
                torch.random.set_rng_state(cpu_rng_state)
        else:
            self.sbox_head = None

        if self.pat_stage < -1 or self.pat_stage >= len(self.stages):
            raise ValueError(
                f"pat_stage must be -1 or a valid stage index, got {self.pat_stage}"
            )
        self.pat_ch = (
            PartialChannelAttentionConv(
                self._out_channels[self.pat_stage],
                n_div=pat_n_div,
            )
            if self.pat_stage >= 0
            else None
        )

        if self.rpc_stage < -1 or self.rpc_stage >= len(self.stages):
            raise ValueError(
                f"rpc_stage must be -1 or a valid stage index, got {self.rpc_stage}"
            )
        if self.rpc_stage >= 0 and not bool(stage_config[f"stage{self.rpc_stage + 1}"][4]):
            raise ValueError("RPC branch must be attached to a downsampling stage")
        if self.rpc_stage >= 0:
            rpc_cfg = stage_config[f"stage{self.rpc_stage + 1}"]
            self.rpc_branch = LowRankComplementaryDownsample(
                in_chs=rpc_cfg[0],
                out_chs=rpc_cfg[2],
                hidden_chs=rpc_hidden_channels,
                init_scale=rpc_init_scale,
                use_lab=use_lab,
            )
        else:
            self.rpc_branch = None

        if self.bpc_stage < -1 or self.bpc_stage >= len(self.stages):
            raise ValueError(
                f"bpc_stage must be -1 or a valid stage index, got {self.bpc_stage}"
            )
        if self.bpc_stage >= 0:
            bpc_key = list(stage_config.keys())[self.bpc_stage]
            bpc_cfg = stage_config[bpc_key]
            if not bool(bpc_cfg[4]):
                raise ValueError("BPC branch must be attached to a downsampling stage")
            conflicts = {
                "btrd_stage": self.btrd_stage,
                "fsd_stage": self.fsd_stage,
                "reinit_downsample_stage": self.reinit_downsample_stage,
                "lad_stage": lad_stage,
                "preserve_stage": preserve_stage,
                "blur_stage": blur_stage,
                "pdbr_stage": self.pdbr_stage,
                "tndp_stage": self.tndp_stage,
                "rpc_stage": self.rpc_stage,
            }
            conflicts = [name for name, stage in conflicts.items() if stage == self.bpc_stage]
            if conflicts:
                raise ValueError(
                    "BPC1 is an isolated carrier test and cannot share its stage with "
                    + ", ".join(conflicts)
                )
            self.bpc_branch = BoundaryPolyphaseCarrier(
                in_channels=bpc_cfg[0],
                out_channels=bpc_cfg[2],
                detail_channels=int(bpc_detail_channels),
                project_groups=int(bpc_project_groups),
                max_scale=float(bpc_max_scale),
                init_scale=float(bpc_init_scale),
            )
        else:
            self.bpc_branch = None

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
            try:
                # If the file doesn't exist locally, download from the URL
                if safe_get_rank() == 0:
                    print(
                        GREEN
                        + "If the pretrained HGNetV2 can't be downloaded automatically. Please check your network connection."
                        + RESET
                    )
                    print(
                        GREEN
                        + "Please check your network connection. Or download the model manually from "
                        + RESET
                        + f"{download_url}"
                        + GREEN
                        + " to "
                        + RESET
                        + f"{local_model_dir}."
                        + RESET
                    )
                    state = torch.hub.load_state_dict_from_url(
                        download_url, map_location="cpu", model_dir=local_model_dir
                    )
                    print(f"Loaded stage1 {name} HGNetV2 from URL.")

                # Wait for rank 0 to download the model
                safe_barrier()

                # All processes load the downloaded model
                model_path = local_model_dir + "PPHGNetV2_" + name + "_stage1.pth"
                state = torch.load(model_path, map_location="cpu")

                if (
                    self.pat_stage >= 0
                    or self.pat_sf_stage >= 0
                    or self.btrd_stage >= 0
                    or self.fsd_stage >= 0
                    or self.reinit_downsample_stage >= 0
                    or self.pdbr_stage >= 0
                    or self.tndp_stage >= 0
                    or self.bpc_stage >= 0
                    or self.spar_enabled
                    or self.sabr_enabled
                ):
                    incompatible = self.load_state_dict(state, strict=False)
                    replacement_stage = (
                        self.fsd_stage
                        if self.fsd_stage >= 0
                        else self.reinit_downsample_stage
                    )
                    bad_missing = []
                    for key in incompatible.missing_keys:
                        allowed_pat_ch = self.pat_stage >= 0 and key.startswith("pat_ch.")
                        allowed_pat_sf = (
                            self.pat_sf_stage >= 0 and ".conv2.pat_sf." in key
                        )
                        allowed_btrd = (
                            self.btrd_stage >= 0
                            and key.startswith(f"stages.{self.btrd_stage}.downsample.")
                        )
                        allowed_fsd_or_control = (
                            replacement_stage >= 0
                            and key.startswith(f"stages.{replacement_stage}.downsample.")
                        )
                        allowed_pdbr = (
                            self.pdbr_stage >= 0
                            and key.startswith(f"stages.{self.pdbr_stage}.pdbr.")
                        )
                        allowed_tndp = self.tndp_stage >= 0 and key.startswith("tndp_head.")
                        allowed_bpc = self.bpc_stage >= 0 and key.startswith("bpc_branch.")
                        allowed_spar = self.spar_enabled and key.startswith("spar_fusion.")
                        allowed_sabr = self.sabr_enabled and key.startswith("sabr_heads.")
                        if not (
                            allowed_pat_ch
                            or allowed_pat_sf
                            or allowed_btrd
                            or allowed_fsd_or_control
                            or allowed_pdbr
                            or allowed_tndp
                            or allowed_bpc
                            or allowed_spar
                            or allowed_sabr
                        ):
                            bad_missing.append(key)

                    bad_unexpected = []
                    for key in incompatible.unexpected_keys:
                        replaced_depthwise = (
                            self.pat_sf_stage >= 0
                            and key.startswith(f"stages.{self.pat_sf_stage}.")
                            and ".conv2.conv.weight" in key
                        )
                        replaced_btrd_downsample = (
                            self.btrd_stage >= 0
                            and key.startswith(f"stages.{self.btrd_stage}.downsample.")
                        )
                        replaced_fsd_or_control_downsample = (
                            replacement_stage >= 0
                            and key.startswith(f"stages.{replacement_stage}.downsample.")
                        )
                        if not (
                            replaced_depthwise
                            or replaced_btrd_downsample
                            or replaced_fsd_or_control_downsample
                        ):
                            bad_unexpected.append(key)

                    if bad_missing or bad_unexpected:
                        raise RuntimeError(
                            "Unexpected HGNetv2 pretrained-key mismatch after PAT transfer: "
                            f"missing={bad_missing}, unexpected={bad_unexpected}"
                        )
                    transferred = []
                    if self.pat_stage >= 0 or self.pat_sf_stage >= 0:
                        transferred.append("PAT")
                    if self.btrd_stage >= 0:
                        transferred.append("BTRD")
                    if self.fsd_stage >= 0:
                        transferred.append("FSD")
                    if self.reinit_downsample_stage >= 0:
                        transferred.append("STD-REINIT")
                    if self.pdbr_stage >= 0:
                        transferred.append("PDBR")
                    if self.tndp_stage >= 0:
                        transferred.append("TNDP")
                    if self.bpc_stage >= 0:
                        transferred.append("BPC")
                    if self.spar_enabled:
                        transferred.append("SPAR")
                    if self.sabr_enabled:
                        transferred.append("SABR")
                    print(
                        "Loaded HGNetv2 base weights with isolated "
                        + "/".join(transferred)
                        + " parameters left at their source initialization."
                    )
                else:
                    self.load_state_dict(state)
                print(f"Loaded stage1 {name} HGNetV2 from URL.")

            except (Exception, KeyboardInterrupt) as e:
                if safe_get_rank() == 0:
                    print(f"{str(e)}")
                    logging.error(
                        RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET
                    )
                    logging.error(
                        GREEN
                        + "Please check your network connection. Or download the model manually from "
                        + RESET
                        + f"{download_url}"
                        + GREEN
                        + " to "
                        + RESET
                        + f"{local_model_dir}."
                        + RESET
                    )
                exit()

    def set_training_epoch(self, epoch):
        self.training_epoch = int(epoch)

    def _spar_backbone_gradient_scale(self):
        if self.spar_gradient_mode == "full":
            return self.spar_gradient_max_scale
        if self.training_epoch < self.spar_gradient_ramp_start:
            return 0.0
        if self.training_epoch >= self.spar_gradient_ramp_end:
            return self.spar_gradient_max_scale
        return self.spar_gradient_max_scale * (
            float(self.training_epoch - self.spar_gradient_ramp_start)
            / float(self.spar_gradient_ramp_end - self.spar_gradient_ramp_start)
        )

    def _sbox_backbone_gradient_scale(self):
        if self.training_epoch < self.sbox_gradient_ramp_start:
            return 0.0
        if self.training_epoch >= self.sbox_gradient_ramp_end:
            return self.sbox_gradient_max_scale
        return self.sbox_gradient_max_scale * (
            float(self.training_epoch - self.sbox_gradient_ramp_start)
            / float(self.sbox_gradient_ramp_end - self.sbox_gradient_ramp_start)
        )

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        self.spatial_importance_logits = None
        self.sqer_s4_source = None
        self.sgc_s4_source = None
        self.pdbr_boundary_logits = None
        self.tndp_detail_prediction = None
        self.tndp_detail_target = None
        self.bpc_boundary_logits = None
        self.qrl_source = None
        self.mdqa_source = None
        self.sqmi_source = None
        self.spar_fused_features = None
        self.sabr_outputs = None
        self.sbox_extreme_logits = None
        self.sbrd_stage_features = None
        self.qcsr_stage_features = None
        source_image = x
        self.srtod_reconstruction_loss = None
        self.srtod_difference_map = None
        self.srtod_difference_mask = None
        x = self.stem(x)
        outs = []
        spar_stage_features = []
        sabr_stage_features = []
        sbrd_stage_features = []
        qcsr_stage_features = []
        for idx, stage in enumerate(self.stages):
            stage_input = x
            if idx == self.qrl_source_stage:
                # QRL consumes this tensor through a detached side path.  The
                # standard stage and returned detection features are untouched.
                self.qrl_source = stage_input
            tndp_target = None
            bpc_active = idx == self.bpc_stage and self.bpc_branch is not None
            if self.training and idx == self.tndp_stage and self.tndp_head is not None:
                tndp_target = self.tndp_head.make_target(stage_input)
            detail_residual = None
            if idx == self.srtod_stage and self.srtod_detail_residual is not None:
                # Extract detail before the standard stride-2 operation.  It
                # is applied only after the complete standard stage output is
                # available, leaving that semantic path unchanged.
                detail_residual = self.srtod_detail_residual.extract(stage_input)
            x = stage(x)
            if idx == 0 and self.sqer_capture_s4:
                self.sqer_s4_source = x
            if idx == self.sqmi_source_stage:
                # Unlike training-only MDQA, S-QMI1 predicts the mask-derived
                # reference at inference too, so its RGB S8 source is retained
                # in both modes.
                self.sqmi_source = x
            if self.training and idx == self.mdqa_source_stage:
                # REP2-MDQA reads the real post-stage S8 feature through a
                # training-only side path. The ordinary detection feature
                # list and every downsampling operation remain unchanged.
                self.mdqa_source = x
            if tndp_target is not None:
                self.tndp_detail_target = tndp_target
                self.tndp_detail_prediction = self.tndp_head.predict(x)
            if bpc_active:
                x = self.bpc_branch(stage_input, x)
                self.bpc_boundary_logits = self.bpc_branch.last_boundary_logits
            if idx == self.rpc_stage and self.rpc_branch is not None:
                x = x + self.rpc_branch(stage_input)
            if idx == self.pat_stage and self.pat_ch is not None:
                x = self.pat_ch(x)
            if idx == self.srtod_stage:
                if self.srtod_student_only:
                    deficiency_probability = self.srtod_deficiency_head(x).sigmoid()
                    if self.srtod_enhancer == "dgfe":
                        x = self.srtod_dgfe.forward_with_mask(x, deficiency_probability)
                    else:
                        x = x + self.srtod_detail_residual.apply_gate(
                            detail_residual, deficiency_probability
                        )
                    self.srtod_difference_map = deficiency_probability.detach()
                    self.srtod_difference_mask = deficiency_probability.detach()
                else:
                    reconstruction = self.srtod_rh(x.clone())
                    if reconstruction.shape[-2:] != source_image.shape[-2:]:
                        # D-FINE-N's first detection feature is S16, whereas the
                        # official SR-TOD input is P2/S4. This is the only spatial
                        # interface adaptation; RH and DGFE remain unchanged.
                        reconstruction = F.interpolate(
                            reconstruction,
                            size=source_image.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    difference_map = torch.sum(
                        torch.abs(reconstruction - source_image), dim=1, keepdim=True
                    ) / source_image.shape[1]
                    difference_mask = (
                        (torch.sign(difference_map - self.srtod_learnable_thresh) + 1)
                        * 0.5
                    )
                    if self.srtod_enhancer == "dgfe":
                        x = self.srtod_dgfe(
                            x,
                            difference_map,
                            self.srtod_learnable_thresh,
                        )
                    else:
                        x = x + self.srtod_detail_residual.apply_gate(
                            detail_residual, difference_mask
                        )
                    self.srtod_reconstruction_loss = F.mse_loss(reconstruction, source_image)
                    self.srtod_difference_map = difference_map.detach()
                    self.srtod_difference_mask = difference_mask.detach()
            if idx == self.spatial_aux_stage and self.spatial_aux_head is not None:
                self.spatial_importance_logits = self.spatial_aux_head(x)
            if idx == self.lad_stage:
                lad_module = (
                    stage.conditional_detail
                    if getattr(stage, "conditional_detail", None) is not None
                    else stage.downsample
                )
                target_logits = getattr(lad_module, "last_target_logits", None)
                if target_logits is not None:
                    self.spatial_importance_logits = target_logits
            if idx == self.pdbr_stage and getattr(stage, "pdbr", None) is not None:
                self.pdbr_boundary_logits = stage.pdbr.last_boundary_logits
            if self.training and idx == 1 and self.sbox_head is not None:
                gradient_scale = self._sbox_backbone_gradient_scale()
                sbox_feature = x.detach() + gradient_scale * (x - x.detach())
                self.sbox_extreme_logits = self.sbox_head(sbox_feature)
            if idx in self.return_idx:
                outs.append(x)
            if self.training and idx == 0 and getattr(self, 'sgc_capture_s4', False):
                self.sgc_s4_source = x
            if self.training and self.sbrd_enabled and self._out_strides[idx] in (8, 16):
                # D-FINE-N returns only S16/S32 to its detector. SBRD needs the
                # internal S8 tensor as the high-resolution teacher, so capture
                # stages by their real strides rather than indexing `outs`.
                sbrd_stage_features.append(x)
            if self.training and self.qcsr_enabled and self._out_strides[idx] in (8, 16):
                # REP3 needs the actual internal S8 tensor before the standard
                # stride-2 stage and its real S16 output.  These are read-only
                # taps; the ordinary detector feature list is unchanged.
                qcsr_stage_features.append(x)
            if self.training and self.spar_fusion is not None:
                spar_stage_features.append(x)
            if self.training and self.sabr_heads is not None and self._out_strides[idx] in (8, 16):
                sabr_stage_features.append(x)
        if self.training and self.spar_fusion is not None:
            if len(spar_stage_features) != len(self.stages):
                raise RuntimeError("SPAR did not capture every HGNetv2 stage")
            gradient_scale = self._spar_backbone_gradient_scale()
            spar_stage_features = [
                feature.detach()
                + gradient_scale * (feature - feature.detach())
                for feature in spar_stage_features
            ]
            self.spar_fused_features = self.spar_fusion(
                spar_stage_features[3],
                spar_stage_features[2],
                spar_stage_features[1],
            )
        if self.training and self.sabr_heads is not None:
            if len(sabr_stage_features) != 2:
                raise RuntimeError("SABR failed to capture real S8 and S16 features")
            gradient_scale = self._spar_backbone_gradient_scale()
            sabr_stage_features = [
                feature.detach()
                + gradient_scale * (feature - feature.detach())
                for feature in sabr_stage_features
            ]
            self.sabr_outputs = self.sabr_heads(
                sabr_stage_features[0], sabr_stage_features[1]
            )
        if self.training and self.sbrd_enabled:
            if len(sbrd_stage_features) != 2:
                raise RuntimeError("SBRD failed to capture internal S8 and S16 features")
            # Training-only taps: the ordinary detector path is not modified.
            self.sbrd_stage_features = tuple(sbrd_stage_features)
        if self.training and self.qcsr_enabled:
            if len(qcsr_stage_features) != 2:
                raise RuntimeError("QCSR failed to capture internal S8 and S16 features")
            self.qcsr_stage_features = tuple(qcsr_stage_features)
        return outs
