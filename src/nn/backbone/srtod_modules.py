"""Paper-faithful SR-TOD reconstruction and DGFE modules.

Source:
    https://github.com/Hiyuur/SR-TOD
    commit 1c4d93b104450055ecbfed178e6ffbbf1aedc9f6
    srtod_project/srtod_detectors/srtod_cascadercnn.py

The module internals below intentionally retain the official implementation.
Only their placement and tensor-size adaptation live in HGNetv2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => ReLU) * 2, as released by SR-TOD."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up_direct(nn.Module):
    """Upscaling then double conv, as released by SR-TOD."""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1):
        x = self.up(x1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.conv(x)


class RH(nn.Module):
    """SR-TOD image reconstruction head."""

    def __init__(self, in_channels=256, out_channels=3):
        super().__init__()
        self.up1 = Up_direct(in_channels, 128)
        self.up2 = Up_direct(128, 64)
        self.out_conv = OutConv(64, out_channels)

    def forward(self, x):
        p0 = self.up1(x)
        p0 = self.up2(p0)
        r_img = self.out_conv(p0)
        return r_img


class DeficiencyPredictor(nn.Module):
    """Low-cost student for an S16 SR-TOD difference mask.

    The predictor is deliberately separate from RH: RH supplies a frozen
    training target, while this head is the only component retained when the
    reconstruction teacher is removed at inference time.
    """

    def __init__(self, in_channels=512, hidden_channels=32):
        super().__init__()
        if hidden_channels % 8 != 0:
            raise ValueError("hidden_channels must be divisible by 8 for GroupNorm")
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DGFE(nn.Module):
    """Difference-map-guided feature enhancement released by SR-TOD."""

    def __init__(self, gate_channels=256, reduction_ratio=16, pool_types=["avg", "max"]):
        super().__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types
        # Evaluation-only causal intervention. ``learned`` is exactly the
        # released SR-TOD computation and is the only mode used in training.
        self.mask_mode = "learned"
        self.last_applied_mask = None
        self.mask_override = None

    def forward(self, x, difference_map, learnable_thresh):
        difference_map_mask = (torch.sign(difference_map - learnable_thresh) + 1) * 0.5
        if self.mask_mode == "external":
            if self.mask_override is None:
                raise RuntimeError("external SR-TOD mask mode requires mask_override")
            difference_map_mask = self.mask_override
        elif self.mask_mode == "one":
            difference_map_mask = torch.ones_like(difference_map_mask)
        elif self.mask_mode == "zero":
            difference_map_mask = torch.zeros_like(difference_map_mask)
        elif self.mask_mode == "shuffled":
            # Deterministic spatial permutation per image. The mask histogram
            # is preserved while its correspondence to the image is broken.
            flat = difference_map_mask.flatten(2)
            generator = torch.Generator(device="cpu").manual_seed(20260809)
            permutation = torch.randperm(flat.shape[-1], generator=generator).to(flat.device)
            difference_map_mask = flat[:, :, permutation].reshape_as(difference_map_mask)
        elif self.mask_mode != "learned":
            raise ValueError(f"unsupported SR-TOD mask mode: {self.mask_mode}")

        return self.forward_with_mask(x, difference_map_mask)

    def forward_with_mask(self, x, difference_map_mask):
        """Enhance ``x`` from an already resolved spatial mask.

        This is the deployment path used by the lightweight deficiency
        student.  It deliberately bypasses reconstruction, thresholding and
        all evaluation-only mask interventions while retaining the released
        DGFE enhancement computation exactly.
        """
        self.last_applied_mask = difference_map_mask.detach()
        feat_difference_map = F.interpolate(
            difference_map_mask,
            size=(x.shape[2], x.shape[3]),
        )

        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(
                    x,
                    (x.size(2), x.size(3)),
                    stride=(x.size(2), x.size(3)),
                )
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(
                    x,
                    (x.size(2), x.size(3)),
                    stride=(x.size(2), x.size(3)),
                )
                channel_att_raw = self.mlp(max_pool)
            else:
                raise ValueError(f"unsupported SR-TOD pool type: {pool_type}")

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        feat_diff_mat = feat_difference_map.repeat(1, x.shape[1], 1, 1)
        x_out = x * scale
        x_out = torch.mul(x_out, feat_diff_mat) + x_out
        return x_out
