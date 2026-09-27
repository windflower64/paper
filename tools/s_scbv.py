#!/usr/bin/env python3
"""S14-SCBV：语义条件的一维框边界代价体。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


SCBV_MODES = (
    "aligned",
    "low_only",
    "shift_detail",
    "swap_direction",
    "s16_only",
    "zero_update",
)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height),
        dim=-1,
    )


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        ((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1
    )


class SemanticConditionedBoundaryVolume(nn.Module):
    """利用目标语义，在预测框四条边的法线方向比较多个S8候选位置。

    目标语义始终从未干预的S8池化。反事实模式只改变局部边界证据，因而不会
    同时破坏“正在寻找哪个目标”。最后一层零初始化使构造时的期望偏移严格为0。
    """

    def __init__(
        self,
        s8_channels: int,
        s16_channels: int,
        detail_channels: int = 16,
        tangent_points: int = 5,
        offset_bins: int = 9,
        max_relative_offset: float = 0.25,
        semantic_grid: int = 3,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if tangent_points < 1 or offset_bins < 3 or offset_bins % 2 == 0:
            raise ValueError("切向点数必须为正，偏移bin必须是大于等于3的奇数")
        if s16_channels % s8_channels != 0:
            raise ValueError(
                f"S16通道{s16_channels}必须能被S8通道{s8_channels}整除，"
                "以使用固定通道折叠对照"
            )
        self.s8_channels = int(s8_channels)
        self.s16_channels = int(s16_channels)
        self.detail_channels = int(detail_channels)
        self.tangent_points = int(tangent_points)
        self.offset_bins = int(offset_bins)
        self.max_relative_offset = float(max_relative_offset)
        self.semantic_grid = int(semantic_grid)
        self.s16_fold = s16_channels // s8_channels

        self.reduce = nn.Conv2d(s8_channels, detail_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.reduce.weight, a=1)

        # local、法线方向差分、目标语义、余弦相似度、框几何、side one-hot、offset
        candidate_dim = 3 * detail_channels + 1 + 4 + 4 + 1
        self.score_head = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        nn.init.zeros_(self.score_head[-1].weight)

        offsets = torch.linspace(-max_relative_offset, max_relative_offset, offset_bins)
        tangents = torch.linspace(-0.35, 0.35, tangent_points)
        self.register_buffer("offset_values", offsets, persistent=True)
        self.register_buffer("tangent_values", tangents, persistent=True)
        self.last_logits: torch.Tensor | None = None
        self.last_expected_offsets: torch.Tensor | None = None
        self.last_mode_statistics: dict[str, float] = {}

    @staticmethod
    def lowpass(feature: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(feature, kernel_size=2, stride=2)
        return F.interpolate(
            pooled, size=feature.shape[-2:], mode="bilinear", align_corners=False
        )

    @staticmethod
    def directional_maps(feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padded_x = F.pad(feature, (1, 1, 0, 0), mode="replicate")
        padded_y = F.pad(feature, (0, 0, 1, 1), mode="replicate")
        dx = 0.5 * (padded_x[..., 2:] - padded_x[..., :-2])
        dy = 0.5 * (padded_y[..., 2:, :] - padded_y[..., :-2, :])
        return dx, dy

    def fold_s16_to_s8_channels(self, s16: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = s16.shape
        if channels != self.s16_channels:
            raise ValueError(f"S16期望{self.s16_channels}通道，实际{channels}")
        return s16.reshape(
            batch, self.s8_channels, self.s16_fold, height, width
        ).mean(dim=2)

    def build_evidence(
        self, s8: torch.Tensor, s16: torch.Tensor, mode: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if mode not in SCBV_MODES:
            raise ValueError(f"未知SCBV模式：{mode}")
        aligned = self.reduce(s8)
        low = self.lowpass(aligned)
        detail = aligned - low
        if mode in ("aligned", "zero_update"):
            local = aligned
        elif mode == "low_only":
            local = low
        elif mode == "shift_detail":
            shifted = torch.roll(
                detail,
                shifts=(max(1, detail.shape[-2] // 3), max(1, detail.shape[-1] // 5)),
                dims=(-2, -1),
            )
            local = low + shifted
        elif mode == "swap_direction":
            local = aligned
        else:
            folded = self.fold_s16_to_s8_channels(s16)
            folded = F.interpolate(
                folded, size=s8.shape[-2:], mode="bilinear", align_corners=False
            )
            # 同一个reduce权重处理固定折叠后的真实S16，避免为对照另训适配器。
            local = self.reduce(folded)

        dx, dy = self.directional_maps(local)
        if mode == "swap_direction":
            dx, dy = dy, dx
        parts = {
            "aligned": aligned,
            "low": low,
            "detail": detail,
            "local": local,
            "dx": dx,
            "dy": dy,
        }
        return local, dx, dy, parts

    @staticmethod
    def sample_map(
        feature: torch.Tensor, points: torch.Tensor, batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """points: [N,...,2]，返回[N,...,C]。"""
        output = feature.new_zeros((*points.shape[:-1], feature.shape[1]))
        flat_shape = points.shape[1:-1]
        grid = points.mul(2.0).sub(1.0)
        for image_index in batch_indices.unique(sorted=True).tolist():
            mask = batch_indices == image_index
            image_grid = grid[mask].reshape(1, -1, 1, 2)
            values = F.grid_sample(
                feature[image_index : image_index + 1],
                image_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            values = values[0, :, :, 0].transpose(0, 1)
            output[mask] = values.reshape(int(mask.sum()), *flat_shape, feature.shape[1])
        return output

    def semantic_points(self, boxes: torch.Tensor) -> torch.Tensor:
        xyxy = box_cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)
        x1, y1, x2, y2 = xyxy.unbind(-1)
        fractions = torch.linspace(
            0.25, 0.75, self.semantic_grid, device=boxes.device, dtype=boxes.dtype
        )
        fy, fx = torch.meshgrid(fractions, fractions, indexing="ij")
        x = x1[:, None, None] + (x2 - x1)[:, None, None] * fx
        y = y1[:, None, None] + (y2 - y1)[:, None, None] * fy
        return torch.stack((x, y), dim=-1)

    def candidate_points(self, boxes: torch.Tensor) -> torch.Tensor:
        xyxy = box_cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)
        x1, y1, x2, y2 = xyxy.unbind(-1)
        width = (x2 - x1).clamp_min(1e-5)
        height = (y2 - y1).clamp_min(1e-5)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        offsets = self.offset_values.to(boxes)[None, :, None]
        tangents = self.tangent_values.to(boxes)[None, None, :]
        n = boxes.shape[0]
        k = self.offset_bins
        t = self.tangent_points

        vertical_y = cy[:, None, None] + height[:, None, None] * tangents
        vertical_y = vertical_y.expand(n, k, t)
        horizontal_x = cx[:, None, None] + width[:, None, None] * tangents
        horizontal_x = horizontal_x.expand(n, k, t)
        left_x = x1[:, None, None] + width[:, None, None] * offsets
        right_x = x2[:, None, None] + width[:, None, None] * offsets
        top_y = y1[:, None, None] + height[:, None, None] * offsets
        bottom_y = y2[:, None, None] + height[:, None, None] * offsets
        left = torch.stack((left_x.expand(n, k, t), vertical_y), dim=-1)
        top = torch.stack((horizontal_x, top_y.expand(n, k, t)), dim=-1)
        right = torch.stack((right_x.expand(n, k, t), vertical_y), dim=-1)
        bottom = torch.stack((horizontal_x, bottom_y.expand(n, k, t)), dim=-1)
        return torch.stack((left, top, right, bottom), dim=1).clamp(0.0, 1.0)

    def candidate_logits(
        self,
        s8: torch.Tensor,
        s16: torch.Tensor,
        boxes: torch.Tensor,
        batch_indices: torch.Tensor,
        mode: str = "aligned",
    ) -> torch.Tensor:
        local_map, dx, dy, parts = self.build_evidence(s8, s16, mode)
        # 语义始终来自未干预的S8表示。
        semantic = self.sample_map(
            parts["aligned"], self.semantic_points(boxes), batch_indices
        ).mean(dim=(1, 2))
        points = self.candidate_points(boxes)
        local = self.sample_map(local_map, points, batch_indices).mean(dim=3)
        dx_values = self.sample_map(dx, points, batch_indices).mean(dim=3)
        dy_values = self.sample_map(dy, points, batch_indices).mean(dim=3)
        directional = torch.stack(
            (dx_values[:, 0], dy_values[:, 1], dx_values[:, 2], dy_values[:, 3]),
            dim=1,
        )

        semantic_expanded = semantic[:, None, None, :].expand_as(local)
        similarity = F.cosine_similarity(local, semantic_expanded, dim=-1)[..., None]
        geometry = boxes[:, None, None, :].expand(
            -1, 4, self.offset_bins, -1
        )
        side = F.one_hot(
            torch.arange(4, device=boxes.device), num_classes=4
        ).to(boxes.dtype)
        side = side[None, :, None, :].expand(boxes.shape[0], -1, self.offset_bins, -1)
        offset = self.offset_values.to(boxes)[None, None, :, None].expand(
            boxes.shape[0], 4, -1, -1
        )
        candidate = torch.cat(
            (local, directional, semantic_expanded, similarity, geometry, side, offset),
            dim=-1,
        )
        logits = self.score_head(candidate).squeeze(-1)
        # 训练器同时使用框回归损失和离散偏移监督；保留当前图，下一批会覆盖。
        self.last_logits = logits
        self.last_mode_statistics = {
            "aligned_rms": float(parts["aligned"].detach().float().square().mean().sqrt()),
            "low_rms": float(parts["low"].detach().float().square().mean().sqrt()),
            "detail_l2": float(parts["detail"].detach().float().square().sum().sqrt()),
            "local_rms": float(parts["local"].detach().float().square().mean().sqrt()),
        }
        return logits

    def forward(
        self,
        s8: torch.Tensor,
        s16: torch.Tensor,
        boxes: torch.Tensor,
        batch_indices: torch.Tensor,
        mode: str = "aligned",
    ) -> torch.Tensor:
        if mode == "zero_update":
            self.last_logits = None
            self.last_expected_offsets = torch.zeros_like(boxes[:, :4])
            return boxes
        logits = self.candidate_logits(s8, s16, boxes, batch_indices, mode)
        probabilities = logits.softmax(dim=-1)
        expected = (probabilities * self.offset_values.to(logits)).sum(dim=-1)
        self.last_expected_offsets = expected.detach()

        xyxy = box_cxcywh_to_xyxy(boxes)
        x1, y1, x2, y2 = xyxy.unbind(-1)
        width = (x2 - x1).clamp_min(1e-5)
        height = (y2 - y1).clamp_min(1e-5)
        updated = torch.stack(
            (
                x1 + expected[:, 0] * width,
                y1 + expected[:, 1] * height,
                x2 + expected[:, 2] * width,
                y2 + expected[:, 3] * height,
            ),
            dim=-1,
        )
        # 每侧最大只移动框尺度的1/4，因此两边最坏情况下仍保留一半宽高。
        # 这里不做坐标裁剪：裁剪会在零偏移时悄悄改变A00位于图像边缘的原框，
        # 从而污染“读取器是否真正产生修正”的因果比较。
        return box_xyxy_to_cxcywh(updated)

    def target_offsets(
        self, predicted: torch.Tensor, truth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_xyxy = box_cxcywh_to_xyxy(predicted)
        truth_xyxy = box_cxcywh_to_xyxy(truth)
        width = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp_min(1e-5)
        height = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp_min(1e-5)
        relative = torch.stack(
            (
                (truth_xyxy[:, 0] - pred_xyxy[:, 0]) / width,
                (truth_xyxy[:, 1] - pred_xyxy[:, 1]) / height,
                (truth_xyxy[:, 2] - pred_xyxy[:, 2]) / width,
                (truth_xyxy[:, 3] - pred_xyxy[:, 3]) / height,
            ),
            dim=-1,
        ).clamp(-self.max_relative_offset, self.max_relative_offset)
        distances = (relative[..., None] - self.offset_values.to(relative)).abs()
        bins = distances.argmin(dim=-1)
        return relative, bins

    def bin_loss_and_accuracy(
        self, logits: torch.Tensor, predicted: torch.Tensor, truth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, bins = self.target_offsets(predicted, truth)
        loss = F.cross_entropy(logits.reshape(-1, self.offset_bins), bins.reshape(-1))
        accuracy = (logits.argmax(dim=-1) == bins).float().mean()
        return loss, accuracy

    def detail_energy_control_error(self, s8: torch.Tensor) -> float:
        with torch.no_grad():
            aligned = self.reduce(s8)
            detail = aligned - self.lowpass(aligned)
            shifted = torch.roll(
                detail,
                shifts=(max(1, detail.shape[-2] // 3), max(1, detail.shape[-1] // 5)),
                dims=(-2, -1),
            )
            left = detail.float().square().sum().sqrt()
            right = shifted.float().square().sum().sqrt()
            return float((left - right).abs() / left.clamp_min(1e-12))

    def extra_repr(self) -> str:
        return (
            f"s8_channels={self.s8_channels}, s16_channels={self.s16_channels}, "
            f"detail_channels={self.detail_channels}, tangent_points={self.tangent_points}, "
            f"offset_bins={self.offset_bins}, max_relative_offset={self.max_relative_offset}"
        )


def gradient_l2(parameters) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    value = torch.stack(values).sum().sqrt()
    return float(value) if math.isfinite(float(value)) else float("nan")
