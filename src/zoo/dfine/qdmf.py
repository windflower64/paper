"""Query-conditioned Dynamic Modality Fusion (QDMF v1).

The implementation in this file intentionally follows the frozen design in
``20260922_S_QDMF_implementation_handoff_ZH.md``.  It only updates the final
ordinary decoder queries; matching and auxiliary decoder outputs are handled
by the caller.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryConditionedDynamicModalityFusion(nn.Module):
    """Read local multi-scale IR evidence and add a gated query residual."""

    def __init__(
        self,
        query_dim: int,
        thermal_channels: Sequence[int],
        roi_expand: float = 1.5,
        roi_grid: int = 3,
        num_heads: int = 4,
        gate_dim: int = 64,
        gate_hidden: int = 128,
        gate_init_bias: float = -2.0,
        output_init_std: float = 1e-3,
        ir_dropout: float = 0.10,
        detach_ir_features: bool = True,
        detach_base_boxes: bool = True,
        detach_rgb_uncertainty: bool = True,
        residual_warmup_start: float = 0.0,
        residual_warmup_end: float = 6.0,
        residual_init_scale: float = 0.10,
        residual_final_scale: float = 1.0,
        log_diagnostics: bool = True,
    ) -> None:
        super().__init__()
        if len(thermal_channels) != 3:
            raise ValueError("QDMF v1 requires exactly S8/S16/S32 thermal channels")
        if roi_grid <= 0:
            raise ValueError("qdmf_roi_grid must be positive")
        if not 0.0 <= ir_dropout < 1.0:
            raise ValueError("qdmf_ir_dropout must be in [0, 1)")
        if residual_warmup_end <= residual_warmup_start:
            raise ValueError("QDMF warmup end must be greater than start")

        self.query_dim = int(query_dim)
        self.roi_expand = float(roi_expand)
        self.roi_grid = int(roi_grid)
        self.ir_dropout = float(ir_dropout)
        self.detach_ir_features = bool(detach_ir_features)
        self.detach_base_boxes = bool(detach_base_boxes)
        self.detach_rgb_uncertainty = bool(detach_rgb_uncertainty)
        self.residual_warmup_start = float(residual_warmup_start)
        self.residual_warmup_end = float(residual_warmup_end)
        self.residual_init_scale = float(residual_init_scale)
        self.residual_final_scale = float(residual_final_scale)
        self.log_diagnostics = bool(log_diagnostics)
        self.register_buffer("training_progress", torch.tensor(0.0))

        requested_heads = max(1, int(num_heads))
        valid_heads = [h for h in range(1, requested_heads + 1) if self.query_dim % h == 0]
        self.num_heads = valid_heads[-1]
        self.feature_projections = nn.ModuleList(
            [nn.Conv2d(int(ch), self.query_dim, kernel_size=1) for ch in thermal_channels]
        )
        self.query_norm = nn.LayerNorm(self.query_dim)
        self.local_attention = nn.MultiheadAttention(
            self.query_dim, self.num_heads, batch_first=True
        )
        self.scale_mlp = nn.Sequential(
            nn.Linear(self.query_dim + 1, max(1, self.query_dim // 2)),
            nn.GELU(),
            nn.Linear(max(1, self.query_dim // 2), 3),
        )
        self.gate_query_projection = nn.Linear(self.query_dim, int(gate_dim))
        self.gate_evidence_norm = nn.LayerNorm(self.query_dim)
        self.gate_evidence_projection = nn.Linear(self.query_dim, int(gate_dim))
        self.gate_mlp = nn.Sequential(
            nn.Linear(4 * int(gate_dim) + 3, int(gate_hidden)),
            nn.GELU(),
            nn.Linear(int(gate_hidden), 1),
        )
        self.output_projection = nn.Linear(self.query_dim, self.query_dim)

        nn.init.normal_(self.gate_mlp[-1].weight, std=1e-3)
        nn.init.constant_(self.gate_mlp[-1].bias, float(gate_init_bias))
        nn.init.normal_(self.output_projection.weight, std=float(output_init_std))
        nn.init.zeros_(self.output_projection.bias)

    def set_training_progress(self, progress: float) -> None:
        self.training_progress.fill_(float(progress))

    def residual_scale(self) -> float:
        progress = float(self.training_progress.item())
        if progress < 2.0:
            return self.residual_init_scale
        if progress >= self.residual_warmup_end:
            return self.residual_final_scale
        ramp_start = max(2.0, self.residual_warmup_start)
        ratio = (progress - ramp_start) / max(self.residual_warmup_end - ramp_start, 1e-12)
        ratio = min(max(ratio, 0.0), 1.0)
        return self.residual_init_scale + ratio * (
            self.residual_final_scale - self.residual_init_scale
        )

    @staticmethod
    def _availability_mask(
        availability: torch.Tensor | None, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if availability is None:
            return torch.ones(batch_size, device=device, dtype=torch.bool)
        value = torch.as_tensor(availability, device=device, dtype=torch.bool)
        if value.numel() == 1:
            value = value.expand(batch_size)
        value = value.reshape(batch_size, -1).all(dim=1)
        return value

    def _sample_local_tokens(
        self, feature: torch.Tensor, boxes: torch.Tensor
    ) -> torch.Tensor:
        """Return [B,N,G*G,D] projected tokens for normalized cxcywh boxes."""
        batch, _, height, width = feature.shape
        if boxes.shape[0] != batch or boxes.shape[-1] != 4:
            raise ValueError("QDMF boxes must have shape [B,N,4]")
        boxes = boxes.float()
        centre = boxes[..., :2].clamp(0.0, 1.0)
        minimum = boxes.new_tensor((1.0 / max(width, 1), 1.0 / max(height, 1)))
        extent = (boxes[..., 2:].abs() * self.roi_expand).clamp_min(minimum)
        half = 0.5 * extent
        lower = (centre - half).clamp(0.0, 1.0)
        upper = (centre + half).clamp(0.0, 1.0)

        axis = torch.linspace(0.0, 1.0, self.roi_grid, device=feature.device, dtype=boxes.dtype)
        x = lower[..., 0, None] + (upper[..., 0] - lower[..., 0])[..., None] * axis
        y = lower[..., 1, None] + (upper[..., 1] - lower[..., 1])[..., None] * axis
        grid_x = x.unsqueeze(-2).expand(-1, -1, self.roi_grid, -1)
        grid_y = y.unsqueeze(-1).expand(-1, -1, -1, self.roi_grid)
        grid = torch.stack((grid_x, grid_y), dim=-1).mul(2.0).sub(1.0)
        grid = grid.reshape(batch, boxes.shape[1] * self.roi_grid, self.roi_grid, 2)
        sampled = F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        sampled = sampled.reshape(
            batch, self.query_dim, boxes.shape[1], self.roi_grid * self.roi_grid
        )
        return sampled.permute(0, 2, 3, 1).contiguous()

    def forward(
        self,
        queries: torch.Tensor,
        boxes_base: torch.Tensor,
        logits_base: torch.Tensor,
        thermal_features: Sequence[torch.Tensor],
        availability: torch.Tensor | None = None,
        bypass: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if len(thermal_features) != 3:
            raise ValueError("QDMF expects S8/S16/S32 thermal features")
        batch, num_queries, query_dim = queries.shape
        if query_dim != self.query_dim:
            raise ValueError("QDMF query dimension mismatch")

        available = self._availability_mask(availability, batch, queries.device)
        dropout_mask = torch.zeros_like(available)
        if self.training and self.ir_dropout > 0.0:
            dropout_mask = torch.rand(batch, device=queries.device) < self.ir_dropout
            available = available & ~dropout_mask

        boxes_for_roi = boxes_base.detach() if self.detach_base_boxes else boxes_base
        projected_features = []
        for projection, feature in zip(self.feature_projections, thermal_features):
            if self.detach_ir_features:
                feature = feature.detach()
            projected_features.append(projection(feature))

        normalized_queries = self.query_norm(queries)
        evidence_by_scale = []
        flat_query = normalized_queries.reshape(batch * num_queries, 1, query_dim)
        for feature in projected_features:
            tokens = self._sample_local_tokens(feature, boxes_for_roi)
            flat_tokens = tokens.reshape(batch * num_queries, -1, query_dim)
            attended, _ = self.local_attention(
                flat_query, flat_tokens, flat_tokens, need_weights=False
            )
            evidence_by_scale.append(attended.reshape(batch, num_queries, query_dim))
        evidence = torch.stack(evidence_by_scale, dim=2)

        area = (boxes_base[..., 2].abs() * boxes_base[..., 3].abs()).clamp(1e-8, 1.0)
        log_area = area.log().unsqueeze(-1)
        scale_weight = self.scale_mlp(torch.cat((normalized_queries, log_area), dim=-1)).softmax(-1)

        query_gate = self.gate_query_projection(normalized_queries).unsqueeze(2).expand(-1, -1, 3, -1)
        evidence_gate = self.gate_evidence_projection(self.gate_evidence_norm(evidence))
        confidence = logits_base.sigmoid().amax(dim=-1, keepdim=True)
        uncertainty = 1.0 - confidence
        if self.detach_rgb_uncertainty:
            uncertainty = uncertainty.detach()
        availability_value = available.to(queries.dtype).view(batch, 1, 1, 1).expand(-1, num_queries, 3, -1)
        scalar_inputs = torch.cat(
            (
                uncertainty.unsqueeze(2).expand(-1, -1, 3, -1),
                log_area.unsqueeze(2).expand(-1, -1, 3, -1),
                availability_value,
            ),
            dim=-1,
        )
        gate_input = torch.cat(
            (
                query_gate,
                evidence_gate,
                query_gate * evidence_gate,
                (query_gate - evidence_gate).abs(),
                scalar_inputs,
            ),
            dim=-1,
        )
        gate = self.gate_mlp(gate_input).sigmoid() * availability_value
        fused_evidence = (
            scale_weight.unsqueeze(-1) * gate * evidence
        ).sum(dim=2)
        delta = self.output_projection(fused_evidence)
        scale = queries.new_tensor(self.residual_scale())
        if bypass:
            delta = torch.zeros_like(delta)
        delta = delta * available.to(delta.dtype).view(batch, 1, 1)
        fused_queries = queries + scale * delta

        diagnostics = {
            "evidence": evidence,
            "gate": gate.squeeze(-1),
            "scale_weight": scale_weight,
            "delta": delta,
            "residual": scale * delta,
            "availability": available,
            "dropout_mask": dropout_mask,
            "residual_scale": scale,
            "area": area,
        }
        return fused_queries, diagnostics


def qdmf_basic_statistics(
    diagnostics: Dict[str, torch.Tensor],
    queries: torch.Tensor,
    logits_base: torch.Tensor,
    logits_final: torch.Tensor,
    boxes_base: torch.Tensor,
    boxes_final: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Tensor-only diagnostics that do not require Hungarian assignments."""
    gate = diagnostics["gate"].detach().float()
    scale = diagnostics["scale_weight"].detach().float()
    residual = diagnostics["residual"].detach().float()
    flat_gate = gate.reshape(-1)
    quantiles = torch.quantile(flat_gate, gate.new_tensor((0.1, 0.5, 0.9)))
    query_norm = queries.detach().float().norm(dim=-1).mean().clamp_min(1e-12)
    missing = ~diagnostics["availability"].detach().bool()
    missing_max = (
        residual[missing].abs().max()
        if missing.any()
        else residual.new_zeros(())
    )
    return {
        "qdmf_gate_mean": gate.mean(),
        "qdmf_gate_std": gate.std(unbiased=False),
        "qdmf_gate_p10": quantiles[0],
        "qdmf_gate_p50": quantiles[1],
        "qdmf_gate_p90": quantiles[2],
        "qdmf_gate_s8": gate[..., 0].mean(),
        "qdmf_gate_s16": gate[..., 1].mean(),
        "qdmf_gate_s32": gate[..., 2].mean(),
        "qdmf_scale_weight_s8": scale[..., 0].mean(),
        "qdmf_scale_weight_s16": scale[..., 1].mean(),
        "qdmf_scale_weight_s32": scale[..., 2].mean(),
        "qdmf_delta_abs_mean": residual.abs().mean(),
        "qdmf_delta_query_norm_ratio": residual.norm(dim=-1).mean() / query_norm,
        "qdmf_logits_delta_abs_mean": (logits_final - logits_base).detach().abs().mean(),
        "qdmf_boxes_delta_abs_mean": (boxes_final - boxes_base).detach().abs().mean(),
        "qdmf_missing_residual_abs_max": missing_max,
        "qdmf_residual_scale": diagnostics["residual_scale"].detach(),
    }
