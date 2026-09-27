"""Target-evidence thermal conditioning for the RGB S16 detection feature."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetEvidenceThermalFusion(nn.Module):
    """Extract IR target/background evidence and condition RGB S16 without
    assuming pixelwise RGB-IR correspondence.

    IR boxes are used only to supervise the dense IR objectness prediction.
    Candidate construction always uses that prediction at both train and test
    time. The detector-facing output is a bounded residual on RGB S16.
    """

    def __init__(
        self,
        visible_channels: int = 128,
        thermal_s8_channels: int = 256,
        thermal_s16_channels: int = 128,
        embed_dim: int = 64,
        num_candidates: int = 4,
        max_rms_ratio: float = 0.03,
        final_residual_scale: float = 0.5,
        objectness_loss_weight: float = 0.1,
        layout_fix: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % 4:
            raise ValueError("embed_dim must be divisible by four")
        if not 0.0 < max_rms_ratio <= 0.1:
            raise ValueError("max_rms_ratio must be in (0, 0.1]")
        if not 0.0 <= final_residual_scale <= 1.0:
            raise ValueError("final_residual_scale must be in [0, 1]")

        self.visible_channels = int(visible_channels)
        self.num_candidates = int(num_candidates)
        self.max_rms_ratio = float(max_rms_ratio)
        self.final_residual_scale = float(final_residual_scale)
        self.objectness_loss_weight = float(objectness_loss_weight)
        # False preserves the exact forward used by historical E2 weights.
        self.layout_fix = bool(layout_fix)
        self.eps = 1e-6

        self.objectness_head = nn.Sequential(
            nn.Conv2d(thermal_s8_channels, thermal_s8_channels // 2, 3, padding=1),
            nn.GroupNorm(1, thermal_s8_channels // 2),
            nn.SiLU(inplace=False),
            nn.Conv2d(thermal_s8_channels // 2, 1, 1),
        )
        self.object_token_s8 = nn.Conv2d(thermal_s8_channels, embed_dim, 1)
        self.object_token_s16 = nn.Conv2d(thermal_s16_channels, embed_dim, 1)
        self.visible_query = nn.Conv2d(visible_channels, embed_dim, 1)
        self.evidence_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.evidence_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.background_norm = nn.LayerNorm(embed_dim)
        self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.residual_generator = nn.Sequential(
            nn.Linear(4 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim),
            nn.SiLU(inplace=False),
            nn.Linear(2 * embed_dim, 2 * visible_channels),
        )
        nn.init.zeros_(self.residual_generator[-1].weight)
        nn.init.zeros_(self.residual_generator[-1].bias)

        self.reliability_gate = nn.Sequential(
            nn.Linear(embed_dim + 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=False),
            nn.Linear(embed_dim, 1),
        )
        nn.init.zeros_(self.reliability_gate[-1].weight)
        nn.init.zeros_(self.reliability_gate[-1].bias)

        self.last_objectness_logits = None
        self.last_objectness_loss = None
        self.last_candidate_scores = None
        self.last_gate_mean = None
        self.last_object_attention_mean = None
        self.last_update_ratio = None

    @staticmethod
    def _box_values(boxes: torch.Tensor):
        fmt = getattr(getattr(boxes, "format", None), "value", None)
        fmt = str(fmt or getattr(boxes, "format", "")).lower()
        values = torch.as_tensor(boxes).float()
        if values.numel() == 0:
            return values.reshape(0, 4), fmt
        if not fmt:
            # The project's final RGB-T preset normalizes boxes to CXCYWH.
            return values.reshape(-1, 4), "cxcywh"
        if "cxcywh" in fmt or "cxcywh" in fmt.replace(" ", ""):
            return values.reshape(-1, 4), "cxcywh"
        if "xyxy" in fmt:
            return values.reshape(-1, 4), "xyxy"
        raise ValueError(f"Unsupported infrared box format: {fmt!r}")

    def _target_heatmaps(self, targets, height, width, device, dtype):
        heatmaps = torch.zeros((len(targets), 1, height, width), device=device, dtype=dtype)
        for batch_index, target in enumerate(targets):
            boxes = target.get("infrared_boxes")
            if boxes is None:
                raise KeyError("M-OTE2 training requires infrared_boxes in targets")
            values, fmt = self._box_values(boxes)
            if values.numel() == 0:
                continue
            values = values.to(device=device, dtype=torch.float32)
            normalized = bool(values.abs().max() <= 1.5)
            for box in values:
                if fmt == "cxcywh":
                    cx, cy, bw, bh = box.unbind(0)
                else:
                    x1, y1, x2, y2 = box.unbind(0)
                    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                    bw, bh = (x2 - x1).clamp_min(0), (y2 - y1).clamp_min(0)
                if normalized:
                    cx, bw = cx * width, bw * width
                    cy, bh = cy * height, bh * height
                gx = (cx - 0.5).clamp(0, width - 1)
                gy = (cy - 0.5).clamp(0, height - 1)
                sigma_x = (bw * 0.20).clamp(0.75, 2.5)
                sigma_y = (bh * 0.20).clamp(0.75, 2.5)
                ys = torch.arange(height, device=device, dtype=torch.float32)
                xs = torch.arange(width, device=device, dtype=torch.float32)
                gaussian = torch.exp(
                    -0.5
                    * (
                        ((ys[:, None] - gy) / sigma_y).square()
                        + ((xs[None, :] - gx) / sigma_x).square()
                    )
                )
                heatmaps[batch_index, 0] = torch.maximum(
                    heatmaps[batch_index, 0], gaussian.to(dtype)
                )
        return heatmaps

    @staticmethod
    def _objectness_loss(logits: torch.Tensor, target: torch.Tensor):
        probability = logits.sigmoid()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        positive_weight = target
        negative_weight = 1.0 - target
        positive = (
            positive_weight * bce * (1.0 - probability).square()
        ).sum(dim=(1, 2, 3)) / positive_weight.sum(dim=(1, 2, 3)).clamp_min(1.0)
        negative = (
            negative_weight * bce * probability.square()
        ).sum(dim=(1, 2, 3)) / negative_weight.sum(dim=(1, 2, 3)).clamp_min(1.0)
        has_positive = positive_weight.sum(dim=(1, 2, 3)) > 0
        per_image = torch.where(has_positive, 0.5 * (positive + negative), negative)
        return per_image.mean()

    def _candidate_tokens(self, thermal_s8, thermal_s16, objectness_logits):
        batch, _, h8, w8 = thermal_s8.shape
        probability = objectness_logits.sigmoid()
        pooled = F.max_pool2d(probability, kernel_size=3, stride=1, padding=1)
        peak_probability = probability * (probability >= pooled).to(probability.dtype)
        scores, indices = peak_probability.flatten(1).topk(
            min(self.num_candidates, h8 * w8), dim=1
        )
        y8 = torch.div(indices, w8, rounding_mode="floor")
        x8 = indices.remainder(w8)

        projected_s8 = self.object_token_s8(thermal_s8)
        projected_s16 = self.object_token_s16(thermal_s16)
        h16, w16 = thermal_s16.shape[-2:]
        x16 = ((x8.float() + 0.5) * w16 / w8).long().clamp(0, w16 - 1)
        y16 = ((y8.float() + 0.5) * h16 / h8).long().clamp(0, h16 - 1)
        linear16 = y16 * w16 + x16

        s8_flat = projected_s8.flatten(2).transpose(1, 2)
        s16_flat = projected_s16.flatten(2).transpose(1, 2)
        token_s8 = s8_flat.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, s8_flat.shape[-1])
        )
        token_s16 = s16_flat.gather(
            1,
            linear16.unsqueeze(-1).expand(-1, -1, s16_flat.shape[-1]),
        )
        object_tokens = self.token_norm(token_s8 + token_s16)

        background_weights = (1.0 - probability).flatten(2)
        background_s8 = torch.bmm(
            background_weights, s8_flat
        ) / background_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        p16 = F.interpolate(probability, size=(h16, w16), mode="bilinear", align_corners=False)
        background_weights16 = (1.0 - p16).flatten(2)
        background_s16 = torch.bmm(
            background_weights16, s16_flat
        ) / background_weights16.sum(dim=-1, keepdim=True).clamp_min(1.0)
        background_token = self.background_norm(background_s8 + background_s16)
        return object_tokens, scores, background_token, probability

    def forward(
        self,
        visible_s16: torch.Tensor,
        thermal_s8: torch.Tensor,
        thermal_s16: torch.Tensor,
        content_mask: torch.Tensor | None = None,
        targets=None,
    ):
        if visible_s16.ndim != 4 or thermal_s8.ndim != 4 or thermal_s16.ndim != 4:
            raise ValueError("M-OTE2 expects BCHW feature tensors")
        if visible_s16.shape[0] != thermal_s8.shape[0] or visible_s16.shape[0] != thermal_s16.shape[0]:
            raise ValueError("M-OTE2 batch sizes must agree")
        if content_mask is None:
            content_mask = torch.ones(
                visible_s16.shape[0], device=visible_s16.device, dtype=torch.bool
            )
        content_mask = content_mask.to(device=visible_s16.device, dtype=torch.bool).reshape(
            -1, 1, 1, 1
        )

        objectness_logits = self.objectness_head(thermal_s8)
        if objectness_logits.shape[-2:] != thermal_s8.shape[-2:]:
            raise RuntimeError("IR objectness head changed the S8 grid")
        object_tokens, candidate_scores, background_token, probability = self._candidate_tokens(
            thermal_s8, thermal_s16, objectness_logits
        )

        batch, _, height, width = visible_s16.shape
        query_map = self.visible_query(visible_s16)
        queries = query_map.flatten(2).transpose(1, 2)
        null = self.null_token.to(dtype=object_tokens.dtype).expand(batch, -1, -1)
        keys = torch.cat(
            (object_tokens, background_token, null), dim=1
        )
        # Candidate confidence affects selection without imposing a spatial
        # RGB-to-IR coordinate correspondence.
        key_bias = torch.cat(
            (
                candidate_scores.clamp_min(1e-6).log(),
                torch.zeros(batch, 2, device=queries.device, dtype=queries.dtype),
            ),
            dim=1,
        )
        query = self.evidence_query(queries)
        key = self.evidence_key(keys)
        logits = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(query.shape[-1])
        logits = logits + key_bias[:, None, :]
        attention = logits.softmax(dim=-1)
        object_attention = attention[..., : object_tokens.shape[1]]
        background_attention = attention[..., object_tokens.shape[1] : object_tokens.shape[1] + 1]
        object_mass = object_attention.sum(dim=-1, keepdim=True)
        weighted_object = torch.bmm(
            object_attention, object_tokens
        ) / object_mass.clamp_min(1e-6)
        evidence = object_mass * (weighted_object - background_token)

        entropy = -(
            attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()
        ).sum(dim=-1, keepdim=True)
        candidate_confidence = candidate_scores.max(dim=1).values
        max_candidate_score = candidate_confidence[:, None, None].expand(
            -1, queries.shape[1], -1
        )
        margin = object_attention.max(dim=-1, keepdim=True).values - background_attention
        gate_input = torch.cat(
            (queries, max_candidate_score, object_mass, margin, entropy), dim=-1
        )
        gate = self.reliability_gate(gate_input).sigmoid()

        def generated_response(evidence_value):
            interaction = torch.cat(
                (
                    queries,
                    evidence_value,
                    queries * evidence_value,
                    (queries - evidence_value).abs(),
                ),
                dim=-1,
            )
            gamma, beta = self.residual_generator(interaction).chunk(2, dim=-1)
            gamma = gamma.transpose(1, 2).reshape(batch, self.visible_channels, height, width)
            beta = beta.transpose(1, 2).reshape(batch, self.visible_channels, height, width)
            visible_float = visible_s16.float()
            normalized = F.group_norm(visible_float, num_groups=1)
            return normalized * torch.tanh(gamma) + torch.tanh(beta)

        full_response = generated_response(evidence)
        null_response = generated_response(torch.zeros_like(evidence))
        raw_update = full_response - null_response
        if not self.layout_fix:
            raw_update = raw_update.transpose(1, 2).reshape(batch, self.visible_channels, height, width)
        confidence = candidate_confidence.reshape(batch, 1, 1, 1)
        gate_map = gate.transpose(1, 2).reshape(batch, 1, height, width)
        raw_update = raw_update * confidence * gate_map
        raw_update = raw_update * content_mask.to(raw_update.dtype)

        visible_float = visible_s16.float()
        visible_rms = visible_float.square().mean(dim=(1, 2, 3), keepdim=True).add(self.eps).sqrt()
        raw_rms = raw_update.square().mean(dim=(1, 2, 3), keepdim=True).add(self.eps).sqrt()
        cap = self.max_rms_ratio * visible_rms
        limiter = cap / torch.sqrt(raw_rms.square() + cap.square() + self.eps)
        update = raw_update * limiter * self.final_residual_scale
        output = visible_s16 + update.to(visible_s16.dtype)

        objectness_loss = None
        if self.training and targets is not None:
            heatmap_target = self._target_heatmaps(
                targets,
                objectness_logits.shape[-2],
                objectness_logits.shape[-1],
                objectness_logits.device,
                objectness_logits.dtype,
            )
            raw_loss = self._objectness_loss(objectness_logits, heatmap_target)
            objectness_loss = self.objectness_loss_weight * raw_loss

        self.last_objectness_logits = objectness_logits.detach()
        self.last_objectness_loss = None if objectness_loss is None else objectness_loss.detach()
        self.last_candidate_scores = candidate_scores.detach()
        self.last_gate_mean = gate.detach().mean()
        self.last_object_attention_mean = object_mass.detach().mean()
        self.last_update_ratio = (
            update.detach().float().square().mean(dim=(1, 2, 3)).sqrt()
            / visible_rms.squeeze(-1).squeeze(-1).squeeze(-1)
        ).mean()
        return output, objectness_loss
