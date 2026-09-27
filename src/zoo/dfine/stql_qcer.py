"""STQL and QCER building blocks.

The modules in this file implement the v1.1 research contract in
``knowledge/交接知识库/STQL_QCER_Codex实现交接_v1.1.md``.  STQL is a
training-only query/shape objective.  QCER reads coordinate-free thermal
tokens and adds a residual to the final classification logits only.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedQueryProjection(nn.Module):
    """The single query projection shared by STQL and QCER."""

    def __init__(self, query_dim: int, retrieval_dim: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(int(query_dim))
        self.proj = nn.Linear(int(query_dim), int(retrieval_dim))

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.norm(queries)), dim=-1, eps=1e-6)


class STQLPixelProjection(nn.Module):
    """Project native RGB S8 features into the shared retrieval space."""

    def __init__(self, in_channels: int, retrieval_dim: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(int(in_channels), int(retrieval_dim), 1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(feature), dim=1, eps=1e-6)


class QueryConditionedEvidenceReader(nn.Module):
    """Read global thermal evidence without assuming RGB/IR pixel alignment."""

    def __init__(
        self,
        thermal_channels: Sequence[int],
        retrieval_dim: int,
        num_classes: int,
        topk: int = 64,
        temperature: float = 0.1,
        gate_bias: float = -2.0,
        uniform_attention: bool = False,
    ):
        super().__init__()
        if not thermal_channels:
            raise ValueError("QCER requires at least one thermal feature level")
        if int(topk) <= 0:
            raise ValueError("QCER topk must be positive")
        if float(temperature) <= 0:
            raise ValueError("QCER temperature must be positive")

        dim = int(retrieval_dim)
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.uniform_attention = bool(uniform_attention)
        self.thermal_proj = nn.ModuleList(
            nn.Conv2d(int(channels), dim, 1) for channels in thermal_channels
        )
        self.objectness_norm = nn.LayerNorm(dim)
        self.objectness_head = nn.Linear(dim, 1)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.context_norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(3 * dim + 1, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        nn.init.constant_(self.gate[-1].bias, float(gate_bias))
        self.output_norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, int(num_classes))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _grid(height: int, width: int, device, dtype) -> torch.Tensor:
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def _tokens(
        self, thermal_features: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(thermal_features) < len(self.thermal_proj):
            raise RuntimeError(
                "QCER received fewer thermal levels than configured: "
                f"{len(thermal_features)} < {len(self.thermal_proj)}"
            )
        tokens, centers, levels = [], [], []
        for level, (project, feature) in enumerate(
            zip(self.thermal_proj, thermal_features)
        ):
            projected = project(feature)
            batch, _, height, width = projected.shape
            tokens.append(projected.flatten(2).transpose(1, 2))
            centers.append(self._grid(height, width, feature.device, feature.dtype))
            levels.append(
                torch.full(
                    (height * width,),
                    level,
                    device=feature.device,
                    dtype=torch.long,
                )
            )
        return torch.cat(tokens, dim=1), torch.cat(centers), torch.cat(levels)

    @staticmethod
    def _gather_tokens(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return values.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        thermal_features: Sequence[torch.Tensor],
        valid_ir: Optional[torch.Tensor] = None,
        availability: Optional[torch.Tensor] = None,
        bypass: bool = False,
    ) -> Dict[str, torch.Tensor]:
        tokens, centers, level_ids = self._tokens(thermal_features)
        batch, token_count, _ = tokens.shape
        if valid_ir is None:
            valid_ir = torch.ones(
                batch, token_count, dtype=torch.bool, device=tokens.device
            )
        else:
            valid_ir = valid_ir.to(device=tokens.device, dtype=torch.bool)
            if valid_ir.shape != (batch, token_count):
                raise RuntimeError(
                    f"valid_ir must be {(batch, token_count)}, got {tuple(valid_ir.shape)}"
                )
        if availability is None:
            availability = torch.ones(batch, dtype=torch.bool, device=tokens.device)
        else:
            availability = availability.to(device=tokens.device, dtype=torch.bool).reshape(batch)

        objectness = self.objectness_head(self.objectness_norm(tokens)).squeeze(-1)
        masked_objectness = objectness.masked_fill(~valid_ir, torch.finfo(objectness.dtype).min)
        selected_count = min(self.topk, token_count)
        selected_indices = masked_objectness.topk(selected_count, dim=1).indices
        selected_valid = valid_ir.gather(1, selected_indices)
        selected_tokens = self._gather_tokens(tokens, selected_indices)

        keys = F.normalize(self.key_proj(selected_tokens), dim=-1, eps=1e-6)
        values = self.value_proj(selected_tokens)
        attention_logits = torch.einsum("bnd,bpd->bnp", query_embeddings, keys)
        attention_logits = attention_logits / self.temperature
        expanded_valid = selected_valid[:, None, :]
        if self.uniform_attention:
            attention = expanded_valid.to(values.dtype)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1.0)
            attention = attention.expand(-1, query_embeddings.shape[1], -1)
        else:
            attention_logits = attention_logits.masked_fill(
                ~expanded_valid, torch.finfo(attention_logits.dtype).min
            )
            attention = F.softmax(attention_logits, dim=-1)
            no_valid = ~selected_valid.any(dim=-1)
            if no_valid.any():
                attention = attention.masked_fill(no_valid[:, None, None], 0.0)

        context = torch.einsum("bnp,bpd->bnd", attention, values)
        normalized_context = self.context_norm(context)
        valid_count = selected_valid.sum(dim=-1).clamp_min(1)
        entropy_denominator = valid_count.to(attention.dtype).log()
        entropy = -(attention.clamp_min(1e-12).log() * attention).sum(dim=-1)
        entropy = torch.where(
            valid_count[:, None] > 1,
            entropy / entropy_denominator[:, None].clamp_min(1e-6),
            torch.zeros_like(entropy),
        )
        gate_input = torch.cat(
            (
                query_embeddings,
                normalized_context,
                query_embeddings * normalized_context,
                entropy.unsqueeze(-1),
            ),
            dim=-1,
        )
        gate = self.gate(gate_input).sigmoid()
        delta = gate * self.output(self.output_norm(context))
        delta = delta * availability[:, None, None].to(delta.dtype)
        if bypass:
            delta = torch.zeros_like(delta)

        return {
            "delta_logits": delta,
            "objectness_logits": objectness,
            "valid_ir": valid_ir,
            "grid_centers": centers,
            "level_ids": level_ids,
            "selected_indices": selected_indices,
            "selected_valid": selected_valid,
            "attention": attention,
            "attention_entropy": entropy,
            "gate": gate,
            "availability": availability,
        }


def _box_occupancy(
    boxes_cxcywh: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Exact fractional occupancy of normalized boxes on a feature grid."""
    if boxes_cxcywh.numel() == 0:
        return boxes_cxcywh.new_zeros((0, height, width))
    boxes = boxes_cxcywh.float()
    x0 = (boxes[:, 0] - boxes[:, 2] / 2).clamp(0, 1)
    y0 = (boxes[:, 1] - boxes[:, 3] / 2).clamp(0, 1)
    x1 = (boxes[:, 0] + boxes[:, 2] / 2).clamp(0, 1)
    y1 = (boxes[:, 1] + boxes[:, 3] / 2).clamp(0, 1)
    cell_x0 = torch.arange(width, device=boxes.device, dtype=boxes.dtype) / width
    cell_x1 = cell_x0 + 1.0 / width
    cell_y0 = torch.arange(height, device=boxes.device, dtype=boxes.dtype) / height
    cell_y1 = cell_y0 + 1.0 / height
    overlap_x = (
        torch.minimum(x1[:, None], cell_x1[None])
        - torch.maximum(x0[:, None], cell_x0[None])
    ).clamp_min(0)
    overlap_y = (
        torch.minimum(y1[:, None], cell_y1[None])
        - torch.maximum(y0[:, None], cell_y0[None])
    ).clamp_min(0)
    return overlap_y[:, :, None] * overlap_x[:, None, :] * (height * width)


def stql_shape_loss(
    query_embeddings: torch.Tensor,
    pixel_features: torch.Tensor,
    targets: Sequence[Dict[str, torch.Tensor]],
    indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    supervision: str,
    temperature: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Balanced soft target/background response loss for matched queries."""
    if supervision not in {"sam", "box"}:
        raise ValueError(f"Unsupported STQL supervision: {supervision}")
    _, _, height, width = pixel_features.shape
    image_losses: List[torch.Tensor] = []
    foreground_response: List[torch.Tensor] = []
    background_response: List[torch.Tensor] = []
    valid_instances = 0

    for batch_index, ((source_indices, target_indices), target) in enumerate(
        zip(indices, targets)
    ):
        if len(source_indices) == 0:
            continue
        boxes = torch.as_tensor(target["boxes"], device=pixel_features.device).float()
        quality = target.get("sam_quality")
        masks = target.get("masks")
        per_image: List[torch.Tensor] = []
        for source_index, target_index in zip(source_indices.tolist(), target_indices.tolist()):
            quality_value = None
            if quality is not None:
                flat_quality = torch.as_tensor(quality).reshape(-1)
                if flat_quality.numel() == 1:
                    quality_value = float(flat_quality[0])
                elif target_index < flat_quality.numel():
                    quality_value = float(flat_quality[target_index])
            if supervision == "sam":
                if masks is None or target_index >= len(masks):
                    continue
                if quality_value is None or quality_value <= 0:
                    continue
                mask = torch.as_tensor(masks[target_index], device=pixel_features.device).float()
                y = F.adaptive_avg_pool2d(mask[None, None], (height, width))[0, 0]
            else:
                # BOX uses exactly the same accepted instance set as SAM.
                if quality_value is None or quality_value <= 0:
                    continue
                y = _box_occupancy(boxes[target_index : target_index + 1], height, width)[0]

            neighborhood_box = boxes[target_index : target_index + 1].clone()
            neighborhood_box[:, 2:] = (2.0 * neighborhood_box[:, 2:]).clamp(max=2.0)
            neighborhood = _box_occupancy(neighborhood_box, height, width)[0]
            if len(boxes) > 1:
                others = torch.cat((boxes[:target_index], boxes[target_index + 1 :]), dim=0)
                other_occupancy = _box_occupancy(others, height, width).amax(dim=0)
            else:
                other_occupancy = torch.zeros_like(neighborhood)

            foreground_weight = neighborhood * y
            background_weight = neighborhood * (1.0 - y) * (1.0 - other_occupancy)
            foreground_mass = foreground_weight.sum()
            background_mass = background_weight.sum()
            if foreground_mass <= 1e-6 or background_mass <= 1e-6:
                continue

            response = torch.einsum(
                "d,dhw->hw",
                query_embeddings[batch_index, source_index],
                pixel_features[batch_index],
            ) / float(temperature)
            instance_loss = 0.5 * (
                foreground_weight * F.softplus(-response)
            ).sum() / foreground_mass
            instance_loss = instance_loss + 0.5 * (
                background_weight * F.softplus(response)
            ).sum() / background_mass
            per_image.append(instance_loss)
            foreground_response.append((foreground_weight * response).sum() / foreground_mass)
            background_response.append((background_weight * response).sum() / background_mass)
            valid_instances += 1
        if per_image:
            image_losses.append(torch.stack(per_image).mean())

    connected_zero = query_embeddings.sum() * 0.0 + pixel_features.sum() * 0.0
    loss = torch.stack(image_losses).mean() if image_losses else connected_zero
    if foreground_response:
        fg = torch.stack(foreground_response).mean().detach()
        bg = torch.stack(background_response).mean().detach()
    else:
        fg = connected_zero.detach()
        bg = connected_zero.detach()
    stats = {
        "foreground_response": fg,
        "background_response": bg,
        "response_margin": fg - bg,
        "valid_instances": query_embeddings.new_tensor(float(valid_instances)),
    }
    return loss, stats


def qcer_objectness_loss(
    logits: torch.Tensor,
    valid_ir: torch.Tensor,
    grid_centers: torch.Tensor,
    level_ids: torch.Tensor,
    targets: Sequence[Dict[str, torch.Tensor]],
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Focal objectness loss with one nearest-cell fallback per tiny box/level."""
    losses: List[torch.Tensor] = []
    fallback_count = 0
    positive_count = 0
    known_images = 0
    unique_levels = level_ids.unique(sorted=True)
    for batch_index, target in enumerate(targets):
        if "infrared_boxes" not in target:
            continue
        known = target.get("infrared_label_known")
        if known is not None and not bool(torch.as_tensor(known).reshape(-1)[0]):
            continue
        known_images += 1
        valid = valid_ir[batch_index]
        labels = torch.zeros_like(logits[batch_index])
        boxes = torch.as_tensor(target["infrared_boxes"], device=logits.device).float()
        for box in boxes:
            x0, y0 = box[0] - box[2] / 2, box[1] - box[3] / 2
            x1, y1 = box[0] + box[2] / 2, box[1] + box[3] / 2
            inside = (
                (grid_centers[:, 0] >= x0)
                & (grid_centers[:, 0] <= x1)
                & (grid_centers[:, 1] >= y0)
                & (grid_centers[:, 1] <= y1)
                & valid
            )
            labels[inside] = 1.0
            for level in unique_levels:
                level_valid = (level_ids == level) & valid
                if not level_valid.any() or (inside & (level_ids == level)).any():
                    continue
                center = box[:2]
                distance = ((grid_centers - center) ** 2).sum(dim=-1)
                distance = distance.masked_fill(~level_valid, float("inf"))
                labels[distance.argmin()] = 1.0
                fallback_count += 1

        selected_logits = logits[batch_index][valid]
        selected_labels = labels[valid]
        if selected_logits.numel() == 0:
            continue
        probability = selected_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(
            selected_logits, selected_labels, reduction="none"
        )
        p_t = probability * selected_labels + (1.0 - probability) * (1.0 - selected_labels)
        alpha_t = alpha * selected_labels + (1.0 - alpha) * (1.0 - selected_labels)
        focal = alpha_t * (1.0 - p_t).pow(gamma) * ce
        positives = int(selected_labels.sum().item())
        positive_count += positives
        losses.append(focal.sum() / max(positives, 1))

    connected_zero = logits.sum() * 0.0
    loss = torch.stack(losses).mean() if losses else connected_zero
    stats = {
        "positive_tokens": logits.new_tensor(float(positive_count)),
        "fallback_tokens": logits.new_tensor(float(fallback_count)),
        "known_images": logits.new_tensor(float(known_images)),
    }
    return loss, stats


__all__ = [
    "QueryConditionedEvidenceReader",
    "SharedQueryProjection",
    "STQLPixelProjection",
    "qcer_objectness_loss",
    "stql_shape_loss",
]
