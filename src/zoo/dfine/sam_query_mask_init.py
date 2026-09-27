"""SAM-supervised query-mask initialization for D-FINE.

The external SAM model is a training-time teacher only.  At train and test
time this module predicts query-specific masks from RGB S8 features, converts
the reliable masks to boxes, and applies a bounded correction to D-FINE's
ordinary encoder proposals.  Invalid masks fall back exactly to the detector
proposal.
"""

import math

import torch
import torch.nn as nn


class SAMQueryMaskInitializer(nn.Module):
    """Turn query masks into safe initial reference-box corrections."""

    def __init__(
        self,
        hidden_dim=256,
        source_channels=256,
        mask_dim=64,
        topk=64,
        max_mix=0.25,
        max_box_delta=0.25,
        gate_bias=-4.0,
        apply_initialization=True,
        min_area_ratio=1e-4,
        max_area_ratio=0.95,
    ):
        super().__init__()
        if mask_dim <= 0 or topk <= 0:
            raise ValueError("SQMI mask_dim and topk must be positive")
        if not 0.0 < max_mix <= 1.0:
            raise ValueError("SQMI max_mix must be in (0, 1]")
        if not 0.0 < max_box_delta <= 1.0:
            raise ValueError("SQMI max_box_delta must be in (0, 1]")
        if not 0.0 <= min_area_ratio < max_area_ratio <= 1.0:
            raise ValueError("SQMI mask area limits are invalid")

        self.mask_dim = int(mask_dim)
        self.topk = int(topk)
        self.max_mix = float(max_mix)
        self.max_box_delta = float(max_box_delta)
        self.apply_initialization = bool(apply_initialization)
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)

        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, self.mask_dim),
        )
        groups = min(8, self.mask_dim)
        while self.mask_dim % groups != 0:
            groups -= 1
        self.pixel_proj = nn.Sequential(
            nn.Conv2d(source_channels, self.mask_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, self.mask_dim),
        )

        # query, detector confidence, mask confidence, mask area, and four
        # absolute proposal disagreements form the reliability estimate.
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 7, hidden_dim // 2),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, float(gate_bias))

    @staticmethod
    def masks_to_boxes(binary_masks):
        """Convert [B,Q,H,W] masks to normalized cxcywh boxes and validity."""
        if binary_masks.ndim != 4:
            raise ValueError("SQMI masks must have shape [B,Q,H,W]")
        batch, queries, height, width = binary_masks.shape
        device = binary_masks.device
        mask = binary_masks.bool()

        x_any = mask.any(dim=-2)
        y_any = mask.any(dim=-1)
        valid = x_any.any(dim=-1) & y_any.any(dim=-1)

        x_index = torch.arange(width, device=device).view(1, 1, width)
        y_index = torch.arange(height, device=device).view(1, 1, height)
        x_min = torch.where(x_any, x_index, width).amin(dim=-1)
        x_max = torch.where(x_any, x_index, -1).amax(dim=-1)
        y_min = torch.where(y_any, y_index, height).amin(dim=-1)
        y_max = torch.where(y_any, y_index, -1).amax(dim=-1)

        # Pixel-center convention with inclusive extrema.
        x0 = x_min.to(torch.float32) / float(width)
        y0 = y_min.to(torch.float32) / float(height)
        x1 = (x_max.to(torch.float32) + 1.0) / float(width)
        y1 = (y_max.to(torch.float32) + 1.0) / float(height)
        boxes = torch.stack(
            ((x0 + x1) * 0.5, (y0 + y1) * 0.5, x1 - x0, y1 - y0),
            dim=-1,
        )
        boxes = torch.where(valid.unsqueeze(-1), boxes, torch.zeros_like(boxes))
        return boxes, valid

    def forward(self, query_features, source_feature, base_boxes, detector_scores=None):
        if query_features.ndim != 3 or source_feature.ndim != 4:
            raise ValueError("SQMI expects [B,Q,C] queries and [B,C,H,W] source")
        if base_boxes.shape[:2] != query_features.shape[:2] or base_boxes.shape[-1] != 4:
            raise ValueError("SQMI base boxes must have shape [B,Q,4]")

        # Detaching the query side protects encoder query selection; SAM loss
        # still trains this private projection and the connected RGB S8 path.
        query_embeddings = self.query_proj(query_features.detach())
        pixel_features = self.pixel_proj(source_feature)
        count = min(self.topk, query_features.shape[1])
        init_queries = query_embeddings[:, :count]
        mask_logits = torch.einsum("bqd,bdhw->bqhw", init_queries, pixel_features)
        mask_logits = mask_logits / math.sqrt(float(self.mask_dim))

        mask_probability = mask_logits.detach().sigmoid()
        binary_masks = mask_probability > 0.5
        mask_boxes, nonempty = self.masks_to_boxes(binary_masks)
        area_ratio = binary_masks.float().mean(dim=(-2, -1))
        valid = (
            nonempty
            & (area_ratio >= self.min_area_ratio)
            & (area_ratio <= self.max_area_ratio)
        )

        base = base_boxes[:, :count]
        disagreement = (mask_boxes - base.detach()).abs()
        certainty = (mask_probability - 0.5).abs().mul(2.0).mean(dim=(-2, -1))
        if detector_scores is None:
            score = torch.zeros_like(certainty)
        else:
            if detector_scores.ndim == 3:
                score = detector_scores[:, :count].sigmoid().amax(dim=-1)
            elif detector_scores.ndim == 2:
                score = detector_scores[:, :count].sigmoid()
            else:
                raise ValueError("SQMI detector scores must be [B,Q] or [B,Q,C]")
            score = score.detach()

        gate_input = torch.cat(
            (
                query_features[:, :count].detach(),
                score.unsqueeze(-1),
                certainty.unsqueeze(-1),
                area_ratio.unsqueeze(-1),
                disagreement,
            ),
            dim=-1,
        )
        mix = self.max_mix * self.gate(gate_input).sigmoid()
        mix = mix * valid.unsqueeze(-1).to(mix.dtype)
        if not self.apply_initialization:
            mix = torch.zeros_like(mix)

        delta = (mask_boxes - base.detach()).clamp(
            min=-self.max_box_delta, max=self.max_box_delta
        )
        if self.apply_initialization:
            refined_head = (base + mix * delta).clamp(1e-4, 1.0 - 1e-4)
            refined_boxes = torch.cat((refined_head, base_boxes[:, count:]), dim=1)
        else:
            # This is an intentionally exact experimental control rather than
            # a numerically-near identity.
            refined_boxes = base_boxes

        diagnostics = {
            "mask_logits": mask_logits,
            "mask_boxes": mask_boxes,
            "base_boxes": base.detach(),
            "refined_boxes": refined_boxes[:, :count].detach(),
            "valid_mask": valid,
            "mix": mix,
            "box_delta": delta,
            "effective_box_delta": mix * delta,
            "area_ratio": area_ratio,
            "certainty": certainty,
        }
        return refined_boxes, query_embeddings, pixel_features, diagnostics


__all__ = ["SAMQueryMaskInitializer"]
