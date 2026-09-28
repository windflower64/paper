"""SAM-supervised, query-conditioned RGB evidence reader (S-QER1).

The teacher mask is used only by the criterion during training.  Inference
reads RGB S4/S8 features and adds a bounded class-logit residual to the
detector's existing query scores; predicted boxes are never modified.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DepthwiseResidual(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.norm(self.pointwise(self.depthwise(x))))


class _QueryRegionBlock(nn.Module):
    def __init__(self, dim: int = 64, heads: int = 4):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, batch_first=True, dropout=0.0,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        return_attention: bool,
    ):
        if return_attention:
            attended, weights = self.attention(
                self.query_norm(queries),
                self.memory_norm(memory),
                self.memory_norm(memory),
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            attended, weights = self.attention(
                self.query_norm(queries),
                self.memory_norm(memory),
                self.memory_norm(memory),
                need_weights=False,
            )
        queries = queries + attended
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries, weights


class SAMQueryEvidenceReader(nn.Module):
    """Read dense RGB S4 evidence for the highest-scoring detector queries."""

    def __init__(
        self,
        s4_channels: int = 64,
        s8_channels: int = 256,
        query_dim: int = 128,
        num_classes: int = 1,
        topk: int = 64,
        roi_size: int = 16,
        roi_expand: float = 2.0,
        min_roi_width_px: float = 32.0,
        min_roi_height_px: float = 32.0,
        evidence_only: bool = False,
        image_height: int = 512,
        image_width: int = 640,
    ):
        super().__init__()
        self.topk = int(topk)
        self.roi_size = int(roi_size)
        self.roi_expand = float(roi_expand)
        self.min_roi_width = float(min_roi_width_px) / float(image_width)
        self.min_roi_height = float(min_roi_height_px) / float(image_height)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.dim = 64
        self.query_dim = int(query_dim)
        self.evidence_only = bool(evidence_only)

        self.s4_projection = nn.Sequential(
            nn.Conv2d(s4_channels, self.dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, self.dim),
            nn.SiLU(),
            _DepthwiseResidual(self.dim),
            _DepthwiseResidual(self.dim),
        )
        self.s8_projection = nn.Sequential(
            nn.Conv2d(s8_channels, self.dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, self.dim),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * self.dim, self.dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, self.dim),
            nn.SiLU(),
        )

        self.query_projection = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, self.dim),
        )
        self.region_embeddings = nn.Parameter(torch.empty(3, self.dim))
        nn.init.normal_(self.region_embeddings, std=0.02)
        self.query_blocks = nn.ModuleList(
            [_QueryRegionBlock(self.dim, heads=4) for _ in range(2)]
        )
        # S-QER1 keeps a direct decoder-query shortcut. S-QER2 only lets
        # pooled local image-feature contrasts enter the detector score head.
        quality_input_dim = (
            2 * self.dim
            if self.evidence_only
            else query_dim + 2 * self.dim + 4
        )
        self.quality_head = nn.Sequential(
            nn.LayerNorm(quality_input_dim),
            nn.Linear(quality_input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, int(num_classes)),
        )
        nn.init.zeros_(self.quality_head[-1].weight)
        nn.init.zeros_(self.quality_head[-1].bias)

    def _select_queries(self, logits: torch.Tensor) -> torch.Tensor:
        count = min(self.topk, int(logits.shape[1]))
        rank = logits.detach().float().sigmoid().amax(dim=-1)
        return rank.topk(count, dim=1, largest=True, sorted=True).indices

    def _roi_grid(self, boxes: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch, count = indices.shape
        chosen = boxes.gather(1, indices.unsqueeze(-1).expand(-1, -1, 4)).detach().float()
        cx, cy, bw, bh = chosen.unbind(dim=-1)
        bw = (bw * self.roi_expand).clamp_min(self.min_roi_width)
        bh = (bh * self.roi_expand).clamp_min(self.min_roi_height)
        x1 = (cx - bw * 0.5).clamp(0.0, 1.0)
        x2 = (cx + bw * 0.5).clamp(0.0, 1.0)
        y1 = (cy - bh * 0.5).clamp(0.0, 1.0)
        y2 = (cy + bh * 0.5).clamp(0.0, 1.0)

        step = (torch.arange(self.roi_size, device=boxes.device, dtype=torch.float32) + 0.5)
        fx = step / self.roi_size
        fy = step / self.roi_size
        xx = x1[..., None, None] + (x2 - x1)[..., None, None] * fx.view(1, 1, 1, -1)
        yy = y1[..., None, None] + (y2 - y1)[..., None, None] * fy.view(1, 1, -1, 1)
        grid_x = (xx.expand(-1, -1, self.roi_size, -1) * 2.0) - 1.0
        grid_y = (yy.expand(-1, -1, -1, self.roi_size) * 2.0) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)
        # Keep query and ROI axes explicit for paired teacher sampling.
        return grid

    def forward(
        self,
        s4: torch.Tensor,
        s8: torch.Tensor,
        query_features: torch.Tensor,
        base_logits: torch.Tensor,
        base_boxes: torch.Tensor,
        *,
        bypass: bool = False,
        return_aux: bool = False,
    ):
        if bypass:
            return {
                "delta_logits": torch.zeros_like(base_logits),
                "query_indices": None,
                "roi_grid": None,
                "attention": None,
            }

        if s4.ndim != 4 or s4.shape[1] != self.s4_projection[0].in_channels:
            raise RuntimeError(f"S-QER1 expected S4 channels, got {tuple(s4.shape)}")
        if s8.ndim != 4 or s8.shape[1] != self.s8_projection[0].in_channels:
            raise RuntimeError(f"S-QER1 expected S8 channels, got {tuple(s8.shape)}")
        if query_features.ndim != 3 or query_features.shape[-1] != self.query_dim:
            raise RuntimeError(
                f"S-QER expected query features [B,Q,{self.query_dim}], "
                f"got {tuple(query_features.shape)}"
            )
        if base_boxes.shape[-1] != 4 or base_logits.shape[:2] != base_boxes.shape[:2]:
            raise RuntimeError("S-QER1 requires matching [B,Q] logits and cxcywh boxes")
        if s4.shape[0] != query_features.shape[0] or s8.shape[0] != query_features.shape[0]:
            raise RuntimeError("S-QER1 feature/query batch sizes do not match")

        selected = self._select_queries(base_logits)
        fine = self.s4_projection(s4)
        coarse = self.s8_projection(s8)
        coarse = F.interpolate(coarse, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        memory_map = self.fusion(torch.cat((fine, coarse), dim=1))

        roi_grid = self._roi_grid(base_boxes, selected)
        batch, count = selected.shape
        sampling_grid = roi_grid.reshape(
            batch, count * self.roi_size, self.roi_size, 2
        ).to(dtype=memory_map.dtype)
        sampled = F.grid_sample(
            memory_map,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        # [B,C,K*R,R] -> [B,K,R*R,C]
        memory = sampled.reshape(
            batch, self.dim, count, self.roi_size, self.roi_size
        ).permute(0, 2, 3, 4, 1).reshape(batch * count, self.roi_size ** 2, self.dim)

        selected_query = query_features.gather(
            1, selected.unsqueeze(-1).expand(-1, -1, query_features.shape[-1])
        )
        q = self.query_projection(selected_query).reshape(batch * count, 1, self.dim)
        q = q + self.region_embeddings.unsqueeze(0)
        attention = None
        for layer_index, block in enumerate(self.query_blocks):
            q, weights = block(
                q,
                memory,
                return_attention=layer_index == len(self.query_blocks) - 1,
            )
            if weights is not None:
                attention = weights.float().mean(dim=1).reshape(
                    batch, count, 3, self.roi_size ** 2
                )

        if attention is None:
            raise RuntimeError("S-QER1 final query-region attention was not computed")
        evidence = None
        if self.evidence_only:
            # The query and role embeddings choose where to read. They do not
            # enter the residual head. Its inputs are weighted RGB memory
            # contrasts, so identical ROI evidence cannot create a score delta.
            memory_by_query = memory.float().reshape(
                batch, count, self.roi_size ** 2, self.dim
            )
            pooled = torch.einsum(
                "bkrt,bktd->bkrd", attention.float(), memory_by_query
            )
            evidence = torch.cat(
                (
                    pooled[:, :, 0] - pooled[:, :, 2],
                    pooled[:, :, 1] - pooled[:, :, 2],
                ),
                dim=-1,
            )
            zero_evidence = torch.zeros_like(evidence[:1, :1])
            delta_selected = (
                self.quality_head(evidence)
                - self.quality_head(zero_evidence)
            ).tanh()
            evidence_norm = torch.linalg.vector_norm(
                evidence, ord=2, dim=-1, keepdim=True
            )
            evidence_gate = evidence_norm / (evidence_norm + 1.0)
            delta_selected = delta_selected * evidence_gate
        else:
            regions = q.reshape(batch, count, 3, self.dim)
            entropy = -(
                attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(self.roi_size ** 2)
            base_area = base_boxes.gather(
                1, selected.unsqueeze(-1).expand(-1, -1, 4)
            )[..., 2:4].float().clamp_min(0.0).prod(dim=-1, keepdim=True)
            quality_input = torch.cat(
                (
                    selected_query.float(),
                    (regions[:, :, 0] - regions[:, :, 2]).float(),
                    (regions[:, :, 1] - regions[:, :, 2]).float(),
                    entropy.float(),
                    base_area,
                ),
                dim=-1,
            )
            delta_selected = self.quality_head(quality_input).tanh()
        delta = torch.zeros_like(base_logits).scatter(
            1,
            selected.unsqueeze(-1).expand(-1, -1, base_logits.shape[-1]),
            delta_selected.to(dtype=base_logits.dtype),
        )
        return {
            "delta_logits": delta,
            "query_indices": selected,
            "roi_grid": roi_grid if return_aux else None,
            "attention": attention if return_aux else None,
            "evidence": evidence if self.evidence_only and return_aux else None,
        }


def _box_occupancy(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Rasterize normalized cxcywh boxes at pixel centers."""
    y = (
        (torch.arange(height, device=boxes.device, dtype=torch.float32) + 0.5)
        / height
    ).view(1, 1, height, 1)
    x = (
        (torch.arange(width, device=boxes.device, dtype=torch.float32) + 0.5)
        / width
    ).view(1, 1, 1, width)
    cx, cy, bw, bh = boxes.float().unbind(dim=-1)
    return (
        (x >= cx[:, :, None, None] - bw[:, :, None, None] * 0.5)
        & (x <= cx[:, :, None, None] + bw[:, :, None, None] * 0.5)
        & (y >= cy[:, :, None, None] - bh[:, :, None, None] * 0.5)
        & (y <= cy[:, :, None, None] + bh[:, :, None, None] * 0.5)
    ).float()


def _region_distributions(
    occupancy: torch.Tensor,
    min_region_mass: float = 0.25,
    eps: float = 1e-4,
):
    """Return normalized inside/transition/near-background distributions."""
    u = occupancy.float().clamp(0.0, 1.0)
    if u.ndim == 2:
        u = u[None, None]
    dilate3 = F.max_pool2d(u, kernel_size=3, stride=1, padding=1)
    erode3 = -F.max_pool2d(-u, kernel_size=3, stride=1, padding=1)
    dilate5 = F.max_pool2d(u, kernel_size=5, stride=1, padding=2)
    regions = torch.cat(
        (
            u,
            (dilate3 - erode3).clamp_min(0.0),
            ((dilate5 - dilate3).clamp_min(0.0) * (1.0 - u)),
        ),
        dim=1,
    )
    masses = regions.sum(dim=(-2, -1))
    valid = (masses >= float(min_region_mass)).all(dim=1)
    distributions = (regions + eps) / (
        masses[:, :, None, None] + eps * regions.shape[-2] * regions.shape[-1]
    )
    return distributions.flatten(2), valid, masses.flatten(1)


def sqer_shape_loss(
    outputs: dict,
    targets,
    matched_indices,
    supervision: str = "sam",
    min_region_mass: float = 0.25,
):
    """KL supervision of the reader's three spatial attention distributions.

    Eligibility is paired per matched query: if either SAM or BOX cannot form
    all three regions, that query is excluded from either teacher arm.
    """
    if supervision not in {"sam", "box"}:
        raise ValueError(f"unsupported S-QER supervision {supervision!r}")
    required = {"sqer_attention", "sqer_roi_grid", "sqer_query_indices"}
    missing = required.difference(outputs)
    if missing:
        raise RuntimeError(f"S-QER loss is missing outputs {sorted(missing)}")

    attention = outputs["sqer_attention"].float()
    roi_grid = outputs["sqer_roi_grid"].float()
    query_indices = outputs["sqer_query_indices"]
    batch_size, topk, _, roi_tokens = attention.shape
    roi_size = int(round(math.sqrt(roi_tokens)))
    if roi_size * roi_size != roi_tokens:
        raise RuntimeError(f"S-QER ROI token count must be square, got {roi_tokens}")
    device = attention.device
    eps_log = 1e-8
    loss_terms = []
    matched_count = 0
    selected_count = 0
    selected_small = 0
    matched_small = 0
    supervised_count = 0
    sam_box_target_difference = []
    skipped_reasons = {"empty_or_multi": 0, "teacher_rejected": 0, "not_topk": 0, "region_mass": 0}

    coverage_by_image = []
    small_coverage_by_image = []
    for batch_index, (target, (src_idx, tgt_idx)) in enumerate(zip(targets, matched_indices)):
        boxes = target.get("boxes")
        image_matched = 0
        image_selected = 0
        image_small_matched = 0
        image_small_selected = 0
        for source_query, target_index in zip(src_idx.tolist(), tgt_idx.tolist()):
            matched_count += 1
            image_matched += 1
            target_box = boxes[target_index] if boxes is not None and len(boxes) > target_index else None
            is_small = False
            if target_box is not None:
                area_px = float(target_box[2] * target_box[3]) * 512.0 * 640.0
                is_small = area_px < 1024.0
            if is_small:
                matched_small += 1
                image_small_matched += 1
            is_selected = bool((query_indices[batch_index] == int(source_query)).any())
            if is_selected:
                selected_count += 1
                image_selected += 1
                if is_small:
                    selected_small += 1
                    image_small_selected += 1
        if image_matched:
            coverage_by_image.append(image_selected / image_matched)
        if image_small_matched:
            small_coverage_by_image.append(image_small_selected / image_small_matched)

        if boxes is None or boxes.numel() == 0:
            skipped_reasons["empty_or_multi"] += 1
            continue
        teacher_quality = target.get("sam_quality")
        if teacher_quality is None or float(torch.as_tensor(teacher_quality).reshape(-1)[0]) <= 0:
            skipped_reasons["teacher_rejected"] += 1
            continue
        source_mask = target.get("masks")
        if source_mask is None or source_mask.numel() == 0 or len(source_mask) != len(boxes):
            skipped_reasons["teacher_rejected"] += 1
            continue

        for query_index, target_index in zip(src_idx.tolist(), tgt_idx.tolist()):
            box = boxes[target_index].to(device=device, dtype=torch.float32).reshape(1, 1, 4)
            area_px = box[0, 0, 2] * box[0, 0, 3] * 512.0 * 640.0
            is_small = bool(area_px < 1024.0)
            selected_matches = torch.nonzero(
                query_indices[batch_index] == int(query_index), as_tuple=False
            ).flatten()
            if selected_matches.numel() == 0:
                if float(torch.as_tensor(teacher_quality).reshape(-1)[0]) > 0:
                    skipped_reasons["not_topk"] += 1
                continue
            selected_index = int(selected_matches[0])

            mask = source_mask[target_index].to(device=device, dtype=torch.float32)
            grid = roi_grid[batch_index, selected_index].reshape(
                1, roi_size, roi_size, 2
            )
            sam_occupancy = F.grid_sample(
                mask[None, None], grid, mode="bilinear", padding_mode="zeros",
                align_corners=False,
            )[0, 0]
            box_mask = _box_occupancy(
                box,
                int(mask.shape[-2]),
                int(mask.shape[-1]),
            )[0, 0]
            box_occupancy = F.grid_sample(
                box_mask[None, None], grid, mode="bilinear", padding_mode="zeros",
                align_corners=False,
            )[0, 0]
            sam_dist, sam_valid, sam_mass = _region_distributions(
                sam_occupancy, min_region_mass=min_region_mass,
            )
            box_dist, box_valid, box_mass = _region_distributions(
                box_occupancy, min_region_mass=min_region_mass,
            )
            if not bool(sam_valid[0]) or not bool(box_valid[0]):
                skipped_reasons["region_mass"] += 1
                continue
            supervised_count += 1
            sam_box_target_difference.append(
                (sam_dist[0] - box_dist[0]).abs().mean().detach()
            )
            desired = sam_dist[0] if supervision == "sam" else box_dist[0]
            predicted = attention[batch_index, selected_index]
            kl = desired * (
                desired.clamp_min(eps_log).log()
                - predicted.clamp_min(eps_log).log()
            )
            loss_terms.append(kl.sum(dim=-1).mean())

    zero = attention.sum() * 0.0
    loss = torch.stack(loss_terms).mean() if loss_terms else zero
    total_small = max(matched_small, 1)
    total_matched = max(matched_count, 1)
    diagnostics = {
        "sqer_matched_topk_coverage": attention.new_tensor(selected_count / total_matched),
        "sqer_small_topk_coverage": attention.new_tensor(selected_small / total_small),
        "sqer_image_mean_topk_coverage": attention.new_tensor(
            sum(coverage_by_image) / max(len(coverage_by_image), 1)
        ),
        "sqer_image_mean_small_topk_coverage": attention.new_tensor(
            sum(small_coverage_by_image) / max(len(small_coverage_by_image), 1)
        ),
        "sqer_supervised_queries": attention.new_tensor(float(supervised_count)),
        "sqer_teacher_map_difference": (
            torch.stack(sam_box_target_difference).mean()
            if sam_box_target_difference else zero.detach()
        ),
        "sqer_skip_not_topk": attention.new_tensor(float(skipped_reasons["not_topk"])),
        "sqer_skip_region_mass": attention.new_tensor(float(skipped_reasons["region_mass"])),
    }
    diagnostics.update(
        {f"sqer_skip_{name}": attention.new_tensor(float(value))
         for name, value in skipped_reasons.items()}
    )
    return loss, diagnostics
