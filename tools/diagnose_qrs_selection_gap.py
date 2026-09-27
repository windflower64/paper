"""Diagnose whether D-FINE's encoder Top-K query selection drops good boxes.

This is a read-only checkpoint audit.  It deliberately evaluates the existing
encoder score and box heads before any QRS module is implemented, so a new
selection branch is only trained when the current selector has a measurable
quality-ranking gap.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torchvision
from torchvision.ops import box_iou


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        dim=-1,
    )


def normalized_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Normalize validation boxes, whose pipeline keeps absolute XYXY TVTensors."""
    if boxes.numel() == 0:
        return boxes.as_subclass(torch.Tensor) if hasattr(boxes, "as_subclass") else boxes
    plain = boxes.as_subclass(torch.Tensor) if hasattr(boxes, "as_subclass") else boxes
    box_format = getattr(boxes, "format", None)
    if box_format is not None:
        format_name = getattr(box_format, "value", str(box_format)).lower()
        if format_name != "cxcywh":
            plain = torchvision.ops.box_convert(
                plain, in_fmt=format_name, out_fmt="cxcywh"
            )
    canvas_size = getattr(boxes, "canvas_size", None)
    if canvas_size is None:
        canvas_size = getattr(boxes, "spatial_size", None)
    if canvas_size is not None and float(plain.max()) > 1.0:
        height, width = canvas_size
        divisor = plain.new_tensor([width, height, width, height])
        plain = plain / divisor
    return plain


def checkpoint_state(checkpoint: dict) -> dict:
    if "ema" in checkpoint:
        ema = checkpoint["ema"]
        return ema["module"] if isinstance(ema, dict) and "module" in ema else ema
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--coarse-topk", type=int, default=900)
    args = parser.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from src.core import YAMLConfig

    config = YAMLConfig(str(args.config.resolve()))
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = config.model
    checkpoint = torch.load(
        args.checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    incompatible = model.load_state_dict(checkpoint_state(checkpoint), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint/config mismatch: "
            f"missing={incompatible.missing_keys[:10]}, "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )

    device = torch.device(args.device)
    model.to(device).eval()
    decoder = model.decoder
    loader = config.val_dataloader

    cutoffs = sorted({10, 30, 100, decoder.num_queries, int(args.coarse_topk)})
    best_iou_by_cutoff: dict[int, list[float]] = {cutoff: [] for cutoff in cutoffs}
    hit_50_by_cutoff: dict[int, int] = {cutoff: 0 for cutoff in cutoffs}
    hit_75_by_cutoff: dict[int, int] = {cutoff: 0 for cutoff in cutoffs}
    best_dense_ious: list[float] = []
    oracle_score_ranks: list[float] = []
    score_iou_correlations: list[float] = []
    selected_level_counts = [0 for _ in decoder.feat_strides]
    target_images = 0
    empty_images = 0
    processed_images = 0

    with torch.inference_mode():
        for samples, targets in loader:
            if args.max_images and processed_images >= args.max_images:
                break
            samples = samples.to(device)
            targets = [
                {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in target.items()
                }
                for target in targets
            ]

            features = model.encoder(model.backbone(samples))
            memory, spatial_shapes = decoder._get_encoder_input(features)
            anchors, valid_mask = decoder._generate_anchors(
                spatial_shapes, device=memory.device
            )
            if memory.shape[0] > 1:
                anchors = anchors.repeat(memory.shape[0], 1, 1)
            output_memory = decoder.enc_output(valid_mask.to(memory.dtype) * memory)
            logits = decoder.enc_score_head(output_memory)
            scores = logits.sigmoid().amax(dim=-1)
            dense_boxes = (
                decoder.enc_bbox_head(output_memory) + anchors
            ).sigmoid()

            level_ends = []
            total = 0
            for height, width in spatial_shapes:
                total += int(height * width)
                level_ends.append(total)

            batch_size = samples.shape[0]
            for batch_index in range(batch_size):
                if args.max_images and processed_images >= args.max_images:
                    break
                processed_images += 1
                gt_boxes = normalized_cxcywh(targets[batch_index]["boxes"])
                if gt_boxes.numel() == 0:
                    empty_images += 1
                    continue
                target_images += 1

                ious = box_iou(
                    cxcywh_to_xyxy(dense_boxes[batch_index]),
                    cxcywh_to_xyxy(gt_boxes),
                ).amax(dim=1)
                best_dense_ious.append(float(ious.max().cpu()))
                score_order = scores[batch_index].argsort(descending=True)
                oracle_index = int(ious.argmax())
                oracle_rank = int((score_order == oracle_index).nonzero()[0]) + 1
                oracle_score_ranks.append(float(oracle_rank))

                correlation_count = min(int(args.coarse_topk), score_order.numel())
                correlation_indices = score_order[:correlation_count]
                score_values = scores[batch_index, correlation_indices].float()
                iou_values = ious[correlation_indices].float()
                if score_values.std() > 0 and iou_values.std() > 0:
                    correlation = torch.corrcoef(
                        torch.stack((score_values, iou_values))
                    )[0, 1]
                    if torch.isfinite(correlation):
                        score_iou_correlations.append(float(correlation.cpu()))

                selected = score_order[: decoder.num_queries]
                for index in selected.tolist():
                    level = next(
                        level_index
                        for level_index, level_end in enumerate(level_ends)
                        if index < level_end
                    )
                    selected_level_counts[level] += 1

                for cutoff in cutoffs:
                    actual_cutoff = min(cutoff, score_order.numel())
                    best = float(ious[score_order[:actual_cutoff]].max().cpu())
                    best_iou_by_cutoff[cutoff].append(best)
                    hit_50_by_cutoff[cutoff] += int(best >= 0.50)
                    hit_75_by_cutoff[cutoff] += int(best >= 0.75)

    if target_images == 0:
        raise RuntimeError("no target images were evaluated")

    cutoff_summary = {}
    for cutoff in cutoffs:
        cutoff_summary[str(cutoff)] = {
            "mean_best_encoder_iou": safe_mean(best_iou_by_cutoff[cutoff]),
            "recall_best_iou_ge_050": hit_50_by_cutoff[cutoff] / target_images,
            "recall_best_iou_ge_075": hit_75_by_cutoff[cutoff] / target_images,
        }

    selected_total = sum(selected_level_counts)
    result = {
        "status": "PASS",
        "purpose": "measure_quality_ranking_gap_before_implementing_QRS",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "processed_images": processed_images,
        "target_images": target_images,
        "empty_images": empty_images,
        "num_queries": decoder.num_queries,
        "encoder_tokens": sum(int(h * w) for h, w in spatial_shapes),
        "cutoffs": cutoff_summary,
        "dense_oracle_mean_iou": safe_mean(best_dense_ious),
        "oracle_box_score_rank": {
            "mean": safe_mean(oracle_score_ranks),
            "median": percentile(oracle_score_ranks, 0.50),
            "p90": percentile(oracle_score_ranks, 0.90),
            "within_top300_ratio": sum(
                rank <= decoder.num_queries for rank in oracle_score_ranks
            )
            / target_images,
        },
        "score_iou_pearson_top_coarse_mean": safe_mean(score_iou_correlations),
        "selected_level_distribution": [
            {
                "stride": int(stride),
                "count": count,
                "ratio": count / max(selected_total, 1),
            }
            for stride, count in zip(decoder.feat_strides, selected_level_counts)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
