#!/usr/bin/env python3
"""M-A3: label-free, past-frame-only sequence geometry self-calibration.

This is an upper-bound audit, not a new trainable module.  Each sequence uses
only its earliest frames and only detector predictions to estimate a robust
translation from mapped Thermal proposals to RGB proposals.  The estimate is
then applied only to later frames of that sequence.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from analyze_ma2_detector_cache import cxcywh_to_xyxy, map_ir_xyxy
from run_ma2_explicit_matching import build_evaluator, evaluate_scores, fuse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--min-anchors", type=int, default=2)
    parser.add_argument("--visible-anchor-score", type=float, default=0.75)
    parser.add_argument("--thermal-anchor-score", type=float, default=0.75)
    parser.add_argument("--anchor-radius", type=float, default=0.10)
    parser.add_argument("--max-offset", type=float, default=0.10)
    parser.add_argument("--mismatch-offset", type=int, default=607)
    return parser.parse_args()


def parse_sequence(file_name):
    stem = Path(file_name).stem
    sequence, frame = stem.rsplit("_", 1)
    return sequence, int(frame)


def sequence_indices(cache):
    grouped = defaultdict(list)
    for index, file_name in enumerate(cache["file_names"]):
        sequence, frame = parse_sequence(file_name)
        grouped[sequence].append((frame, index))
    return {
        sequence: [index for _frame, index in sorted(frames)]
        for sequence, frames in grouped.items()
    }


def build_offsets(cache, args, thermal_order=None):
    visible_scores = cache["visible_logits"].squeeze(-1).float().sigmoid()
    thermal_scores = cache["thermal_logits"].squeeze(-1).float().sigmoid()
    thermal_boxes = cache["thermal_boxes"].float()
    if thermal_order is not None:
        thermal_scores = thermal_scores[thermal_order]
        thermal_boxes = thermal_boxes[thermal_order]
    visible_boxes = cxcywh_to_xyxy(cache["visible_boxes"].float()).clamp(0.0, 1.0)
    thermal_boxes = map_ir_xyxy(cxcywh_to_xyxy(thermal_boxes))
    batch = torch.arange(visible_scores.shape[0])
    visible_top_score, visible_top = visible_scores.max(dim=1)
    thermal_top_score, thermal_top = thermal_scores.max(dim=1)
    visible_center = (
        visible_boxes[batch, visible_top, :2] + visible_boxes[batch, visible_top, 2:]
    ) / 2
    thermal_center = (
        thermal_boxes[batch, thermal_top, :2] + thermal_boxes[batch, thermal_top, 2:]
    ) / 2
    residual = visible_center - thermal_center
    distance = torch.linalg.vector_norm(residual, dim=1)
    anchors = (
        (visible_top_score >= args.visible_anchor_score)
        & (thermal_top_score >= args.thermal_anchor_score)
        & (distance <= args.anchor_radius)
    )

    offsets = torch.zeros_like(residual)
    adapted = torch.zeros(len(residual), dtype=torch.bool)
    per_sequence = []
    for sequence, indices in sequence_indices(cache).items():
        warmup = indices[: args.warmup_frames]
        future = indices[args.warmup_frames :]
        anchor_indices = [index for index in warmup if bool(anchors[index])]
        if len(anchor_indices) < args.min_anchors:
            per_sequence.append(
                {
                    "sequence": sequence,
                    "frames": len(indices),
                    "warmup_anchors": len(anchor_indices),
                    "adapted_frames": 0,
                    "offset": None,
                }
            )
            continue
        offset = residual[anchor_indices].median(dim=0).values
        norm = torch.linalg.vector_norm(offset)
        if norm > args.max_offset:
            offset = offset * (args.max_offset / norm)
        offsets[future] = offset
        adapted[future] = True
        per_sequence.append(
            {
                "sequence": sequence,
                "frames": len(indices),
                "warmup_anchors": len(anchor_indices),
                "adapted_frames": len(future),
                "offset": [float(value) for value in offset],
            }
        )

    metadata = {
        "sequences": len(per_sequence),
        "adapted_sequences": sum(item["offset"] is not None for item in per_sequence),
        "adapted_frames": int(adapted.sum()),
        "adapted_frame_fraction": float(adapted.float().mean()),
        "candidate_anchor_frames": int(anchors.sum()),
        "candidate_anchor_fraction": float(anchors.float().mean()),
        "mean_anchor_distance": float(distance[anchors].mean()) if anchors.any() else None,
        "mean_offset_norm_adapted_sequences": float(
            torch.tensor(
                [
                    sum(value * value for value in item["offset"]) ** 0.5
                    for item in per_sequence
                    if item["offset"] is not None
                ]
            ).mean()
        )
        if any(item["offset"] is not None for item in per_sequence)
        else None,
        "per_sequence": per_sequence,
    }
    return offsets, metadata


def main():
    args = parse_args()
    if args.warmup_frames < 1 or args.min_anchors < 1:
        raise ValueError("warmup-frames and min-anchors must be positive")
    if args.min_anchors > args.warmup_frames:
        raise ValueError("min-anchors cannot exceed warmup-frames")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    parameters = calibration["chosen"]["parameters"]
    indices = list(range(len(cache["image_ids"])))
    coco_gt = build_evaluator(args.annotations)
    visible_xyxy = cxcywh_to_xyxy(cache["visible_boxes"].float()).clamp(0.0, 1.0)

    baseline = cache["visible_logits"].squeeze(-1).float().sigmoid()
    ma2_scores, _base, _boxes, ma2_diagnostics = fuse(cache, parameters)
    normal_offsets, normal_adaptation = build_offsets(cache, args)
    ma3_scores, _base, _boxes, ma3_diagnostics = fuse(
        cache, parameters, thermal_offsets=normal_offsets
    )

    mismatch_order = torch.arange(len(indices)).roll(args.mismatch_offset)
    mismatch_offsets, mismatch_adaptation = build_offsets(
        cache, args, thermal_order=mismatch_order
    )
    mismatch_scores, _base, _boxes, mismatch_diagnostics = fuse(
        cache,
        parameters,
        thermal_order=mismatch_order,
        thermal_offsets=mismatch_offsets,
    )
    no_adaptation_scores, _base, _boxes, no_adaptation_diagnostics = fuse(
        cache, parameters, thermal_offsets=torch.zeros_like(normal_offsets)
    )

    conditions = {
        "visible_baseline": {
            "metrics": evaluate_scores(coco_gt, cache, visible_xyxy, baseline, indices),
            "diagnostics": {"changed_query_count": 0},
        },
        "ma2_global_affine": {
            "metrics": evaluate_scores(coco_gt, cache, visible_xyxy, ma2_scores, indices),
            "diagnostics": ma2_diagnostics,
        },
        "ma3_past_frame_self_calibrated": {
            "metrics": evaluate_scores(coco_gt, cache, visible_xyxy, ma3_scores, indices),
            "diagnostics": ma3_diagnostics,
        },
        "ma3_global_mismatch": {
            "metrics": evaluate_scores(coco_gt, cache, visible_xyxy, mismatch_scores, indices),
            "diagnostics": mismatch_diagnostics,
        },
        "ma3_zero_offset_control": {
            "metrics": evaluate_scores(coco_gt, cache, visible_xyxy, no_adaptation_scores, indices),
            "diagnostics": no_adaptation_diagnostics,
        },
    }
    report = {
        "schema": "ma3_sequence_self_calibration_audit_v1",
        "purpose": "No-label, past-frame-only geometry adaptation upper bound.",
        "cache": str(args.cache.resolve()),
        "annotations": str(args.annotations.resolve()),
        "parameters_frozen_from_ma2_train_selection": parameters,
        "adaptation_protocol": {
            "warmup_frames": args.warmup_frames,
            "min_anchors": args.min_anchors,
            "visible_anchor_score": args.visible_anchor_score,
            "thermal_anchor_score": args.thermal_anchor_score,
            "anchor_radius": args.anchor_radius,
            "max_offset": args.max_offset,
            "uses_labels": False,
            "uses_future_frames_for_each_adapted_frame": False,
        },
        "normal_adaptation": normal_adaptation,
        "mismatch_adaptation": mismatch_adaptation,
        "conditions": conditions,
        "contrasts": {
            "ma3_minus_baseline_AP": conditions["ma3_past_frame_self_calibrated"]["metrics"]["AP"]
            - conditions["visible_baseline"]["metrics"]["AP"],
            "ma3_minus_ma2_global_AP": conditions["ma3_past_frame_self_calibrated"]["metrics"]["AP"]
            - conditions["ma2_global_affine"]["metrics"]["AP"],
            "ma3_minus_mismatch_AP": conditions["ma3_past_frame_self_calibrated"]["metrics"]["AP"]
            - conditions["ma3_global_mismatch"]["metrics"]["AP"],
            "ma3_minus_zero_offset_AP": conditions["ma3_past_frame_self_calibrated"]["metrics"]["AP"]
            - conditions["ma3_zero_offset_control"]["metrics"]["AP"],
            "ma3_match_fraction_minus_ma2": ma3_diagnostics["matched_fraction"]
            - ma2_diagnostics["matched_fraction"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
