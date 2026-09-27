#!/usr/bin/env python3
"""Calibrate and evaluate M-A2 explicit object-level RGB-T matching.

The detector weights and boxes remain frozen.  At most one Visible query per
image receives a positive logit correction from the top Thermal proposal, and
only when the two proposals pass explicit confidence and geometry checks.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import torch

from analyze_ma2_detector_cache import IR_TO_VISIBLE, cxcywh_to_xyxy, map_ir_xyxy


METRIC_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--cache", type=Path, required=True)
    calibrate.add_argument("--annotations", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--sequence-modulus", type=int, default=5)
    calibrate.add_argument("--sequence-residue", type=int, default=0)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--cache", type=Path, required=True)
    evaluate.add_argument("--annotations", type=Path, required=True)
    evaluate.add_argument("--calibration", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--mismatch-offset", type=int, default=607)
    return parser.parse_args()


def sequence_name(file_name):
    return Path(file_name).stem.rsplit("_", 1)[0]


def stable_bucket(value, modulus):
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def build_evaluator(annotation_path):
    from faster_coco_eval import COCO

    return COCO(str(annotation_path), print_function=lambda *_args, **_kwargs: None)


def evaluate_scores(coco_gt, cache, visible_xyxy, scores, indices):
    from faster_coco_eval import COCOeval_faster

    detections = []
    selected_image_ids = []
    for index in indices:
        image_id = int(cache["image_ids"][index])
        selected_image_ids.append(image_id)
        width, height = cache["orig_sizes"][index].tolist()
        image_scores = scores[index]
        keep = image_scores.topk(min(100, image_scores.numel())).indices
        boxes = visible_xyxy[index, keep].clone()
        boxes[:, 0::2] *= width
        boxes[:, 1::2] *= height
        boxes[:, 2:] -= boxes[:, :2]
        for box, score in zip(boxes.tolist(), image_scores[keep].tolist()):
            if box[2] <= 0 or box[3] <= 0 or not math.isfinite(score):
                continue
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": box,
                    "score": float(score),
                }
            )
    coco_dt = coco_gt.loadRes(detections, min_score=0.0)
    evaluator = COCOeval_faster(
        coco_gt, coco_dt, "bbox", print_function=lambda *_args, **_kwargs: None
    )
    evaluator.params.imgIds = selected_image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    result = {name: float(value) for name, value in zip(METRIC_NAMES, evaluator.stats)}
    del evaluator, coco_dt, detections
    gc.collect()
    return result


def fuse(
    cache,
    parameters,
    mapping="affine",
    thermal_order=None,
    thermal_offsets=None,
):
    visible_logits = cache["visible_logits"].squeeze(-1).float()
    visible_scores = visible_logits.sigmoid()
    visible_boxes = cxcywh_to_xyxy(cache["visible_boxes"].float()).clamp(0.0, 1.0)
    thermal_logits = cache["thermal_logits"].squeeze(-1).float()
    thermal_boxes = cxcywh_to_xyxy(cache["thermal_boxes"].float())
    if thermal_order is not None:
        thermal_logits = thermal_logits[thermal_order]
        thermal_boxes = thermal_boxes[thermal_order]
    thermal_scores = thermal_logits.sigmoid()
    if mapping == "affine":
        thermal_boxes = map_ir_xyxy(thermal_boxes, IR_TO_VISIBLE)
    elif mapping == "identity":
        thermal_boxes = thermal_boxes.clamp(0.0, 1.0)
    else:
        raise ValueError(mapping)
    if thermal_offsets is not None:
        if thermal_offsets.shape != (thermal_boxes.shape[0], 2):
            raise ValueError(
                "thermal_offsets must have shape [samples, 2], got "
                f"{tuple(thermal_offsets.shape)}"
            )
        thermal_boxes = thermal_boxes + thermal_offsets[:, None, :].repeat(1, 1, 2)
        thermal_boxes = thermal_boxes.clamp(0.0, 1.0)

    thermal_top_score, thermal_top_index = thermal_scores.max(dim=1)
    batch_index = torch.arange(visible_scores.shape[0])
    thermal_top_box = thermal_boxes[batch_index, thermal_top_index]
    thermal_center = (thermal_top_box[:, :2] + thermal_top_box[:, 2:]) / 2

    visible_topk = min(int(parameters["visible_topk"]), visible_scores.shape[1])
    candidate_scores, candidate_indices = visible_scores.topk(visible_topk, dim=1)
    candidate_boxes = visible_boxes.gather(
        1, candidate_indices[..., None].expand(-1, -1, 4)
    )
    candidate_centers = (candidate_boxes[..., :2] + candidate_boxes[..., 2:]) / 2
    distances = torch.linalg.vector_norm(
        candidate_centers - thermal_center[:, None, :], dim=-1
    )

    radius = float(parameters["radius"])
    sigma = radius / 2.0
    geometry = torch.exp(-0.5 * (distances / sigma) ** 2)
    match_quality = geometry * candidate_scores.clamp_min(1e-8).pow(
        float(parameters["visible_score_power"])
    )
    best_local = match_quality.argmax(dim=1)
    matched_query = candidate_indices[batch_index, best_local]
    matched_distance = distances[batch_index, best_local]
    matched_geometry = geometry[batch_index, best_local]
    matched_visible_score = visible_scores[batch_index, matched_query]
    reliable = (thermal_top_score >= float(parameters["thermal_threshold"])) & (
        matched_distance <= radius
    )

    evidence = thermal_top_score * matched_geometry
    uncertainty = (1.0 - matched_visible_score).pow(
        float(parameters["uncertainty_power"])
    )
    delta = float(parameters["alpha"]) * evidence * uncertainty
    delta = delta * reliable.float()
    fused_logits = visible_logits.clone()
    fused_logits[batch_index, matched_query] += delta
    fused_scores = fused_logits.sigmoid()
    diagnostics = {
        "samples": int(visible_scores.shape[0]),
        "matched_samples": int(reliable.sum()),
        "matched_fraction": float(reliable.float().mean()),
        "thermal_top_score_mean": float(thermal_top_score.mean()),
        "matched_distance_mean_all": float(matched_distance.mean()),
        "matched_distance_mean_reliable": float(matched_distance[reliable].mean())
        if reliable.any()
        else None,
        "matched_visible_score_mean_reliable": float(matched_visible_score[reliable].mean())
        if reliable.any()
        else None,
        "logit_delta_mean_reliable": float(delta[reliable].mean())
        if reliable.any()
        else None,
        "logit_delta_max": float(delta.max()),
        "changed_query_count": int((delta != 0).sum()),
    }
    return fused_scores, visible_scores, visible_boxes, diagnostics


def parameter_key(parameters):
    return tuple(
        parameters[key]
        for key in (
            "visible_topk",
            "radius",
            "thermal_threshold",
            "visible_score_power",
            "uncertainty_power",
            "alpha",
        )
    )


def calibration_rank(item):
    p = item["parameters"]
    # AP is authoritative; remaining terms only make exact ties conservative.
    return (
        item["metrics"]["AP"],
        item["metrics"]["AP50"],
        p["thermal_threshold"],
        -p["radius"],
        -p["visible_topk"],
        -p["alpha"],
    )


def calibrate(args):
    if args.sequence_modulus <= 1:
        raise ValueError("sequence-modulus must exceed one")
    if not 0 <= args.sequence_residue < args.sequence_modulus:
        raise ValueError("invalid sequence-residue")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    selected = [
        index
        for index, file_name in enumerate(cache["file_names"])
        if stable_bucket(sequence_name(file_name), args.sequence_modulus)
        == args.sequence_residue
    ]
    if len(selected) < 100:
        raise RuntimeError(f"Calibration subset unexpectedly small: {len(selected)}")
    coco_gt = build_evaluator(args.annotations)
    visible_xyxy = cxcywh_to_xyxy(cache["visible_boxes"].float()).clamp(0.0, 1.0)
    baseline_scores = cache["visible_logits"].squeeze(-1).float().sigmoid()
    baseline_metrics = evaluate_scores(
        coco_gt, cache, visible_xyxy, baseline_scores, selected
    )
    print(f"baseline AP={baseline_metrics['AP']:.9f}, images={len(selected)}", flush=True)

    stage1 = []
    for visible_topk in (3, 10, 30):
        for radius in (0.03, 0.06, 0.10):
            for thermal_threshold in (0.60, 0.75):
                for visible_score_power in (0.5, 1.0):
                    parameters = {
                        "visible_topk": visible_topk,
                        "radius": radius,
                        "thermal_threshold": thermal_threshold,
                        "visible_score_power": visible_score_power,
                        "uncertainty_power": 0.0,
                        "alpha": 0.30,
                    }
                    scores, _base, _boxes, diagnostics = fuse(cache, parameters)
                    metrics = evaluate_scores(
                        coco_gt, cache, visible_xyxy, scores, selected
                    )
                    stage1.append(
                        {
                            "parameters": parameters,
                            "metrics": metrics,
                            "diagnostics": diagnostics,
                        }
                    )
                    print(
                        f"stage1 {len(stage1):02d}/36 AP={metrics['AP']:.9f} "
                        f"params={parameter_key(parameters)}",
                        flush=True,
                    )

    stage1.sort(key=calibration_rank, reverse=True)
    matcher_candidates = []
    seen_matchers = set()
    for item in stage1:
        p = item["parameters"]
        key = (
            p["visible_topk"],
            p["radius"],
            p["thermal_threshold"],
            p["visible_score_power"],
        )
        if key not in seen_matchers:
            matcher_candidates.append(p)
            seen_matchers.add(key)
        if len(matcher_candidates) == 3:
            break

    stage2 = []
    for matcher in matcher_candidates:
        for uncertainty_power in (0.0, 1.0):
            for alpha in (0.05, 0.10, 0.20, 0.40, 0.80):
                parameters = {
                    **matcher,
                    "uncertainty_power": uncertainty_power,
                    "alpha": alpha,
                }
                scores, _base, _boxes, diagnostics = fuse(cache, parameters)
                metrics = evaluate_scores(coco_gt, cache, visible_xyxy, scores, selected)
                stage2.append(
                    {
                        "parameters": parameters,
                        "metrics": metrics,
                        "diagnostics": diagnostics,
                    }
                )
                print(
                    f"stage2 {len(stage2):02d}/30 AP={metrics['AP']:.9f} "
                    f"params={parameter_key(parameters)}",
                    flush=True,
                )

    stage2.sort(key=calibration_rank, reverse=True)
    chosen = stage2[0]
    report = {
        "schema": "ma2_explicit_match_calibration_v1",
        "cache": str(args.cache.resolve()),
        "annotations": str(args.annotations.resolve()),
        "selection_protocol": {
            "data": "train sequence-grouped deterministic subset",
            "sequence_modulus": args.sequence_modulus,
            "sequence_residue": args.sequence_residue,
            "selected_images": len(selected),
            "selected_sequences": len(
                {sequence_name(cache["file_names"][index]) for index in selected}
            ),
            "test_metrics_used_for_selection": False,
            "stage1_fixed_alpha": 0.30,
            "stage1_fixed_uncertainty_power": 0.0,
            "stage1_candidates": len(stage1),
            "stage2_matcher_candidates": len(matcher_candidates),
            "stage2_candidates": len(stage2),
        },
        "baseline_metrics": baseline_metrics,
        "chosen": chosen,
        "chosen_delta_AP": chosen["metrics"]["AP"] - baseline_metrics["AP"],
        "stage1_top5": stage1[:5],
        "stage2_all": stage2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"baseline": baseline_metrics, "chosen": chosen}, ensure_ascii=False, indent=2))


def evaluate(args):
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    parameters = calibration["chosen"]["parameters"]
    coco_gt = build_evaluator(args.annotations)
    indices = list(range(int(cache["image_ids"].shape[0])))

    normal, baseline, visible_xyxy, normal_diagnostics = fuse(cache, parameters)
    mismatch_order = torch.arange(len(indices)).roll(args.mismatch_offset)
    mismatch, _base, _boxes, mismatch_diagnostics = fuse(
        cache, parameters, mapping="affine", thermal_order=mismatch_order
    )
    identity, _base, _boxes, identity_diagnostics = fuse(
        cache, parameters, mapping="identity"
    )
    conditions = {
        "visible_baseline_no_ir": {
            "metrics": evaluate_scores(coco_gt, cache, visible_xyxy, baseline, indices),
            "diagnostics": {
                "matched_samples": 0,
                "changed_query_count": 0,
            },
        },
        "normal_affine_paired": {
            "metrics": evaluate_scores(coco_gt, cache, visible_xyxy, normal, indices),
            "diagnostics": normal_diagnostics,
        },
        "global_mismatch": {
            "metrics": evaluate_scores(coco_gt, cache, visible_xyxy, mismatch, indices),
            "diagnostics": mismatch_diagnostics,
        },
        "identity_mapping": {
            "metrics": evaluate_scores(coco_gt, cache, visible_xyxy, identity, indices),
            "diagnostics": identity_diagnostics,
        },
    }
    baseline_ap = conditions["visible_baseline_no_ir"]["metrics"]["AP"]
    normal_ap = conditions["normal_affine_paired"]["metrics"]["AP"]
    mismatch_ap = conditions["global_mismatch"]["metrics"]["AP"]
    identity_ap = conditions["identity_mapping"]["metrics"]["AP"]
    sequence_folds = []
    for fold in range(5):
        fold_indices = [
            index
            for index, file_name in enumerate(cache["file_names"])
            if stable_bucket(sequence_name(file_name), 5) == fold
        ]
        fold_baseline = evaluate_scores(
            coco_gt, cache, visible_xyxy, baseline, fold_indices
        )
        fold_normal = evaluate_scores(
            coco_gt, cache, visible_xyxy, normal, fold_indices
        )
        sequence_folds.append(
            {
                "fold": fold,
                "images": len(fold_indices),
                "sequences": len(
                    {sequence_name(cache["file_names"][index]) for index in fold_indices}
                ),
                "baseline_AP": fold_baseline["AP"],
                "normal_AP": fold_normal["AP"],
                "delta_AP": fold_normal["AP"] - fold_baseline["AP"],
            }
        )
    report = {
        "schema": "ma2_explicit_match_test_v1",
        "cache": str(args.cache.resolve()),
        "annotations": str(args.annotations.resolve()),
        "calibration": str(args.calibration.resolve()),
        "parameters_frozen_before_test": parameters,
        "mismatch_offset": args.mismatch_offset,
        "conditions": conditions,
        "contrasts": {
            "normal_minus_baseline_AP": normal_ap - baseline_ap,
            "normal_minus_mismatch_AP": normal_ap - mismatch_ap,
            "normal_minus_identity_AP": normal_ap - identity_ap,
        },
        "fixed_parameter_sequence_fold_robustness": {
            "protocol": "Five deterministic sequence-grouped folds; parameters remain frozen.",
            "folds": sequence_folds,
            "positive_folds": sum(item["delta_AP"] > 0 for item in sequence_folds),
            "negative_folds": sum(item["delta_AP"] < 0 for item in sequence_folds),
            "mean_fold_delta_AP": sum(item["delta_AP"] for item in sequence_folds)
            / len(sequence_folds),
        },
        "decision": {
            "beats_visible_baseline": bool(normal_ap > baseline_ap),
            "uses_correct_pairing": bool(normal_ap > mismatch_ap),
            "uses_train_fitted_mapping": bool(normal_ap > identity_ap),
            "rule": "M-A2 continues only if normal paired AP beats the frozen Visible baseline.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    if args.command == "calibrate":
        calibrate(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
