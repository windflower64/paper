#!/usr/bin/env python3
"""固定SCBV0 best，诊断AP75上涨但总AP下降的来源。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.core import YAMLConfig
from s_scbv import SemanticConditionedBoundaryVolume, box_cxcywh_to_xyxy
from train_s_hrbr1_refinebox import (
    BackboneFeatureTap,
    collect_detections,
    load_frozen_detector,
    matched_training_boxes,
    move_targets,
    regression_loss,
)
from train_s_qbdm0_frozen_detail_reader import save_json
from train_s_scbv0_frozen_boundary_volume import validation_targets_for_matcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT.parent / "outputs/S_SCBV0_ALIGNED_FROZEN_A00_SEED0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent
        / "reports/26_semantic_boundary_volume/S_SCBV0/failure_diagnosis.json",
    )
    parser.add_argument("--checkpoint-name", default="best.pth")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--gradient-batches", type=int, default=12)
    return parser.parse_args()


def aligned_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    intersection_lt = torch.maximum(left[:, :2], right[:, :2])
    intersection_rb = torch.minimum(left[:, 2:], right[:, 2:])
    intersection_wh = (intersection_rb - intersection_lt).clamp_min(0)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    left_wh = (left[:, 2:] - left[:, :2]).clamp_min(0)
    right_wh = (right[:, 2:] - right[:, :2]).clamp_min(0)
    left_area = left_wh[:, 0] * left_wh[:, 1]
    right_area = right_wh[:, 0] * right_wh[:, 1]
    union = left_area + right_area - intersection
    return intersection / union.clamp_min(1e-12)


def ap_by_iou(coco_gt, detections) -> dict[str, float]:
    from faster_coco_eval import COCOeval_faster

    thresholds = np.linspace(0.50, 0.95, 10)
    evaluator = COCOeval_faster(coco_gt, coco_gt.loadRes(detections), "bbox")
    evaluator.params.iouThrs = thresholds
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]
    result = {}
    for index, threshold in enumerate(thresholds):
        values = precision[index, :, :, 0, -1]
        valid = values[values > -1]
        result[f"AP@{threshold:.2f}"] = float(valid.mean()) if valid.size else None
    return result


def gradient_pair_statistics(box_loss, bin_loss, parameters) -> dict[str, float]:
    box_gradients = torch.autograd.grad(
        box_loss, parameters, retain_graph=True, allow_unused=True
    )
    bin_gradients = torch.autograd.grad(
        bin_loss, parameters, retain_graph=False, allow_unused=True
    )
    dot = box_loss.new_zeros(())
    box_square = box_loss.new_zeros(())
    bin_square = box_loss.new_zeros(())
    for box_gradient, bin_gradient in zip(box_gradients, bin_gradients):
        if box_gradient is None or bin_gradient is None:
            continue
        box_gradient = box_gradient.float()
        bin_gradient = bin_gradient.float()
        dot = dot + (box_gradient * bin_gradient).sum()
        box_square = box_square + box_gradient.square().sum()
        bin_square = bin_square + bin_gradient.square().sum()
    box_norm = box_square.sqrt()
    bin_norm = bin_square.sqrt()
    cosine = dot / (box_norm * bin_norm).clamp_min(1e-12)
    return {
        "cosine": float(cosine),
        "box_gradient_l2": float(box_norm),
        "bin_gradient_l2": float(bin_norm),
        "weighted_bin_to_box_gradient_ratio": float(0.5 * bin_norm / box_norm.clamp_min(1e-12)),
    }


def add_stratum(storage, name: str, baseline: torch.Tensor, delta: torch.Tensor) -> None:
    masks = {
        "lt_0.50": baseline < 0.50,
        "0.50_to_0.75": (baseline >= 0.50) & (baseline < 0.75),
        "ge_0.75": baseline >= 0.75,
    }
    for label, mask in masks.items():
        count = int(mask.sum())
        if count:
            storage[name][label]["count"] += count
            storage[name][label]["delta_sum"] += float(delta[mask].sum())
            storage[name][label]["improved"] += int((delta[mask] > 0).sum())
            storage[name][label]["worsened"] += int((delta[mask] < 0).sum())


def finalize_strata(storage) -> dict:
    result = {}
    for name, groups in storage.items():
        result[name] = {}
        for label, values in groups.items():
            count = values["count"]
            result[name][label] = {
                **values,
                "mean_iou_delta": values["delta_sum"] / max(1, count),
                "improved_fraction": values["improved"] / max(1, count),
                "worsened_fraction": values["worsened"] / max(1, count),
            }
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SCBV0失败诊断需要CUDA")
    summary_path = args.run_dir / "summary.json"
    checkpoint_path = args.run_dir / args.checkpoint_name
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("SCBV0正式summary或指定权重不存在")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = summary["metadata"]
    cfg = YAMLConfig(metadata["config"])
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    detector, weight_source = load_frozen_detector(cfg, Path(metadata["checkpoint"]))
    matcher = cfg.criterion.cuda().eval().matcher
    loader = cfg.val_dataloader
    postprocessor = cfg.postprocessor
    tap = BackboneFeatureTap(detector.backbone, stage_indices=(1, 2))
    reader = SemanticConditionedBoundaryVolume(
        s8_channels=int(detector.backbone._out_channels[1]),
        s16_channels=int(detector.backbone._out_channels[2]),
        detail_channels=int(metadata["detail_channels"]),
        tangent_points=int(metadata["tangent_points"]),
        offset_bins=int(metadata["offset_bins"]),
        max_relative_offset=float(metadata["max_relative_offset"]),
    ).cuda().eval()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reader.load_state_dict(payload["reader"], strict=True)
    parameters = tuple(reader.parameters())

    coco = loader.dataset.coco
    category_ids = sorted(coco.getCatIds())
    detections = {"baseline": [], "aligned": []}
    aggregate = defaultdict(float)
    strata = defaultdict(
        lambda: defaultdict(
            lambda: {"count": 0, "delta_sum": 0.0, "improved": 0, "worsened": 0}
        )
    )
    gradient_records = []

    for batch_index, (samples, targets) in enumerate(loader):
        samples = samples.cuda(non_blocking=True)
        targets_cuda = move_targets(targets, "cuda")
        tap.clear()
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.float16, enabled=args.precision == "fp16"
        ):
            outputs = detector(samples)
        s8 = tap.outputs[1].detach().float()
        s16 = tap.outputs[2].detach().float()
        logits = outputs["pred_logits"].float()
        detector_boxes = outputs["pred_boxes"].float()
        scores = logits.sigmoid().max(-1).values
        selected = scores.topk(min(args.topk, scores.shape[1]), dim=1).indices
        image_ids = torch.arange(samples.shape[0], device=samples.device)[:, None]
        selected_boxes = detector_boxes[image_ids, selected]
        flat_boxes = selected_boxes.flatten(0, 1)
        flat_batch = image_ids.expand_as(selected).flatten()
        with torch.no_grad():
            refined_topk = reader(s8, s16, flat_boxes, flat_batch, "aligned")
        adjusted = detector_boxes.clone()
        adjusted[image_ids, selected] = refined_topk.view_as(selected_boxes)
        sizes = torch.stack([target["orig_size"] for target in targets_cuda])
        baseline_results = postprocessor(
            {"pred_logits": logits, "pred_boxes": detector_boxes}, sizes
        )
        aligned_results = postprocessor(
            {"pred_logits": logits, "pred_boxes": adjusted}, sizes
        )
        collect_detections(detections["baseline"], targets_cuda, baseline_results, category_ids)
        collect_detections(detections["aligned"], targets_cuda, aligned_results, category_ids)

        matcher_targets = validation_targets_for_matcher(targets_cuda, samples)
        predicted, truth, batch_ids = matched_training_boxes(
            outputs, matcher_targets, matcher
        )
        if predicted is None:
            continue
        with torch.no_grad():
            refined = reader(s8, s16, predicted.float(), batch_ids, "aligned")
            predicted_offsets = reader.last_expected_offsets.clone()
            target_offsets, target_bins = reader.target_offsets(predicted.float(), truth.float())
            volume_logits = reader.last_logits
            probabilities = volume_logits.softmax(dim=-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
            top_bins = volume_logits.topk(2, dim=-1).indices
            exact = volume_logits.argmax(-1) == target_bins
            top2 = (top_bins == target_bins[..., None]).any(-1)
            within_one = (volume_logits.argmax(-1) - target_bins).abs() <= 1
            active = target_offsets.abs() >= (
                float(reader.offset_values[1] - reader.offset_values[0]) * 0.5
            )
            sign_correct = predicted_offsets.sign() == target_offsets.sign()
            base_iou = aligned_iou(
                box_cxcywh_to_xyxy(predicted.float()), box_cxcywh_to_xyxy(truth.float())
            )
            refined_iou = aligned_iou(
                box_cxcywh_to_xyxy(refined), box_cxcywh_to_xyxy(truth.float())
            )
            delta = refined_iou - base_iou
            count = len(predicted)
            aggregate["matched_boxes"] += count
            aggregate["baseline_iou_sum"] += float(base_iou.sum())
            aggregate["refined_iou_sum"] += float(refined_iou.sum())
            aggregate["iou_delta_sum"] += float(delta.sum())
            aggregate["iou_improved"] += int((delta > 0).sum())
            aggregate["iou_worsened"] += int((delta < 0).sum())
            aggregate["offset_abs_error_sum"] += float(
                (predicted_offsets - target_offsets).abs().sum()
            )
            aggregate["side_count"] += int(target_offsets.numel())
            aggregate["exact_bin"] += int(exact.sum())
            aggregate["top2_bin"] += int(top2.sum())
            aggregate["within_one_bin"] += int(within_one.sum())
            aggregate["active_sides"] += int(active.sum())
            aggregate["active_sign_correct"] += int((sign_correct & active).sum())
            aggregate["entropy_sum"] += float(entropy.sum())
            aggregate["predicted_offset_abs_sum"] += float(predicted_offsets.abs().sum())
            aggregate["target_offset_abs_sum"] += float(target_offsets.abs().sum())
            add_stratum(strata, "matched_iou", base_iou, delta)

        if batch_index < args.gradient_batches:
            reader.zero_grad(set_to_none=True)
            refined_gradient = reader(
                s8, s16, predicted.float(), batch_ids, "aligned"
            )
            volume_logits = reader.last_logits
            box_loss = regression_loss(refined_gradient, truth.float())[0]
            bin_loss = reader.bin_loss_and_accuracy(
                volume_logits, predicted.float(), truth.float()
            )[0]
            gradient_records.append(
                gradient_pair_statistics(box_loss, bin_loss, parameters)
            )
        if batch_index == 0 or (batch_index + 1) % 10 == 0:
            print(f"batch={batch_index + 1}/{len(loader)}", flush=True)

    count = int(aggregate["matched_boxes"])
    side_count = int(aggregate["side_count"])
    active_sides = int(aggregate["active_sides"])
    matched = {
        "matched_boxes": count,
        "baseline_mean_iou": aggregate["baseline_iou_sum"] / max(1, count),
        "refined_mean_iou": aggregate["refined_iou_sum"] / max(1, count),
        "mean_iou_delta": aggregate["iou_delta_sum"] / max(1, count),
        "iou_improved_fraction": aggregate["iou_improved"] / max(1, count),
        "iou_worsened_fraction": aggregate["iou_worsened"] / max(1, count),
        "offset_mae": aggregate["offset_abs_error_sum"] / max(1, side_count),
        "predicted_mean_abs_offset": aggregate["predicted_offset_abs_sum"] / max(1, side_count),
        "target_mean_abs_offset": aggregate["target_offset_abs_sum"] / max(1, side_count),
        "exact_bin_accuracy": aggregate["exact_bin"] / max(1, side_count),
        "top2_bin_accuracy": aggregate["top2_bin"] / max(1, side_count),
        "within_one_bin_accuracy": aggregate["within_one_bin"] / max(1, side_count),
        "active_side_sign_accuracy": aggregate["active_sign_correct"] / max(1, active_sides),
        "active_sides": active_sides,
        "mean_entropy": aggregate["entropy_sum"] / max(1, side_count),
        "maximum_entropy": math.log(reader.offset_bins),
        "strata": finalize_strata(strata),
    }
    if gradient_records:
        gradient_summary = {
            "batches": len(gradient_records),
            "mean_cosine": float(np.mean([item["cosine"] for item in gradient_records])),
            "negative_cosine_fraction": float(
                np.mean([item["cosine"] < 0 for item in gradient_records])
            ),
            "mean_box_gradient_l2": float(
                np.mean([item["box_gradient_l2"] for item in gradient_records])
            ),
            "mean_bin_gradient_l2": float(
                np.mean([item["bin_gradient_l2"] for item in gradient_records])
            ),
            "mean_weighted_bin_to_box_gradient_ratio": float(
                np.mean(
                    [item["weighted_bin_to_box_gradient_ratio"] for item in gradient_records]
                )
            ),
            "per_batch": gradient_records,
        }
    else:
        gradient_summary = {
            "batches": 0,
            "mean_cosine": None,
            "negative_cosine_fraction": None,
            "mean_box_gradient_l2": None,
            "mean_bin_gradient_l2": None,
            "mean_weighted_bin_to_box_gradient_ratio": None,
            "per_batch": [],
        }
    baseline_curve = ap_by_iou(coco, detections["baseline"])
    aligned_curve = ap_by_iou(coco, detections["aligned"])
    delta_curve = {
        key: aligned_curve[key] - baseline_curve[key] for key in baseline_curve
    }
    report = {
        "experiment": "S-SCBV0-FIXED-BEST-FAILURE-DIAGNOSIS",
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "detector_weight_source": weight_source,
        "ap_by_iou": {
            "baseline": baseline_curve,
            "aligned": aligned_curve,
            "delta": delta_curve,
        },
        "matched_box_diagnostics": matched,
        "gradient_conflict": gradient_summary,
        "interpretation_rule": {
            "objective_conflict_supported": (
                gradient_summary["mean_cosine"] is not None
                and gradient_summary["mean_cosine"] < 0
                and gradient_summary["negative_cosine_fraction"] >= 0.5
            ),
            "hard_bin_dominates_if_ratio_gt_1": (
                gradient_summary["mean_weighted_bin_to_box_gradient_ratio"] is not None
                and gradient_summary["mean_weighted_bin_to_box_gradient_ratio"] > 1.0
            ),
        },
    }
    save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
