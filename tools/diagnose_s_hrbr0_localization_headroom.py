#!/usr/bin/env python3
"""S-HRBR0: quantify localization versus classification headroom in frozen D-FINE.

The diagnostic follows RefineBox's motivation experiment. Hungarian-matched
positive queries are given perfect boxes, perfect positive classification, or
both, while the detector remains frozen. No training state is updated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


MODES = (
    "baseline",
    "oracle_positive_box",
    "oracle_positive_class",
    "oracle_full_class",
    "oracle_box_and_full_class",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("E:/two_paper/outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("E:/two_paper/reports/24_high_resolution_box_refinement/S_HRBR0_HEADROOM/report.json"),
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    return parser.parse_args()


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
        dim=-1,
    )


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1
    )


def aligned_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes_a[:, :2], boxes_b[:, :2])
    bottom_right = torch.minimum(boxes_a[:, 2:], boxes_b[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(-1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp_min(0).prod(-1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp_min(0).prod(-1)
    return intersection / (area_a + area_b - intersection).clamp_min(1e-12)


def collect_detections(storage, targets, results, category_ids):
    for target, result in zip(targets, results):
        boxes = result["boxes"].detach().cpu().clone()
        boxes[:, 2:] -= boxes[:, :2]
        for box, score, label in zip(
            boxes.tolist(), result["scores"].tolist(), result["labels"].tolist()
        ):
            storage.append(
                {
                    "image_id": int(target["image_id"].item()),
                    "category_id": int(category_ids[int(label)]),
                    "bbox": box,
                    "score": float(score),
                }
            )


def coco_metrics(coco_gt, detections):
    from faster_coco_eval import COCOeval_faster

    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = (
        "AP",
        "AP50",
        "AP75",
        "APS",
        "APM",
        "APL",
        "AR1",
        "AR10",
        "AR100",
        "ARS",
        "ARM",
        "ARL",
    )
    return {name: float(value) for name, value in zip(names, evaluator.stats)}


def numeric_summary(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo))
    from src.core import YAMLConfig

    if not torch.cuda.is_available():
        raise RuntimeError("S-HRBR0 full validation requires CUDA")

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model.cuda().eval()
    criterion = cfg.criterion.cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = checkpoint.get("ema", {}).get("module")
    weight_source = "ema.module"
    if weights is None:
        weights = checkpoint.get("model", checkpoint)
        weight_source = "model_or_raw"
    model.load_state_dict(weights, strict=True)

    loader, postprocessor = cfg.val_dataloader, cfg.postprocessor
    coco = loader.dataset.coco
    category_ids = sorted(coco.getCatIds())
    detections = {mode: [] for mode in MODES}
    matched_ious, matched_scores = [], []
    mean_side_errors_pixels, max_side_errors_pixels = [], []
    images_seen = 0
    started = time.time()
    use_amp = args.precision == "fp16"

    with torch.inference_mode():
        for batch_index, (samples, targets) in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            samples = samples.cuda(non_blocking=True)
            targets = [
                {
                    key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
                    for key, value in target.items()
                }
                for target in targets
            ]
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(samples)
            core_outputs = {
                "pred_logits": outputs["pred_logits"].float(),
                "pred_boxes": outputs["pred_boxes"].float(),
            }
            matcher_targets = []
            for target in targets:
                width, height = target["orig_size"].float()
                scale = torch.stack((width, height, width, height))
                normalized_xyxy = target["boxes"] / scale
                matcher_targets.append(
                    {
                        "labels": target["labels"],
                        "boxes": box_xyxy_to_cxcywh(normalized_xyxy),
                    }
                )
            indices = criterion.matcher(core_outputs, matcher_targets)["indices"]
            sizes = torch.stack([target["orig_size"] for target in targets])

            mode_outputs = {
                mode: {
                    "pred_logits": core_outputs["pred_logits"].clone(),
                    "pred_boxes": core_outputs["pred_boxes"].clone(),
                }
                for mode in MODES
            }
            for mode in ("oracle_full_class", "oracle_box_and_full_class"):
                mode_outputs[mode]["pred_logits"].fill_(-20.0)
            for image_index, ((query_indices, target_indices), target, matcher_target) in enumerate(
                zip(indices, targets, matcher_targets)
            ):
                query_indices = query_indices.to(core_outputs["pred_boxes"].device)
                target_indices = target_indices.to(matcher_target["boxes"].device)
                if not len(query_indices):
                    continue
                target_boxes = matcher_target["boxes"][target_indices]
                predicted_boxes = core_outputs["pred_boxes"][image_index, query_indices]
                predicted_logits = core_outputs["pred_logits"][image_index, query_indices]

                predicted_xyxy = box_cxcywh_to_xyxy(predicted_boxes)
                target_xyxy = box_cxcywh_to_xyxy(target_boxes)
                ious = aligned_iou(predicted_xyxy, target_xyxy)
                matched_ious.extend(ious.detach().cpu().tolist())
                matched_scores.extend(predicted_logits.sigmoid().max(-1).values.cpu().tolist())

                width, height = target["orig_size"].float()
                scale = torch.stack((width, height, width, height))
                side_errors = ((predicted_xyxy - target_xyxy).abs() * scale).cpu()
                mean_side_errors_pixels.extend(side_errors.mean(-1).tolist())
                max_side_errors_pixels.extend(side_errors.max(-1).values.tolist())

                for mode in ("oracle_positive_box", "oracle_box_and_full_class"):
                    mode_outputs[mode]["pred_boxes"][image_index, query_indices] = target_boxes
                for mode in ("oracle_positive_class",):
                    # Lift every matched positive above unmatched queries while
                    # preserving the detector's ordering among positives.
                    mode_outputs[mode]["pred_logits"][image_index, query_indices] = (
                        predicted_logits + 5.0
                    )
                for mode in ("oracle_full_class", "oracle_box_and_full_class"):
                    mode_outputs[mode]["pred_logits"][image_index, query_indices] = (
                        predicted_logits + 5.0
                    )

            for mode, adjusted_outputs in mode_outputs.items():
                results = postprocessor(adjusted_outputs, sizes)
                collect_detections(detections[mode], targets, results, category_ids)
            images_seen += len(targets)
            if batch_index == 0 or (batch_index + 1) % 10 == 0:
                print(
                    f"batch={batch_index + 1}/{len(loader)} images={images_seen}",
                    flush=True,
                )

    metrics = {
        mode: coco_metrics(coco, mode_detections)
        for mode, mode_detections in detections.items()
    }
    baseline = metrics["baseline"]
    deltas = {
        mode: {
            key: metrics[mode][key] - baseline[key]
            for key in ("AP", "AP50", "AP75", "APS", "APM", "AR100")
        }
        for mode in MODES[1:]
    }
    iou_array = np.asarray(matched_ious, dtype=np.float64)
    report = {
        "protocol": {
            "diagnostic": "S-HRBR0冻结A00定位/分类上限诊断",
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_source": weight_source,
            "images_seen": images_seen,
            "updates": 0,
            "elapsed_seconds": time.time() - started,
        },
        "metrics": metrics,
        "delta_from_baseline": deltas,
        "matched_positive_geometry": {
            "iou": numeric_summary(matched_ious),
            "fraction_iou_ge_050": float((iou_array >= 0.50).mean()) if len(iou_array) else None,
            "fraction_iou_ge_075": float((iou_array >= 0.75).mean()) if len(iou_array) else None,
            "fraction_iou_ge_090": float((iou_array >= 0.90).mean()) if len(iou_array) else None,
            "score": numeric_summary(matched_scores),
            "mean_side_error_pixels": numeric_summary(mean_side_errors_pixels),
            "max_side_error_pixels": numeric_summary(max_side_errors_pixels),
        },
        "entry_gate": {
            "localization_headroom_exceeds_classification": deltas[
                "oracle_positive_box"
            ]["AP"]
            > deltas["oracle_full_class"]["AP"],
            "localization_headroom_ap75_at_least_0p005": deltas[
                "oracle_positive_box"
            ]["AP75"]
            >= 0.005,
        },
        "interpretation_boundary": (
            "Oracle replacement uses GT only for frozen-model diagnosis and is never an inference path. "
            "The full-class oracle removes unmatched detections and is an optimistic classification bound."
        ),
    }
    report["entry_gate"]["pass"] = all(report["entry_gate"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
