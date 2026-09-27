"""P0 audit for M-B1 target-level spatial alignment.

The audit keeps RGB and thermal boxes in their native coordinate systems.  It
first checks whether each frozen detector recalls its own target, then asks if
a train-fitted spatial prior separates paired targets from globally mismatched
targets.  Test labels are used only for evaluation, never for fitting the
mapping or selecting distance thresholds.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


TOPK = (1, 3, 10, 100)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center = boxes[..., :2]
    half = boxes[..., 2:] * 0.5
    return torch.cat((center - half, center + half), dim=-1)


def box_iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area_a = (box[2:] - box[:2]).clamp_min(0).prod()
    area_b = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(dim=-1)
    return intersection / (area_a + area_b - intersection).clamp_min(1e-12)


def largest_thermal_box(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    boxes = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            x, y, width, height = map(float, fields[1:5])
        except ValueError:
            continue
        if 0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1:
            boxes.append(np.asarray([x, y, width, height], dtype=np.float64))
    return max(boxes, key=lambda box: float(box[2] * box[3])) if boxes else None


def load_rows(cache_path: Path, visible_coco: Path, thermal_labels: Path) -> dict:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    coco = json.loads(visible_coco.read_text(encoding="utf-8"))
    image_info = {int(item["id"]): item for item in coco["images"]}
    visible_annotations: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if not annotation.get("iscrowd", 0):
            visible_annotations[int(annotation["image_id"])].append(annotation)

    visible_scores = cache["visible_logits"].squeeze(-1).float().sigmoid()
    thermal_scores = cache["thermal_logits"].squeeze(-1).float().sigmoid()
    visible_boxes = cache["visible_boxes"].float().clamp(0, 1)
    thermal_boxes = cache["thermal_boxes"].float().clamp(0, 1)
    rows = []
    for index, image_id in enumerate(cache["image_ids"].tolist()):
        info = image_info[int(image_id)]
        annotations = visible_annotations.get(int(image_id), [])
        visible_gt = None
        if annotations:
            annotation = max(
                annotations, key=lambda item: float(item["bbox"][2] * item["bbox"][3])
            )
            x, y, width, height = map(float, annotation["bbox"])
            visible_gt = np.asarray(
                [
                    (x + 0.5 * width) / info["width"],
                    (y + 0.5 * height) / info["height"],
                    width / info["width"],
                    height / info["height"],
                ],
                dtype=np.float64,
            )
        stem = Path(info["file_name"]).stem
        thermal_gt = largest_thermal_box(thermal_labels / f"{stem}.txt")
        rows.append(
            {
                "stem": stem,
                "sequence": stem.rsplit("_", 1)[0],
                "visible_gt": visible_gt,
                "thermal_gt": thermal_gt,
                "visible_scores": visible_scores[index],
                "thermal_scores": thermal_scores[index],
                "visible_boxes": visible_boxes[index],
                "thermal_boxes": thermal_boxes[index],
            }
        )
    return {"rows": rows, "cache": cache}


def fit_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.concatenate((source, np.ones((len(source), 1))), axis=1)
    matrix, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return matrix


def apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    design = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    return design @ matrix


def quantiles(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def candidate_recall(rows: list[dict], modality: str) -> dict:
    result = {topk: {"count": 0, "iou_0p5": 0, "center_0p05": 0} for topk in TOPK}
    for row in rows:
        gt_array = row[f"{modality}_gt"]
        if gt_array is None:
            continue
        gt = torch.as_tensor(gt_array, dtype=torch.float32)
        boxes = row[f"{modality}_boxes"]
        scores = row[f"{modality}_scores"]
        order = scores.argsort(descending=True)
        gt_xyxy = cxcywh_to_xyxy(gt)
        boxes_xyxy = cxcywh_to_xyxy(boxes)
        for topk in TOPK:
            selected = order[:topk]
            iou = box_iou_one_to_many(gt_xyxy, boxes_xyxy[selected]).amax()
            distance = torch.linalg.vector_norm(
                boxes[selected, :2] - gt[:2], dim=-1
            ).amin()
            result[topk]["count"] += 1
            result[topk]["iou_0p5"] += int(iou >= 0.5)
            result[topk]["center_0p05"] += int(distance <= 0.05)
    return {
        str(topk): {
            "targets": values["count"],
            "recall_iou_0p5": values["iou_0p5"] / max(1, values["count"]),
            "recall_center_0p05": values["center_0p05"] / max(1, values["count"]),
        }
        for topk, values in result.items()
    }


def auc_from_distances(true_distance: np.ndarray, mismatch_distance: np.ndarray) -> float:
    positive_scores = -np.asarray(true_distance, dtype=np.float64)
    negative_scores = -np.asarray(mismatch_distance, dtype=np.float64)
    scores = np.concatenate((positive_scores, negative_scores))
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    rank_sum = ranks[: len(positive_scores)].sum()
    n_pos = len(positive_scores)
    n_neg = len(negative_scores)
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def choose_threshold(true_distance: np.ndarray, mismatch_distance: np.ndarray) -> dict:
    candidates = np.unique(
        np.quantile(
            np.concatenate((true_distance, mismatch_distance)),
            np.linspace(0, 1, 1001),
        )
    )
    best = None
    for threshold in candidates:
        true_accept = float((true_distance <= threshold).mean())
        mismatch_reject = float((mismatch_distance > threshold).mean())
        balanced_accuracy = 0.5 * (true_accept + mismatch_reject)
        candidate = (balanced_accuracy, mismatch_reject, -float(threshold))
        if best is None or candidate > best[0]:
            best = (candidate, float(threshold), true_accept, mismatch_reject)
    assert best is not None
    return {
        "threshold": best[1],
        "balanced_accuracy": best[0][0],
        "true_accept_rate": best[2],
        "mismatch_reject_rate": best[3],
    }


def evaluate_threshold(
    true_distance: np.ndarray, mismatch_distance: np.ndarray, threshold: float
) -> dict:
    true_accept = float((true_distance <= threshold).mean())
    mismatch_reject = float((mismatch_distance > threshold).mean())
    return {
        "threshold_from_train": threshold,
        "balanced_accuracy": 0.5 * (true_accept + mismatch_reject),
        "true_accept_rate": true_accept,
        "mismatch_reject_rate": mismatch_reject,
        "auc_lower_distance_is_paired": auc_from_distances(
            true_distance, mismatch_distance
        ),
        "true_distance": quantiles(true_distance),
        "mismatch_distance": quantiles(mismatch_distance),
    }


def topk_spatial_distances(
    rows: list[dict], affine: np.ndarray, topk: int, mismatch_shift: int
) -> tuple[np.ndarray, np.ndarray]:
    paired = [row for row in rows if row["visible_gt"] is not None and row["thermal_gt"] is not None]
    predicted_rgb = []
    thermal_centers = []
    for row in paired:
        visible_order = row["visible_scores"].argsort(descending=True)
        thermal_order = row["thermal_scores"].argsort(descending=True)
        rgb_center = row["visible_boxes"][visible_order[0], :2].numpy()
        predicted_rgb.append(apply_affine(rgb_center[None], affine)[0])
        thermal_centers.append(row["thermal_boxes"][thermal_order[:topk], :2].numpy())
    predicted_rgb = np.stack(predicted_rgb)
    thermal_centers = np.stack(thermal_centers)
    true_distance = np.linalg.norm(
        thermal_centers - predicted_rgb[:, None, :], axis=-1
    ).min(axis=1)
    mismatched_centers = np.roll(thermal_centers, mismatch_shift, axis=0)
    mismatch_distance = np.linalg.norm(
        mismatched_centers - predicted_rgb[:, None, :], axis=-1
    ).min(axis=1)
    return true_distance, mismatch_distance


def gt_spatial_distances(
    rows: list[dict], affine: np.ndarray, mismatch_shift: int
) -> tuple[np.ndarray, np.ndarray]:
    paired = [row for row in rows if row["visible_gt"] is not None and row["thermal_gt"] is not None]
    visible = np.stack([row["visible_gt"][:2] for row in paired])
    thermal = np.stack([row["thermal_gt"][:2] for row in paired])
    predicted = apply_affine(visible, affine)
    true_distance = np.linalg.norm(predicted - thermal, axis=1)
    mismatch_distance = np.linalg.norm(
        predicted - np.roll(thermal, mismatch_shift, axis=0), axis=1
    )
    return true_distance, mismatch_distance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--visible-train-coco", type=Path, required=True)
    parser.add_argument("--visible-test-coco", type=Path, required=True)
    parser.add_argument("--thermal-train-labels", type=Path, required=True)
    parser.add_argument("--thermal-test-labels", type=Path, required=True)
    parser.add_argument("--mismatch-shift", type=int, default=607)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = load_rows(args.train_cache, args.visible_train_coco, args.thermal_train_labels)
    test = load_rows(args.test_cache, args.visible_test_coco, args.thermal_test_labels)
    train_paired = [
        row for row in train["rows"] if row["visible_gt"] is not None and row["thermal_gt"] is not None
    ]
    visible_train = np.stack([row["visible_gt"][:2] for row in train_paired])
    thermal_train = np.stack([row["thermal_gt"][:2] for row in train_paired])
    affine = fit_affine(visible_train, thermal_train)

    train_gt = gt_spatial_distances(train["rows"], affine, args.mismatch_shift)
    test_gt = gt_spatial_distances(test["rows"], affine, args.mismatch_shift)
    gt_threshold = choose_threshold(*train_gt)
    gate_results = {
        "gt_centres": {
            "train_selection": gt_threshold,
            "test": evaluate_threshold(*test_gt, gt_threshold["threshold"]),
        }
    }
    for topk in (1, 3, 10):
        train_distances = topk_spatial_distances(
            train["rows"], affine, topk, args.mismatch_shift
        )
        test_distances = topk_spatial_distances(
            test["rows"], affine, topk, args.mismatch_shift
        )
        selection = choose_threshold(*train_distances)
        gate_results[f"frozen_detector_top{topk}"] = {
            "train_selection": selection,
            "test": evaluate_threshold(
                *test_distances, selection["threshold"]
            ),
        }

    report = {
        "schema": "mb1_spatial_alignment_gate_p0_v1",
        "status": "PASS",
        "purpose_zh": "先验证空间对齐前提，不使用无空间置信度宣称能够识别配对。",
        "fit_protocol": {
            "affine_and_threshold_fit_split": "train only",
            "test_used_for_fit_or_threshold": False,
            "mismatch_shift": args.mismatch_shift,
            "train_paired_targets": len(train_paired),
        },
        "train_fitted_visible_to_thermal_affine": affine.tolist(),
        "native_coordinate_candidate_recall": {
            "train": {
                "visible": candidate_recall(train["rows"], "visible"),
                "thermal": candidate_recall(train["rows"], "thermal"),
            },
            "test": {
                "visible": candidate_recall(test["rows"], "visible"),
                "thermal": candidate_recall(test["rows"], "thermal"),
            },
        },
        "spatial_pair_gate": gate_results,
        "decision_rule_zh": (
            "若GT中心可分而冻结检测候选不可分，下一步应先改善目标候选或引入特征级局部变换；"
            "若冻结候选也可分，才允许训练目标级对齐置信度模型。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
