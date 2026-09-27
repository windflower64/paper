"""Audit how much geometric correspondence exists in paired RGB-T labels.

The project deliberately avoids assuming pixel alignment.  This diagnostic
quantifies whether that assumption is too strict by comparing normalized RGB
and thermal boxes, testing a train-fitted global affine mapping, and testing
sequence-specific mappings with held-out frames.  It never changes the data.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _quantiles(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _radius_coverage(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "within_0p02": float((values <= 0.02).mean()),
        "within_0p04": float((values <= 0.04).mean()),
        "within_0p06": float((values <= 0.06).mean()),
        "within_0p10": float((values <= 0.10).mean()),
        "within_0p15": float((values <= 0.15).mean()),
    }


def _valid_yolo_box(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    candidates = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            x, y, w, h = map(float, fields[1:5])
        except ValueError:
            continue
        if (
            0.0 <= x <= 1.0
            and 0.0 <= y <= 1.0
            and 0.0 < w <= 1.0
            and 0.0 < h <= 1.0
        ):
            candidates.append(np.asarray([x, y, w, h], dtype=np.float64))
    if not candidates:
        return None
    # Anti-UAV-6K normally has one target.  Selecting the largest valid box
    # makes the audit deterministic for the rare multi-line label.
    return max(candidates, key=lambda box: float(box[2] * box[3]))


def _load_pairs(coco_path: Path, thermal_label_dir: Path) -> dict:
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in coco["images"]}
    visible_by_image: dict[int, list[np.ndarray]] = defaultdict(list)
    for ann in coco["annotations"]:
        image = images[int(ann["image_id"])]
        x, y, w, h = map(float, ann["bbox"])
        if w <= 0.0 or h <= 0.0:
            continue
        visible_by_image[int(ann["image_id"])].append(
            np.asarray(
                [
                    (x + 0.5 * w) / float(image["width"]),
                    (y + 0.5 * h) / float(image["height"]),
                    w / float(image["width"]),
                    h / float(image["height"]),
                ],
                dtype=np.float64,
            )
        )

    rows = []
    counts = defaultdict(int)
    for image_id, image in images.items():
        stem = Path(image["file_name"]).stem
        visible_candidates = visible_by_image.get(image_id, [])
        visible = (
            max(visible_candidates, key=lambda box: float(box[2] * box[3]))
            if visible_candidates
            else None
        )
        thermal = _valid_yolo_box(thermal_label_dir / f"{stem}.txt")
        visible_positive = visible is not None
        thermal_positive = thermal is not None
        key = (
            "both_positive"
            if visible_positive and thermal_positive
            else "visible_only"
            if visible_positive
            else "thermal_only"
            if thermal_positive
            else "both_negative"
        )
        counts[key] += 1
        if visible_positive and thermal_positive:
            rows.append(
                {
                    "stem": stem,
                    "sequence": stem.rsplit("_", 1)[0],
                    "visible": visible,
                    "thermal": thermal,
                }
            )
    counts["images"] = len(images)
    counts["presence_agreement"] = counts["both_positive"] + counts["both_negative"]
    return {"rows": rows, "presence": dict(counts)}


def _boxes_to_corners(boxes: np.ndarray) -> np.ndarray:
    center = boxes[:, :2]
    half = 0.5 * boxes[:, 2:]
    return np.concatenate((center - half, center + half), axis=1)


def _same_coordinate_iou(visible: np.ndarray, thermal: np.ndarray) -> np.ndarray:
    a = _boxes_to_corners(visible)
    b = _boxes_to_corners(thermal)
    top_left = np.maximum(a[:, :2], b[:, :2])
    bottom_right = np.minimum(a[:, 2:], b[:, 2:])
    intersection = np.maximum(bottom_right - top_left, 0.0).prod(axis=1)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0.0).prod(axis=1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0.0).prod(axis=1)
    return intersection / np.maximum(area_a + area_b - intersection, 1e-12)


def _fit_affine(source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    design = np.concatenate(
        (source_xy, np.ones((source_xy.shape[0], 1), dtype=np.float64)), axis=1
    )
    matrix, _, _, _ = np.linalg.lstsq(design, target_xy, rcond=None)
    return matrix


def _apply_affine(source_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    design = np.concatenate(
        (source_xy, np.ones((source_xy.shape[0], 1), dtype=np.float64)), axis=1
    )
    return design @ matrix


def _center_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(predicted - target, axis=1)


def _correlation(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    result = {}
    for index, axis in enumerate(("x", "y")):
        if np.std(source[:, index]) == 0.0 or np.std(target[:, index]) == 0.0:
            result[axis] = 0.0
        else:
            result[axis] = float(np.corrcoef(source[:, index], target[:, index])[0, 1])
    return result


def _sequence_cross_validated_error(rows: list[dict]) -> tuple[np.ndarray, int]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["sequence"]].append(row)
    errors = []
    used_sequences = 0
    for sequence_rows in grouped.values():
        sequence_rows.sort(key=lambda row: row["stem"])
        if len(sequence_rows) < 8:
            continue
        train_rows = sequence_rows[::2]
        test_rows = sequence_rows[1::2]
        if len(train_rows) < 3 or not test_rows:
            continue
        train_ir = np.stack([row["thermal"][:2] for row in train_rows])
        train_rgb = np.stack([row["visible"][:2] for row in train_rows])
        test_ir = np.stack([row["thermal"][:2] for row in test_rows])
        test_rgb = np.stack([row["visible"][:2] for row in test_rows])
        matrix = _fit_affine(train_ir, train_rgb)
        errors.append(_center_error(_apply_affine(test_ir, matrix), test_rgb))
        used_sequences += 1
    if not errors:
        return np.empty(0, dtype=np.float64), 0
    return np.concatenate(errors), used_sequences


def _sequence_offset_audit(rows: list[dict]) -> dict:
    """Measure how much a simple per-sequence translation varies and helps."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["sequence"]].append(row)
    offsets = []
    residuals = []
    used = 0
    for sequence_rows in grouped.values():
        if len(sequence_rows) < 3:
            continue
        rgb = np.stack([row["visible"][:2] for row in sequence_rows])
        ir = np.stack([row["thermal"][:2] for row in sequence_rows])
        offset = np.median(rgb - ir, axis=0)
        offsets.append(offset)
        residuals.append(_center_error(ir + offset, rgb))
        used += 1
    if not offsets:
        return {"used_sequences": 0}
    offset_array = np.stack(offsets)
    residual_array = np.concatenate(residuals)
    return {
        "used_sequences": used,
        "median_offset_x": _quantiles(offset_array[:, 0]),
        "median_offset_y": _quantiles(offset_array[:, 1]),
        "median_offset_norm": _quantiles(np.linalg.norm(offset_array, axis=1)),
        "within_sequence_center_error_after_oracle_median_offset": {
            **_quantiles(residual_array),
            **_radius_coverage(residual_array),
        },
    }


def _arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([row["visible"] for row in rows]),
        np.stack([row["thermal"] for row in rows]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible-train-coco", type=Path, required=True)
    parser.add_argument("--thermal-train-labels", type=Path, required=True)
    parser.add_argument("--visible-test-coco", type=Path, required=True)
    parser.add_argument("--thermal-test-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = _load_pairs(args.visible_train_coco, args.thermal_train_labels)
    test = _load_pairs(args.visible_test_coco, args.thermal_test_labels)
    train_rgb, train_ir = _arrays(train["rows"])
    test_rgb, test_ir = _arrays(test["rows"])

    train_matrix = _fit_affine(train_ir[:, :2], train_rgb[:, :2])
    train_reverse_matrix = _fit_affine(train_rgb[:, :2], train_ir[:, :2])
    test_oracle_matrix = _fit_affine(test_ir[:, :2], test_rgb[:, :2])
    test_oracle_reverse_matrix = _fit_affine(test_rgb[:, :2], test_ir[:, :2])
    raw_error = _center_error(test_ir[:, :2], test_rgb[:, :2])
    train_affine_error = _center_error(
        _apply_affine(test_ir[:, :2], train_matrix), test_rgb[:, :2]
    )
    train_reverse_affine_error = _center_error(
        _apply_affine(test_rgb[:, :2], train_reverse_matrix), test_ir[:, :2]
    )
    oracle_affine_error = _center_error(
        _apply_affine(test_ir[:, :2], test_oracle_matrix), test_rgb[:, :2]
    )
    sequence_error, sequence_count = _sequence_cross_validated_error(test["rows"])
    iou = _same_coordinate_iou(test_rgb, test_ir)

    for presence in (train["presence"], test["presence"]):
        presence["presence_agreement_fraction"] = (
            float(presence["presence_agreement"]) / float(presence["images"])
        )

    report = {
        "status": "PASS",
        "purpose_zh": "量化成对RGB与红外标签中可恢复的几何对应程度，不修改数据。",
        "train_presence": train["presence"],
        "test_presence": test["presence"],
        "test_both_positive_pairs": len(test["rows"]),
        "test_center_correlation": _correlation(test_ir[:, :2], test_rgb[:, :2]),
        "test_raw_normalized_center_error": {
            **_quantiles(raw_error),
            **_radius_coverage(raw_error),
        },
        "test_same_coordinate_iou": {
            **_quantiles(iou),
            "fraction_ge_0p1": float((iou >= 0.1).mean()),
            "fraction_ge_0p5": float((iou >= 0.5).mean()),
        },
        "train_fitted_affine_on_test_center_error": {
            **_quantiles(train_affine_error),
            **_radius_coverage(train_affine_error),
        },
        "train_fitted_visible_to_thermal_on_test_center_error": {
            **_quantiles(train_reverse_affine_error),
            **_radius_coverage(train_reverse_affine_error),
        },
        "test_oracle_affine_center_error": {
            **_quantiles(oracle_affine_error),
            **_radius_coverage(oracle_affine_error),
        },
        "test_sequence_affine_cross_validation": {
            "used_sequences": sequence_count,
            "center_error": {
                **_quantiles(sequence_error),
                **_radius_coverage(sequence_error),
            },
        },
        "train_sequence_offset_audit": _sequence_offset_audit(train["rows"]),
        "test_sequence_offset_audit": _sequence_offset_audit(test["rows"]),
        "train_fitted_affine_matrix": train_matrix.tolist(),
        "train_fitted_visible_to_thermal_matrix": train_reverse_matrix.tolist(),
        "test_oracle_affine_matrix": test_oracle_matrix.tolist(),
        "test_oracle_visible_to_thermal_matrix": test_oracle_reverse_matrix.tolist(),
        "interpretation_zh": [
            "原坐标误差用于判断直接同位置融合是否合理。",
            "训练集拟合映射在test上的误差用于判断固定全局对齐能否跨场景泛化。",
            "按序列拟合并留出帧验证的误差用于判断局部、样本条件对齐是否仍有潜力。",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
