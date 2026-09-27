"""Audit paired RGB-T signal quality without changing the dataset.

This complements the label-geometry audit.  It measures whether the thermal
files contain independent image information, how separable the labelled target
is from its local background, and how much local edge agreement remains after
the train-fitted Visible-to-Thermal affine mapping.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def quantiles(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "negative_fraction": float((array < 0).mean()),
        "positive_fraction": float((array > 0).mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def largest_visible_box(annotations: list[dict], width: int, height: int):
    valid = [ann["bbox"] for ann in annotations if ann["bbox"][2] > 0 and ann["bbox"][3] > 0]
    if not valid:
        return None
    x, y, w, h = max(valid, key=lambda box: float(box[2] * box[3]))
    return np.asarray(
        [(x + 0.5 * w) / width, (y + 0.5 * h) / height, w / width, h / height],
        dtype=np.float32,
    )


def largest_thermal_box(path: Path):
    if not path.is_file():
        return None
    boxes = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            x, y, w, h = map(float, fields[1:5])
        except ValueError:
            continue
        if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
            boxes.append(np.asarray([x, y, w, h], dtype=np.float32))
    return max(boxes, key=lambda box: float(box[2] * box[3])) if boxes else None


def box_mask(shape: tuple[int, int], box: np.ndarray, scale: float = 1.0) -> np.ndarray:
    height, width = shape
    cx, cy, bw, bh = map(float, box)
    half_w = 0.5 * bw * scale
    half_h = 0.5 * bh * scale
    x0 = max(0, min(width, int(np.floor((cx - half_w) * width))))
    x1 = max(0, min(width, int(np.ceil((cx + half_w) * width))))
    y0 = max(0, min(height, int(np.floor((cy - half_h) * height))))
    y1 = max(0, min(height, int(np.ceil((cy + half_h) * height))))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def target_contrast(gray: np.ndarray, box: np.ndarray):
    inner = box_mask(gray.shape, box, 1.0)
    outer = box_mask(gray.shape, box, 2.5)
    ring = outer & ~inner
    if inner.sum() < 4 or ring.sum() < 8:
        return None
    scale = float(gray.std()) + 1e-6
    signed = float((gray[inner].mean() - gray[ring].mean()) / scale)
    return signed, abs(signed)


def edge_map(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    av = a[mask].astype(np.float64)
    bv = b[mask].astype(np.float64)
    if av.size < 32 or av.std() < 1e-8 or bv.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def sample_thermal_in_visible_coordinates(
    thermal: np.ndarray, visible_shape: tuple[int, int], matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    visible_h, visible_w = visible_shape
    thermal_h, thermal_w = thermal.shape
    ys, xs = np.mgrid[0:visible_h, 0:visible_w].astype(np.float32)
    vx = xs / max(visible_w - 1, 1)
    vy = ys / max(visible_h - 1, 1)
    tx = matrix[0, 0] * vx + matrix[1, 0] * vy + matrix[2, 0]
    ty = matrix[0, 1] * vx + matrix[1, 1] * vy + matrix[2, 1]
    valid = (tx >= 0) & (tx <= 1) & (ty >= 0) & (ty <= 1)
    map_x = tx * max(thermal_w - 1, 1)
    map_y = ty * max(thermal_h - 1, 1)
    sampled = cv2.remap(
        thermal,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return sampled, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible-coco", type=Path, required=True)
    parser.add_argument("--visible-images", type=Path, required=True)
    parser.add_argument("--thermal-images", type=Path, required=True)
    parser.add_argument("--thermal-labels", type=Path, required=True)
    parser.add_argument("--visible-to-thermal-affine", nargs=6, type=float, required=True)
    parser.add_argument("--max-images", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    coco = json.loads(args.visible_coco.read_text(encoding="utf-8"))
    annotations = defaultdict(list)
    for ann in coco["annotations"]:
        annotations[int(ann["image_id"])].append(ann)
    images = sorted(coco["images"], key=lambda item: item["file_name"])
    if args.max_images > 0 and len(images) > args.max_images:
        indices = np.linspace(0, len(images) - 1, args.max_images, dtype=int)
        images = [images[index] for index in indices]

    affine = np.asarray(args.visible_to_thermal_affine, dtype=np.float32).reshape(3, 2)
    visible_sizes = Counter()
    thermal_sizes = Counter()
    thermal_modes = Counter()
    metrics: dict[str, list[float]] = defaultdict(list)
    missing = []

    for item in images:
        name = item["file_name"]
        visible_path = args.visible_images / name
        thermal_path = args.thermal_images / name
        if not visible_path.is_file() or not thermal_path.is_file():
            missing.append(name)
            continue
        visible_bgr = cv2.imread(str(visible_path), cv2.IMREAD_COLOR)
        thermal_unchanged = cv2.imread(str(thermal_path), cv2.IMREAD_UNCHANGED)
        thermal_bgr = cv2.imread(str(thermal_path), cv2.IMREAD_COLOR)
        if visible_bgr is None or thermal_bgr is None or thermal_unchanged is None:
            missing.append(name)
            continue

        vh, vw = visible_bgr.shape[:2]
        th, tw = thermal_bgr.shape[:2]
        visible_sizes[f"{vw}x{vh}"] += 1
        thermal_sizes[f"{tw}x{th}"] += 1
        mode = "gray" if thermal_unchanged.ndim == 2 else f"channels_{thermal_unchanged.shape[2]}"
        thermal_modes[mode] += 1

        visible_gray = cv2.cvtColor(visible_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        thermal_gray = cv2.cvtColor(thermal_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        metrics["visible_global_std"].append(float(visible_gray.std()))
        metrics["thermal_global_std"].append(float(thermal_gray.std()))
        metrics["visible_laplacian_variance"].append(
            float(cv2.Laplacian(visible_gray, cv2.CV_32F).var())
        )
        metrics["thermal_laplacian_variance"].append(
            float(cv2.Laplacian(thermal_gray, cv2.CV_32F).var())
        )
        channel_range = thermal_bgr.max(axis=2).astype(np.float32) - thermal_bgr.min(axis=2).astype(np.float32)
        metrics["thermal_channel_range_mean_0_255"].append(float(channel_range.mean()))
        metrics["thermal_exact_grayscale_pixel_fraction"].append(float((channel_range == 0).mean()))

        visible_box = largest_visible_box(annotations[int(item["id"])], int(item["width"]), int(item["height"]))
        thermal_box = largest_thermal_box(args.thermal_labels / f"{Path(name).stem}.txt")
        if visible_box is not None:
            contrast = target_contrast(visible_gray, visible_box)
            if contrast is not None:
                metrics["visible_target_signed_contrast_z"].append(contrast[0])
                metrics["visible_target_absolute_contrast_z"].append(contrast[1])
        if thermal_box is not None:
            contrast = target_contrast(thermal_gray, thermal_box)
            if contrast is not None:
                metrics["thermal_target_signed_contrast_z"].append(contrast[0])
                metrics["thermal_target_absolute_contrast_z"].append(contrast[1])

        if visible_box is not None and thermal_box is not None:
            thermal_affine, valid = sample_thermal_in_visible_coordinates(
                thermal_gray, visible_gray.shape, affine
            )
            thermal_identity = cv2.resize(thermal_gray, (vw, vh), interpolation=cv2.INTER_LINEAR)
            visible_edges = edge_map(visible_gray)
            affine_edges = edge_map(thermal_affine)
            identity_edges = edge_map(thermal_identity)
            local = box_mask(visible_gray.shape, visible_box, 4.0)
            metrics["edge_corr_affine_global"].append(
                correlation(visible_edges, affine_edges, valid)
            )
            metrics["edge_corr_identity_global"].append(
                correlation(visible_edges, identity_edges, np.ones_like(valid))
            )
            metrics["edge_corr_affine_target_neighbourhood"].append(
                correlation(visible_edges, affine_edges, valid & local)
            )
            metrics["edge_corr_identity_target_neighbourhood"].append(
                correlation(visible_edges, identity_edges, local)
            )

    report = {
        "status": "PASS" if not missing else "WARN",
        "purpose_zh": "审计红外信号本身、目标局部可分性及全局仿射后的跨模态局部结构一致性。",
        "sampled_images": len(images),
        "successfully_read_images": len(images) - len(missing),
        "missing_or_unreadable_count": len(missing),
        "missing_examples": missing[:20],
        "visible_sizes": dict(visible_sizes),
        "thermal_sizes": dict(thermal_sizes),
        "thermal_file_modes": dict(thermal_modes),
        "visible_to_thermal_affine": affine.tolist(),
        "metrics": {name: quantiles(values) for name, values in sorted(metrics.items())},
        "interpretation_zh": [
            "热红外通道差异接近零表示三通道复制，不等于图像无信息，但RGB首层是冗余计算。",
            "目标局部绝对对比越高，说明该模态越容易提供目标存在证据。",
            "仿射后局部边缘相关仍低，说明固定全局变换不足以支持逐位置特征相加。",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
