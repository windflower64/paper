#!/usr/bin/env python3
"""CPU-only annotation geometry and exact side-perturbation diagnosis for R-DIAG0."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SIZE_BINS = (("lt8", 0, 8), ("8to16", 8, 16), ("16to32", 16, 32),
             ("32to48", 32, 48), ("ge48", 48, float("inf")))
ASPECT_BINS = (("very_tall", 0, 0.5), ("tall", 0.5, 0.8), ("near_square", 0.8, 1.25),
               ("wide", 1.25, 2.0), ("very_wide", 2.0, float("inf")))
SIDES = ("left", "right", "top", "bottom")
SIDE_INDEX = {"left": 0, "top": 1, "right": 2, "bottom": 3}


def bin_name(value, bins):
    return next(name for name, lo, hi in bins if lo <= value < hi)


def quantiles(values):
    x = np.asarray(values, dtype=np.float64)
    if not x.size:
        return {"n": 0}
    return {
        "n": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
        "q05": float(np.quantile(x, .05)), "q25": float(np.quantile(x, .25)),
        "median": float(np.median(x)), "q75": float(np.quantile(x, .75)),
        "q95": float(np.quantile(x, .95)), "min": float(x.min()), "max": float(x.max()),
    }


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-12)


def perturb(box, side, delta):
    out = list(box)
    out[SIDE_INDEX[side]] += delta
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def load_records(path, resize_h, resize_w):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    images = {int(i["id"]): i for i in data["images"]}
    records = []
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        image = images[int(ann["image_id"])]
        x, y, w, h = map(float, ann["bbox"])
        sx, sy = resize_w / float(image["width"]), resize_h / float(image["height"])
        x, y, w, h = x * sx, y * sy, w * sx, h * sy
        if w <= 0 or h <= 0:
            continue
        edge, aspect = math.sqrt(w * h), w / h
        records.append({
            "ann_id": int(ann.get("id", len(records))), "image_id": int(ann["image_id"]),
            "w": w, "h": h, "edge": edge, "aspect": aspect,
            "size_bin": bin_name(edge, SIZE_BINS), "aspect_bin": bin_name(aspect, ASPECT_BINS),
            "box": (x, y, x + w, y + h),
        })
    return records, len(images)


def summarize_records(records, image_count):
    return {
        "images": image_count,
        "objects": len(records),
        "width_px": quantiles([r["w"] for r in records]),
        "height_px": quantiles([r["h"] for r in records]),
        "sqrt_area_px": quantiles([r["edge"] for r in records]),
        "aspect_w_over_h": quantiles([r["aspect"] for r in records]),
        "size_counts": dict(Counter(r["size_bin"] for r in records)),
        "aspect_counts": dict(Counter(r["aspect_bin"] for r in records)),
        "wide_fraction_w_over_h_gt_1_25": float(np.mean([r["aspect"] > 1.25 for r in records])),
        "very_wide_fraction_w_over_h_gt_2": float(np.mean([r["aspect"] > 2 for r in records])),
    }


def perturbation_stats(records):
    values = defaultdict(list)
    rows = []
    for r in records:
        box = r["box"]
        for mode, amount in (("pixel_1", 1.0), ("pixel_2", 2.0), ("relative_5pct", 0.05)):
            for side in SIDES:
                scale = r["w"] if side in ("left", "right") else r["h"]
                delta_abs = amount if mode.startswith("pixel") else amount * scale
                drops = []
                for sign in (-1.0, 1.0):
                    changed = perturb(box, side, sign * delta_abs)
                    if changed is not None:
                        drops.append(1.0 - iou(box, changed))
                if not drops:
                    continue
                drop = float(np.mean(drops))
                keys = (("all", "all"), (r["size_bin"], "all"),
                        ("all", r["aspect_bin"]), (r["size_bin"], r["aspect_bin"]))
                for size_group, aspect_group in keys:
                    values[(mode, side, size_group, aspect_group)].append(drop)
                rows.append({"ann_id": r["ann_id"], "mode": mode, "side": side,
                             "size_bin": r["size_bin"], "aspect_bin": r["aspect_bin"],
                             "iou_drop": drop})
    summary = []
    for (mode, side, size_group, aspect_group), vals in sorted(values.items()):
        item = {"mode": mode, "side": side, "size_bin": size_group,
                "aspect_bin": aspect_group}
        item.update(quantiles(vals))
        summary.append(item)
    return rows, summary


def markdown(train_summary, val_summary, perturb_summary):
    def count_table(summary, key):
        total = summary["objects"]
        return "\n".join(f"| {name} | {count} | {count / max(total, 1):.2%} |"
                          for name, count in summary[key].items())

    selected = {(x["mode"], x["side"]): x for x in perturb_summary
                if x["size_bin"] == "all" and x["aspect_bin"] == "all"}
    perturb_rows = "\n".join(
        f"| {mode} | {side} | {selected[(mode, side)]['mean']:.6f} | "
        f"{selected[(mode, side)]['median']:.6f} |"
        for mode in ("pixel_1", "pixel_2", "relative_5pct") for side in SIDES
    )
    return f"""# R-DIAG0A：标注几何与逐边扰动（CPU）

生成日期：2026-08-13  
输入尺寸：512×640  
说明：宽高比分布只使用train形成假设；Val仅用于预注册的几何扰动验证，不使用Test。

## 1. Train几何分布

- 图像/目标：{train_summary['images']}/{train_summary['objects']}
- 宽高比中位数：{train_summary['aspect_w_over_h']['median']:.4f}
- w/h>1.25比例：{train_summary['wide_fraction_w_over_h_gt_1_25']:.2%}
- w/h>2比例：{train_summary['very_wide_fraction_w_over_h_gt_2']:.2%}

| 尺寸分箱 | 数量 | 比例 |
|---|---:|---:|
{count_table(train_summary, 'size_counts')}

| 形状分箱 | 数量 | 比例 |
|---|---:|---:|
{count_table(train_summary, 'aspect_counts')}

## 2. Val精确几何扰动

每次只移动一条GT边，正负两个方向取平均。该表回答几何敏感性，不包含模型学习误差。

| 扰动 | 边 | 平均IoU下降 | 中位数下降 |
|---|---|---:|---:|
{perturb_rows}

## 3. 解释边界

- 等像素扰动若显示top/bottom更敏感，首先是扁平框高度较小造成的几何效应；
- 等相对边长扰动应在四边间近似对称，它是排除纯尺度效应的必要对照；
- 是否存在“学习不对称”，必须等GPU恢复后统计matched query逐边误差、FDR熵和跨层KL；
- 本结果不能单独触发R-FDR1或R-LSD1训练。
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ann", required=True)
    parser.add_argument("--val-ann", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, train_images = load_records(args.train_ann, args.height, args.width)
    val, val_images = load_records(args.val_ann, args.height, args.width)
    train_summary = summarize_records(train, train_images)
    val_summary = summarize_records(val, val_images)
    raw_rows, perturb_summary = perturbation_stats(val)
    result = {"protocol": {"height": args.height, "width": args.width,
                            "test_used": False},
              "train": train_summary, "val": val_summary,
              "val_side_perturbation": perturb_summary}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(
        markdown(train_summary, val_summary, perturb_summary), encoding="utf-8")
    with (args.output_dir / "val_side_perturbation_raw.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=raw_rows[0].keys())
        writer.writeheader(); writer.writerows(raw_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
