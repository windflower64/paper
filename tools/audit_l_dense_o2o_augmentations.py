#!/usr/bin/env python3
"""Audit 1,000 real L-DQ1 augmented samples before any GPU training."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core import YAMLConfig


BINS = (
    ("lt8", 0.0, 8.0),
    ("8to16", 8.0, 16.0),
    ("16to32", 16.0, 32.0),
    ("32to48", 32.0, 48.0),
    ("ge48", 48.0, float("inf")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/phase_l/l_dq1_gq1_dense_o2o_mal_b16_60e_local.yml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--visualizations", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def size_bin(value: float) -> str:
    for name, low, high in BINS:
        if low <= value < high:
            return name
    raise AssertionError(value)


def proportions(counts: Counter, total: int) -> dict[str, float]:
    return {name: counts[name] / max(total, 1) for name, _, _ in BINS}


def load_original_distribution(annotation_path: Path) -> tuple[Counter, int]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    counts: Counter = Counter()
    total = 0
    for annotation in data["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        _, _, width, height = annotation["bbox"]
        counts[size_bin(math.sqrt(max(width * height, 0.0)))] += 1
        total += 1
    return counts, total


def draw_sample(image: torch.Tensor, target: dict, destination: Path) -> Image.Image:
    array = (
        image.detach().cpu().clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).numpy()
    )
    canvas = Image.fromarray(array)
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    mixup = target.get("mixup")
    for index, box in enumerate(target["boxes"].detach().cpu()):
        cx, cy, bw, bh = [float(value) for value in box]
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        color = (255, 215, 0)
        if mixup is not None and float(mixup[index]) < 0.5:
            color = (0, 255, 255)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=90, method=6)
    return canvas


def make_contact_sheet(images: list[Image.Image], destination: Path) -> None:
    if not images:
        return
    thumb_width, thumb_height = 320, 256
    columns = 4
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), color=(24, 24, 24))
    for index, image in enumerate(images):
        thumb = image.resize((thumb_width, thumb_height), Image.Resampling.BILINEAR)
        sheet.paste(thumb, ((index % columns) * thumb_width, (index // columns) * thumb_height))
    sheet.save(destination, quality=90, method=6)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = YAMLConfig(args.config, use_amp=True, seed=args.seed)
    config.yaml_cfg["train_dataloader"]["num_workers"] = 0
    dataloader = config.train_dataloader
    dataloader.set_epoch(args.epoch)

    train_cfg = config.yaml_cfg["train_dataloader"]
    annotation_path = Path(train_cfg["dataset"]["ann_file"])
    original_counts, original_total = load_original_distribution(annotation_path)

    augmented_counts: Counter = Counter()
    target_count_histogram: Counter = Counter()
    invalid = Counter()
    empty_samples = 0
    mixup_samples = 0
    processed = 0
    augmented_targets = 0
    visualization_images: list[Image.Image] = []

    for images, targets in dataloader:
        for image, target in zip(images, targets):
            if processed >= args.samples:
                break
            boxes = target["boxes"].detach().cpu()
            labels = target["labels"].detach().cpu()
            count = len(labels)
            target_count_histogram[str(count)] += 1
            augmented_targets += count
            empty_samples += int(count == 0)
            mixup_samples += int("mixup" in target)

            if boxes.ndim != 2 or boxes.shape[-1] != 4:
                invalid["box_shape"] += 1
            if len(boxes) != count:
                invalid["box_label_mismatch"] += 1
            for key in ("area", "iscrowd", "mixup"):
                if key in target and len(target[key]) != count:
                    invalid[f"{key}_length_mismatch"] += 1

            if boxes.numel():
                if not torch.isfinite(boxes).all():
                    invalid["non_finite_box"] += int((~torch.isfinite(boxes)).any(dim=1).sum())
                cx, cy, width, height = boxes.unbind(-1)
                x1, y1 = cx - width / 2, cy - height / 2
                x2, y2 = cx + width / 2, cy + height / 2
                invalid["non_positive_size"] += int(((width <= 0) | (height <= 0)).sum())
                invalid["out_of_bounds"] += int(
                    ((x1 < -1e-6) | (y1 < -1e-6) | (x2 > 1 + 1e-6) | (y2 > 1 + 1e-6)).sum()
                )
                pixel_sizes = torch.sqrt((width * images.shape[-1]) * (height * images.shape[-2]))
                for value in pixel_sizes.tolist():
                    augmented_counts[size_bin(float(value))] += 1

            if not torch.isfinite(image).all():
                invalid["non_finite_image"] += 1
            if float(image.min()) < -1e-6 or float(image.max()) > 1 + 1e-6:
                invalid["image_range"] += 1

            if processed < args.visualizations:
                visualization_images.append(
                    draw_sample(
                        image,
                        target,
                        output / "visualizations" / f"sample_{processed + 1:04d}.webp",
                    )
                )
            processed += 1
        if processed >= args.samples:
            break

    if processed != args.samples:
        raise RuntimeError(f"Expected {args.samples} samples, received {processed}")

    original_props = proportions(original_counts, original_total)
    augmented_props = proportions(augmented_counts, augmented_targets)
    invalid_total = sum(invalid.values())
    mean_targets = augmented_targets / processed
    empty_fraction = empty_samples / processed
    lt8_increase = augmented_props["lt8"] - original_props["lt8"]
    gates = {
        "processed_exactly_1000": processed == 1000,
        "no_invalid_boxes_or_images": invalid_total == 0,
        "mean_targets_at_least_2": mean_targets >= 2.0,
        "empty_sample_fraction_at_most_10pct": empty_fraction <= 0.10,
        "lt8_fraction_at_most_15pct": augmented_props["lt8"] <= 0.15,
        "lt8_increase_at_most_10_points": lt8_increase <= 0.10,
        "visualizations_complete": len(visualization_images) == args.visualizations,
    }

    try:
        deim_commit = subprocess.check_output(
            ["git", "-C", str(Path("../_third_party/DEIM").resolve()), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        deim_commit = "unknown"

    summary = {
        "protocol": {
            "config": str(Path(args.config).resolve()),
            "epoch": args.epoch,
            "seed": args.seed,
            "samples": processed,
            "visualizations": len(visualization_images),
            "deim_commit": deim_commit,
        },
        "original": {
            "targets": original_total,
            "size_bins": dict(original_counts),
            "size_proportions": original_props,
        },
        "augmented": {
            "targets": augmented_targets,
            "mean_targets_per_sample": mean_targets,
            "empty_samples": empty_samples,
            "empty_fraction": empty_fraction,
            "mixup_samples": mixup_samples,
            "size_bins": dict(augmented_counts),
            "size_proportions": augmented_props,
            "target_count_histogram": dict(sorted(target_count_histogram.items(), key=lambda x: int(x[0]))),
        },
        "invalid": dict(invalid),
        "invalid_total": invalid_total,
        "lt8_fraction_increase": lt8_increase,
        "gates": gates,
        "eligible_for_training": all(gates.values()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_contact_sheet(visualization_images, output / "contact_sheet_32.webp")

    rows = []
    for name, _, _ in BINS:
        rows.append(
            f"| {name} | {original_counts[name]} | {original_props[name]:.3%} | "
            f"{augmented_counts[name]} | {augmented_props[name]:.3%} |"
        )
    gate_rows = [f"- {'通过' if passed else '失败'}：`{name}`" for name, passed in gates.items()]
    report = f"""# L-DQ1 Dense O2O增强审计

## 结论

训练门：**{'通过' if summary['eligible_for_training'] else '失败'}**。

- 审计样本：{processed}
- 增强后目标总数：{augmented_targets}
- 平均目标数：{mean_targets:.4f}/图
- 空目标图：{empty_samples}（{empty_fraction:.3%}）
- MixUp样本：{mixup_samples}
- 非法项总数：{invalid_total}
- `<8`像素比例相对原始变化：{lt8_increase:+.3%}

## 尺寸分布

| sqrt面积区间 | 原始数量 | 原始占比 | 增强数量 | 增强占比 |
|---|---:|---:|---:|---:|
{os.linesep.join(rows)}

## 预注册门槛

{os.linesep.join(gate_rows)}

## 人工复核入口

- `contact_sheet_32.webp`
- `visualizations/sample_0001.webp`至`sample_{args.visualizations:04d}.webp`

黄色框与青色框表示MixUp中的两组目标。本报告只决定是否允许启动训练，不使用Val或Test调参。
"""
    (output / "report_zh.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
