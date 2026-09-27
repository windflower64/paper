#!/usr/bin/env python3
"""Visualize an aggressive, frozen-weight BPC1 intervention sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_bpc1_sam_carrier_overview import (  # noqa: E402
    Capture,
    boundary_overlay,
    draw_gt,
    load_model,
    load_samples,
    random_records,
)
from export_random_multistage_overlays import (  # noqa: E402
    activation_overlay,
    joint_normalize_maps,
    resize_heatmap_smooth,
)
from export_s8_s16_edge_loss_examples import feature_crop  # noqa: E402
from visualize_s8_s16_edge_loss import edge_energy, square_crop  # noqa: E402


SETTINGS = (
    ("关闭", 0.0, 0.0, False),
    ("原强度1×", 1.0, 0.0, False),
    ("放大3.2×", 3.2, 0.0, False),
    ("放大6.4×", 6.4, 0.0, False),
    ("放大12.8×", 12.8, 0.0, False),
    ("12.8×+门偏置4+取消tanh", 12.8, 4.0, True),
)


def font(size):
    for candidate in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def capture_sweep(model, samples):
    branch = model.backbone.bpc_branch
    old = (
        branch.intervention_output_gain,
        branch.intervention_gate_logit_bias,
        branch.intervention_linear_residual,
    )
    capture = Capture(model.backbone)
    results = []
    try:
        with torch.inference_mode():
            for item in samples:
                tensor = item["tensor"].cuda(non_blocking=True)
                sample = {}
                for name, gain, bias, linear in SETTINGS:
                    branch.intervention_output_gain = gain
                    branch.intervention_gate_logit_bias = bias
                    branch.intervention_linear_residual = linear
                    with torch.autocast("cuda", dtype=torch.float16):
                        model.backbone(tensor)
                    sample[name] = {
                        "standard": capture.standard_s16.cpu().clone(),
                        "enhanced": capture.enhanced_s16.cpu().clone(),
                        "gate_mean": float(branch.last_gate.mean()),
                        "residual_rms_ratio": float(branch.last_residual_rms_ratio),
                    }
                results.append(sample)
    finally:
        (
            branch.intervention_output_gain,
            branch.intervention_gate_logit_bias,
            branch.intervention_linear_residual,
        ) = old
        capture.close()
    return results


def full_image_panel(image, crop, size):
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    draw.rectangle(crop, outline=(255, 190, 0), width=max(3, min(image.size) // 128))
    fitted = ImageOps.contain(marked, (size, size), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size, size), (238, 240, 244))
    panel.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return panel


def overlay(crop_image, normalized, size, alpha=0.80):
    heat = resize_heatmap_smooth(normalized, size, size)
    return Image.fromarray(
        activation_overlay(np.asarray(crop_image), heat, max_alpha=alpha), mode="RGB"
    )


def delta_map(enhanced, standard):
    return (enhanced - standard).square().mean(1).sqrt()[0].numpy()


def sample_panels(item, result, size):
    image, mask = item["image"], item["mask"]
    width, height = image.size
    box = item["record"]["bbox_xyxy"]
    crop = square_crop(box, width, height, scale=5.5)
    crop_image = image.crop(crop).resize((size, size), Image.Resampling.BICUBIC)
    crop_gt = draw_gt(crop_image, box, crop)
    mask_crop = mask.crop(crop).resize((size, size), Image.Resampling.NEAREST)
    sam = draw_gt(boundary_overlay(crop_image, mask_crop), box, crop)

    standard = result["关闭"]["standard"]
    feature_tensors = [standard] + [result[name]["enhanced"] for name, *_ in SETTINGS[2:]]
    feature_maps = [feature_crop(edge_energy(x), crop, width, height) for x in feature_tensors]
    feature_norm = joint_normalize_maps(feature_maps)
    feature_panels = [draw_gt(overlay(crop_image, x, size), box, crop) for x in feature_norm]

    delta_names = [name for name, *_ in SETTINGS[1:]]
    deltas = [
        feature_crop(delta_map(result[name]["enhanced"], standard), crop, width, height)
        for name in delta_names
    ]
    vmax = float(np.percentile(deltas[-1], 99.0))
    if not np.isfinite(vmax) or vmax <= 0:
        delta_norm = [np.zeros_like(x) for x in deltas]
    else:
        # A single logarithmic display transform reveals the weak settings
        # while retaining one common scale and the correct ordering.
        delta_norm = [np.log1p(30.0 * np.clip(x / vmax, 0, None)) / np.log1p(30.0) for x in deltas]
        delta_norm = [np.clip(x, 0.0, 1.0) for x in delta_norm]
    delta_panels = [draw_gt(overlay(crop_image, x, size, 0.88), box, crop) for x in delta_norm]

    feature_row = [full_image_panel(image, crop, size), crop_gt, sam, *feature_panels]
    delta_row = [full_image_panel(image, crop, size), crop_gt, sam, *delta_panels]
    return feature_row, delta_row


def compose(rows, headers, output, size, footer):
    header_h, footer_h, label_w, gap = 72, 82, 88, 9
    width = label_w + len(headers) * size + (len(headers) - 1) * gap
    height = header_h + len(rows) * size + (len(rows) - 1) * gap + footer_h
    canvas = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(canvas)
    for col, title in enumerate(headers):
        x = label_w + col * (size + gap) + size // 2
        draw.text((x, header_h // 2), title, fill=(23, 35, 60), font=font(21), anchor="mm")
    for row_index, panels in enumerate(rows):
        y = header_h + row_index * (size + gap)
        draw.text((label_w // 2, y + size // 2), f"样本{row_index + 1}", fill=(55, 68, 86), font=font(20), anchor="mm")
        for col, panel in enumerate(panels):
            canvas.paste(panel, (label_w + col * (size + gap), y))
    draw.text((width // 2, height - footer_h // 2), footer, fill=(174, 43, 54), font=font(20), anchor="mm")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--panel-size", type=int, default=255)
    return parser.parse_args()


def main():
    args = parse_args()
    records = json.loads(args.records.read_text(encoding="utf-8"))
    selected = random_records(records, args.count, args.seed)
    samples = load_samples(selected, args.image_root, args.mask_root)
    model, weight_source = load_model(args.repo, args.config, args.checkpoint)
    results = capture_sweep(model, samples)

    feature_rows, delta_rows = [], []
    metadata_samples = []
    for index, (item, result) in enumerate(zip(samples, results), start=1):
        feature_row, delta_row = sample_panels(item, result, args.panel_size)
        feature_rows.append(feature_row)
        delta_rows.append(delta_row)
        metadata_samples.append(
            {
                "sample": index,
                "image_id": item["record"]["image_id"],
                "file_name": item["record"]["file_name"],
                "settings": {
                    name: {
                        "gate_mean": result[name]["gate_mean"],
                        "residual_rms_ratio": result[name]["residual_rms_ratio"],
                    }
                    for name, *_ in SETTINGS
                },
            }
        )

    feature_headers = ("完整原图", "目标裁剪", "SAM边界", "标准S16", "3.2×S16", "6.4×S16", "12.8×S16", "最激进S16")
    delta_headers = ("完整原图", "目标裁剪", "SAM边界", "1×增量", "3.2×增量", "6.4×增量", "12.8×增量", "最激进增量")
    feature_path = args.output_dir / "01_BPC1激进增强_S16主特征统一色标.png"
    delta_path = args.output_dir / "02_BPC1激进增强_增量统一对数色标.png"
    compose(
        feature_rows,
        feature_headers,
        feature_path,
        args.panel_size,
        "同一线性色标：强度越大，目标外高频也被带入；完整Val AP从关闭0.65193降至最激进0.63451",
    )
    compose(
        delta_rows,
        delta_headers,
        delta_path,
        args.panel_size,
        "同一log1p色标（仅为看清弱增量）：12.8×及门放宽后响应明显扩散，AP与AP75同步下降",
    )
    metadata = {
        "checkpoint": str(args.checkpoint),
        "weight_source": weight_source,
        "selection": "five random accepted SAM samples; no metric or feature filtering",
        "seed": args.seed,
        "settings": [
            {"name": name, "output_gain": gain, "gate_logit_bias": bias, "linear_residual": linear}
            for name, gain, bias, linear in SETTINGS
        ],
        "visualization": {
            "S16": "one shared linear percentile scale per sample",
            "delta": "one shared log1p scale per sample, anchored to the aggressive setting",
        },
        "samples": metadata_samples,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "README_激进增强可视化说明.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"feature": str(feature_path), "delta": str(delta_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
