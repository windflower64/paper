#!/usr/bin/env python3
"""Export random, scale-controlled BPC1 feature heatmaps for presentation.

The standard, 1x, and 3.2x S16 panels share a per-sample color scale.  The
1x/3.2x residual panels also share a separate per-sample absolute color scale,
so the visual comparison preserves the true amplitude ratio.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_bpc1_sam_carrier_overview import (  # noqa: E402
    Capture,
    boundary_overlay,
    draw_gt,
    feature_crop,
    load_model,
    load_samples,
    random_records,
    robust_unit,
)
from export_random_multistage_overlays import (  # noqa: E402
    activation_overlay,
    joint_normalize_maps,
    resize_heatmap_smooth,
)
from visualize_s8_s16_edge_loss import edge_energy, square_crop  # noqa: E402


HEADERS = (
    "完整原图",
    "目标裁剪",
    "SAM边界教师",
    "关闭BPC：标准S16",
    "原强度1×：增强S16",
    "放大3.2×：增强S16",
    "1×细节增量",
    "3.2×细节增量",
)


def font(size: int):
    for candidate in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def set_effective_scale(branch, target_scale: float) -> None:
    if target_scale < 0 or target_scale >= branch.max_scale:
        raise ValueError(
            f"target scale must be in [0, {branch.max_scale}), got {target_scale}"
        )
    if target_scale == 0:
        value = -20.0
    else:
        value = math.atanh(target_scale / branch.max_scale)
    with torch.no_grad():
        branch.scale_logit.copy_(
            torch.tensor(value, dtype=branch.scale_logit.dtype, device=branch.scale_logit.device)
        )


def capture_at_scales(model, samples, multipliers=(0.0, 1.0, 3.2)):
    branch = model.backbone.bpc_branch
    learned_scale = float(branch.max_scale * torch.tanh(branch.scale_logit).detach())
    capture = Capture(model.backbone)
    all_results = []
    try:
        with torch.inference_mode():
            for item in samples:
                sample_result = {}
                tensor = item["tensor"].cuda(non_blocking=True)
                for multiplier in multipliers:
                    target_scale = min(
                        learned_scale * multiplier,
                        float(branch.max_scale) * (1.0 - 1e-6),
                    )
                    set_effective_scale(branch, target_scale)
                    with torch.autocast("cuda", dtype=torch.float16):
                        model.backbone(tensor)
                    sample_result[str(multiplier)] = {
                        "standard": capture.standard_s16.cpu().clone(),
                        "enhanced": capture.enhanced_s16.cpu().clone(),
                        "gate": branch.last_gate.detach().float().cpu()[0, 0].numpy(),
                        "effective_scale": float(branch.last_scale),
                        "residual_rms_ratio": float(branch.last_residual_rms_ratio),
                    }
                all_results.append(sample_result)
    finally:
        set_effective_scale(branch, learned_scale)
        capture.close()
    return learned_scale, all_results


def full_image_panel(image: Image.Image, crop, size: int) -> Image.Image:
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    draw.rectangle(crop, outline=(255, 190, 0), width=max(3, min(image.size) // 128))
    fitted = ImageOps.contain(marked, (size, size), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size, size), (238, 240, 244))
    panel.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return panel


def heat_overlay(crop_image, normalized, size, max_alpha=0.78):
    heat = resize_heatmap_smooth(normalized, size, size)
    return Image.fromarray(
        activation_overlay(np.asarray(crop_image), heat, max_alpha=max_alpha), mode="RGB"
    )


def rms_delta(enhanced, standard):
    return (
        (enhanced - standard).square().mean(1).sqrt()[0].numpy().astype(np.float32)
    )


def make_row(item, result, size):
    image = item["image"]
    mask = item["mask"]
    width, height = image.size
    box = item["record"]["bbox_xyxy"]
    crop = square_crop(box, width, height, scale=5.5)
    crop_image = image.crop(crop).resize((size, size), Image.Resampling.BICUBIC)
    crop_gt = draw_gt(crop_image, box, crop)
    mask_crop = mask.crop(crop).resize((size, size), Image.Resampling.NEAREST)
    sam_boundary = draw_gt(boundary_overlay(crop_image, mask_crop), box, crop)

    standard = result["0.0"]["standard"]
    enhanced_1x = result["1.0"]["enhanced"]
    enhanced_32x = result["3.2"]["enhanced"]
    feature_maps = [
        feature_crop(edge_energy(tensor), crop, width, height)
        for tensor in (standard, enhanced_1x, enhanced_32x)
    ]
    normalized_features = joint_normalize_maps(feature_maps)
    feature_panels = [
        draw_gt(heat_overlay(crop_image, normalized, size), box, crop)
        for normalized in normalized_features
    ]

    deltas = [
        feature_crop(rms_delta(enhanced_1x, standard), crop, width, height),
        feature_crop(rms_delta(enhanced_32x, standard), crop, width, height),
    ]
    # One absolute scale for both residual panels.  Unlike independent robust
    # normalization, this keeps the 1x panel visibly weaker than the 3.2x one.
    vmax = float(np.percentile(deltas[1], 99.0))
    if not np.isfinite(vmax) or vmax <= 0:
        normalized_deltas = [np.zeros_like(delta) for delta in deltas]
    else:
        normalized_deltas = [np.clip(delta / vmax, 0.0, 1.0) for delta in deltas]
    delta_panels = [
        draw_gt(heat_overlay(crop_image, normalized, size, max_alpha=0.86), box, crop)
        for normalized in normalized_deltas
    ]

    return [
        full_image_panel(image, crop, size),
        crop_gt,
        sam_boundary,
        *feature_panels,
        *delta_panels,
    ]


def make_overview(rows, output_path: Path, panel_size: int) -> None:
    header_height, footer_height, label_width, gap = 76, 82, 92, 9
    width = label_width + len(HEADERS) * panel_size + (len(HEADERS) - 1) * gap
    height = header_height + len(rows) * panel_size + (len(rows) - 1) * gap + footer_height
    canvas = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(canvas)
    header_font, row_font, note_font = font(22), font(21), font(21)
    for column, title in enumerate(HEADERS):
        x = label_width + column * (panel_size + gap) + panel_size // 2
        draw.text((x, header_height // 2), title, fill=(23, 35, 60), font=header_font, anchor="mm")
    for row_index, images in enumerate(rows):
        y = header_height + row_index * (panel_size + gap)
        draw.text(
            (label_width // 2, y + panel_size // 2),
            f"样本{row_index + 1}",
            fill=(55, 68, 86),
            font=row_font,
            anchor="mm",
        )
        for column, panel in enumerate(images):
            x = label_width + column * (panel_size + gap)
            canvas.paste(panel, (x, y))
    note_y = height - footer_height // 2
    note = (
        "同一色标对比｜3.2×相对关闭：AP −0.000473，AP75 +0.003029，"
        "8–16像素目标AP −0.008284（定位变严，但总体稳定性下降）"
    )
    draw.text((width // 2, note_y), note, fill=(174, 43, 54), font=note_font, anchor="mm")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


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
    parser.add_argument("--panel-size", type=int, default=280)
    return parser.parse_args()


def main():
    args = parse_args()
    records = json.loads(args.records.read_text(encoding="utf-8"))
    selected = random_records(records, args.count, args.seed)
    samples = load_samples(selected, args.image_root, args.mask_root)
    model, weight_source = load_model(args.repo, args.config, args.checkpoint)
    learned_scale, results = capture_at_scales(model, samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    metadata_samples = []
    names = (
        "00_完整原图.png",
        "01_目标裁剪_GT框.png",
        "02_SAM边界教师.png",
        "03_关闭BPC_标准S16.png",
        "04_原强度1x_增强S16.png",
        "05_放大3p2x_增强S16.png",
        "06_原强度1x_细节增量.png",
        "07_放大3p2x_细节增量.png",
    )
    for index, (item, result) in enumerate(zip(samples, results), start=1):
        row = make_row(item, result, args.panel_size)
        rows.append(row)
        sample_dir = args.output_dir / f"sample_{index:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for name, panel in zip(names, row):
            panel.save(sample_dir / name)
        metadata_samples.append(
            {
                "sample": index,
                "image_id": item["record"]["image_id"],
                "annotation_id": item["record"]["annotation_id"],
                "file_name": item["record"]["file_name"],
                "bbox_xyxy": item["record"]["bbox_xyxy"],
                "pred_iou": item["record"].get("pred_iou"),
                "effective_scale_1x": result["1.0"]["effective_scale"],
                "effective_scale_3p2x": result["3.2"]["effective_scale"],
                "carrier_rms_ratio_1x": result["1.0"]["residual_rms_ratio"],
                "carrier_rms_ratio_3p2x": result["3.2"]["residual_rms_ratio"],
            }
        )

    overview = args.output_dir / "随机5例_BPC1增强幅度与失败现象总览.png"
    make_overview(rows, overview, args.panel_size)
    metadata = {
        "model": "S-BPC1 epoch56 EMA",
        "checkpoint": str(args.checkpoint),
        "weight_source": weight_source,
        "selection": "random accepted SAM records, distinct images, no feature/metric filtering",
        "seed": args.seed,
        "learned_scale": learned_scale,
        "comparison": {
            "off": 0.0,
            "one_x": learned_scale,
            "three_point_two_x": min(3.2 * learned_scale, 0.25 * (1.0 - 1e-6)),
        },
        "full_val_metrics": {
            "off": {"AP": 0.6519314873, "AP75": 0.7911954137, "AP_8to16": 0.3350582665},
            "three_point_two_x": {
                "AP": 0.6514583645,
                "AP75": 0.7942245546,
                "AP_8to16": 0.3267747294,
            },
        },
        "normalization": {
            "s16_panels": "standard/1x/3.2x share one percentile color scale within each sample",
            "delta_panels": "1x/3.2x share the 99th percentile of the 3.2x absolute RMS delta within each sample",
            "warning": "heatmaps show response location; AP failure is established by the recorded full-Val metrics, not by cherry-picked images",
        },
        "samples": metadata_samples,
    }
    (args.output_dir / "README_可视化说明.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"overview": str(overview), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
