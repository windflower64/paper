#!/usr/bin/env python3
"""Export separate, layout-free examples of severe S8->S16 edge attenuation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_s8_s16_edge_loss import (  # noqa: E402
    S8S16Capture,
    edge_energy,
    separation,
    side_points,
    square_crop,
)


def joint_normalize(first, second, lower=2.0, upper=98.0):
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    joined = np.concatenate([first.ravel(), second.ravel()])
    low, high = np.percentile(joined, [lower, upper], method="nearest")
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(first), np.zeros_like(second)

    def scale(values):
        return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)

    return scale(first), scale(second)


def feature_crop(feature_map, image_crop, image_width, image_height):
    left, top, right, bottom = image_crop
    feature_height, feature_width = feature_map.shape
    x1 = max(0, int(math.floor(left / image_width * feature_width)))
    y1 = max(0, int(math.floor(top / image_height * feature_height)))
    x2 = min(feature_width, int(math.ceil(right / image_width * feature_width)))
    y2 = min(feature_height, int(math.ceil(bottom / image_height * feature_height)))
    return feature_map[y1:y2, x1:x2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scan-images", type=int, default=400)
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--output-size", type=int, default=512)
    return parser.parse_args()


def scan_severe_examples(model, loader, scan_images, num_examples):
    capture = S8S16Capture(model.backbone)
    candidates = []
    try:
        with torch.inference_mode():
            for batch_index, (samples, targets) in enumerate(loader):
                if batch_index >= scan_images:
                    break
                samples = samples.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    model.backbone(samples)
                height, width = samples.shape[-2:]
                for object_index, box_tensor in enumerate(targets[0]["boxes"]):
                    box = box_tensor.detach().float().cpu().tolist()
                    geometric_size = math.sqrt(max(0.0, (box[2] - box[0]) * (box[3] - box[1])))
                    if not 16 <= geometric_size < 32:
                        continue
                    points = side_points(box, height, width)
                    if len(points) != 4:
                        continue
                    before_values = [separation(capture.before, inside, outside) for inside, outside in points]
                    after_values = [separation(capture.after, inside, outside) for inside, outside in points]
                    before_mean = float(np.mean(before_values))
                    after_mean = float(np.mean(after_values))
                    retention = math.exp(
                        np.mean(np.log(np.maximum(after_values, 1e-8)))
                        - np.mean(np.log(np.maximum(before_values, 1e-8)))
                    )
                    if before_mean < 0.35 or retention >= 0.55:
                        continue
                    candidates.append(
                        {
                            "image_id": int(targets[0]["image_id"].item()),
                            "image_path": str(targets[0]["image_path"]),
                            "object_index": object_index,
                            "box": box,
                            "input": samples.detach().float().cpu(),
                            "s8": capture.before.detach().float().cpu(),
                            "s16": capture.after.detach().float().cpu(),
                            "geometric_size": geometric_size,
                            "s8_separation": before_mean,
                            "s16_separation": after_mean,
                            "retention": retention,
                        }
                    )
                    candidates.sort(key=lambda row: row["retention"])
                    candidates = candidates[: max(24, num_examples * 6)]
    finally:
        capture.close()

    selected = []
    used_images = set()
    for candidate in sorted(candidates, key=lambda row: row["retention"]):
        if candidate["image_id"] in used_images:
            continue
        selected.append(candidate)
        used_images.add(candidate["image_id"])
        if len(selected) == num_examples:
            break
    if len(selected) < num_examples:
        raise RuntimeError(f"Only found {len(selected)} qualifying examples; requested {num_examples}.")
    return selected


def resize_scalar_map(values, size):
    values = np.asarray(values, dtype=np.float32)
    rgba = matplotlib.colormaps["magma"](values, bytes=True)
    return Image.fromarray(rgba[:, :, :3], mode="RGB").resize((size, size), Image.Resampling.NEAREST)


def export_example(record, output_dir, output_size):
    image = (record["input"][0].permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
    height, width = image.shape[:2]
    crop = square_crop(record["box"], width, height, scale=5.0)
    left, top, right, bottom = crop
    original = Image.fromarray(image, mode="RGB").crop(crop).resize((output_size, output_size), Image.Resampling.BICUBIC)
    original_gt = original.copy()
    draw = ImageDraw.Draw(original_gt)
    sx = output_size / (right - left)
    sy = output_size / (bottom - top)
    box = record["box"]
    gt = [(box[0] - left) * sx, (box[1] - top) * sy, (box[2] - left) * sx, (box[3] - top) * sy]
    line_width = max(3, output_size // 128)
    draw.rectangle(gt, outline=(0, 229, 255), width=line_width)

    s8_native = feature_crop(edge_energy(record["s8"]), crop, width, height)
    s16_native = feature_crop(edge_energy(record["s16"]), crop, width, height)
    s8_norm, s16_norm = joint_normalize(s8_native, s16_native)

    output_dir.mkdir(parents=True, exist_ok=True)
    original.save(output_dir / "01_original.png")
    original_gt.save(output_dir / "02_original_gt.png")
    resize_scalar_map(s8_norm, output_size).save(output_dir / "03_S8_edge_energy.png")
    resize_scalar_map(s16_norm, output_size).save(output_dir / "04_S16_edge_energy.png")

    metadata = {
        "image_id": record["image_id"],
        "image_path": record["image_path"],
        "object_index": record["object_index"],
        "box_xyxy_at_640x512": record["box"],
        "geometric_size_px": record["geometric_size"],
        "s8_target_grid_wh": [(box[2] - box[0]) / 8, (box[3] - box[1]) / 8],
        "s16_target_grid_wh": [(box[2] - box[0]) / 16, (box[3] - box[1]) / 16],
        "s8_side_separation_mean": record["s8_separation"],
        "s16_side_separation_mean": record["s16_separation"],
        "retention": record["retention"],
        "loss_percent": (1.0 - record["retention"]) * 100.0,
        "visual_scale": "S8 and S16 share one percentile color scale within this example.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main():
    args = parse_args()
    sys.path.insert(0, str(args.repo))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = 1
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    model = cfg.model.cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = checkpoint.get("ema", {}).get("module")
    weight_source = "ema.module"
    if weights is None:
        weights = checkpoint.get("model", checkpoint)
        weight_source = "model_or_raw"
    model.load_state_dict(weights, strict=True)

    examples = scan_severe_examples(model, cfg.val_dataloader, args.scan_images, args.num_examples)
    all_metadata = []
    for index, example in enumerate(examples, start=1):
        metadata = export_example(example, args.output_dir / f"group_{index:02d}", args.output_size)
        metadata["group"] = index
        all_metadata.append(metadata)
    summary = {
        "model": "A00 visible D-FINE-N",
        "checkpoint": str(args.checkpoint),
        "weight_source": weight_source,
        "split": "Val",
        "selection_scope": f"first {args.scan_images} validation images",
        "selection_rule": "Distinct images; 16-32 px targets; S8 separation >= 0.35; retention < 0.55; sorted by lowest retention.",
        "interpretation": "These are deliberately selected severe attenuation examples, not an estimate of average prevalence.",
        "whole_val_reference": {"s8_to_s16_delta_ap75_when_aligned_edge_detail_is_removed": -0.03292023598175098},
        "examples": all_metadata,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "README_数据说明.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
