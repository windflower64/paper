#!/usr/bin/env python3
"""Export random SAM masks and A00-vs-JointInit S16 feature comparisons."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_random_multistage_overlays import activation_overlay, joint_normalize_maps, resize_heatmap_smooth  # noqa: E402
from export_s8_s16_edge_loss_examples import feature_crop  # noqa: E402
from visualize_s8_s16_edge_loss import S8S16Capture, edge_energy, square_crop  # noqa: E402


def sam_mask_overlay(image, mask):
    image = np.asarray(image, dtype=np.uint8)
    binary = np.asarray(mask) > 0
    if not binary.any():
        return image.copy()
    result = image.astype(np.float32).copy()
    green = np.array([30, 210, 120], dtype=np.float32)
    result[binary] = 0.52 * result[binary] + 0.48 * green
    mask_image = Image.fromarray((binary * 255).astype(np.uint8), mode="L")
    dilated = np.asarray(mask_image.filter(ImageFilter.MaxFilter(5))) > 0
    eroded = np.asarray(mask_image.filter(ImageFilter.MinFilter(5))) > 0
    boundary = dilated ^ eroded
    result[boundary] = np.array([255, 220, 30], dtype=np.float32)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def difference_overlay(image, baseline, guided):
    image = np.asarray(image, dtype=np.uint8)
    difference = np.asarray(guided, dtype=np.float32) - np.asarray(baseline, dtype=np.float32)
    scale = float(np.percentile(np.abs(difference), 98, method="nearest"))
    if not np.isfinite(scale) or scale <= 1e-8:
        return image.copy()
    signed = np.clip(difference / scale, -1.0, 1.0)
    colors = matplotlib.colormaps["coolwarm"]((signed + 1.0) / 2.0, bytes=True)[..., :3].astype(np.float32)
    alpha = (0.72 * np.power(np.abs(signed), 0.65))[..., None]
    result = image.astype(np.float32) * (1.0 - alpha) + colors * alpha
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def random_accepted_records(records, count, seed):
    candidates = [row for row in records if row.get("accepted", False)]
    random.Random(seed).shuffle(candidates)
    selected, used_images = [], set()
    for row in candidates:
        if row["image_id"] in used_images:
            continue
        selected.append(row)
        used_images.add(row["image_id"])
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Only {len(selected)} distinct accepted SAM images found; requested {count}.")


def load_detector(repo, config_path, checkpoint_path):
    sys.path.insert(0, str(repo))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(config_path))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model.cuda().eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = checkpoint.get("ema", {}).get("module")
    if weights is None:
        weights = checkpoint.get("model", checkpoint)
    model.load_state_dict(weights, strict=True)
    return model


def mask_filename(record):
    recorded = record.get("mask_path")
    if recorded:
        return str(recorded).replace("\\", "/").rsplit("/", 1)[-1]
    return f"{int(record['image_id']):06d}.png"


def load_selected_images(records, image_root, mask_root):
    loaded = []
    for row in records:
        image = Image.open(image_root / row["file_name"]).convert("RGB")
        mask = Image.open(mask_root / mask_filename(row)).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1).unsqueeze(0) / 255.0
        loaded.append({"record": row, "image": image, "mask": mask, "tensor": tensor})
    return loaded


def capture_s16(model, loaded):
    capture = S8S16Capture(model.backbone)
    features = []
    try:
        with torch.inference_mode():
            for item in loaded:
                tensor = item["tensor"].cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    model.backbone(tensor)
                features.append(capture.after.detach().float().cpu())
    finally:
        capture.close()
    return features


def crop_assets(item, baseline_feature, guided_feature, output_size):
    image = np.asarray(item["image"], dtype=np.uint8)
    mask = np.asarray(item["mask"], dtype=np.uint8)
    height, width = image.shape[:2]
    box = item["record"]["bbox_xyxy"]
    crop = square_crop(box, width, height, scale=4.5)
    original = Image.fromarray(image, mode="RGB").crop(crop).resize((output_size, output_size), Image.Resampling.BICUBIC)
    mask_crop = Image.fromarray(mask, mode="L").crop(crop).resize((output_size, output_size), Image.Resampling.NEAREST)
    mask_visual = Image.fromarray(sam_mask_overlay(np.asarray(original), np.asarray(mask_crop)), mode="RGB")

    baseline_native = feature_crop(edge_energy(baseline_feature), crop, width, height)
    guided_native = feature_crop(edge_energy(guided_feature), crop, width, height)
    baseline_norm, guided_norm = joint_normalize_maps([baseline_native, guided_native])
    baseline_heat = resize_heatmap_smooth(baseline_norm, output_size, output_size)
    guided_heat = resize_heatmap_smooth(guided_norm, output_size, output_size)
    baseline_visual = Image.fromarray(activation_overlay(np.asarray(original), baseline_heat), mode="RGB")
    guided_visual = Image.fromarray(activation_overlay(np.asarray(original), guided_heat), mode="RGB")
    difference_visual = Image.fromarray(
        difference_overlay(np.asarray(original), baseline_heat, guided_heat), mode="RGB"
    )
    return [original, mask_visual, baseline_visual, guided_visual, difference_visual]


def chinese_font(size):
    for path in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def make_overview(rows, output_path, panel_size):
    headers = ("原图", "SAM掩码", "A00-S16", "SAM引导-S16", "差异图")
    header_height, row_label_width, gap = 70, 82, 10
    canvas = Image.new(
        "RGB",
        (row_label_width + len(headers) * panel_size + (len(headers) - 1) * gap, header_height + len(rows) * panel_size + (len(rows) - 1) * gap),
        (247, 248, 250),
    )
    draw = ImageDraw.Draw(canvas)
    header_font, row_font = chinese_font(28), chinese_font(22)
    for column, header in enumerate(headers):
        x = row_label_width + column * (panel_size + gap) + panel_size // 2
        draw.text((x, header_height // 2), header, fill=(23, 35, 60), font=header_font, anchor="mm")
    for row_index, images in enumerate(rows):
        y = header_height + row_index * (panel_size + gap)
        draw.text((row_label_width // 2, y + panel_size // 2), f"样本{row_index + 1}", fill=(55, 68, 86), font=row_font, anchor="mm")
        for column, image in enumerate(images):
            x = row_label_width + column * (panel_size + gap)
            canvas.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--guided-config", type=Path, required=True)
    parser.add_argument("--guided-checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--output-size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    records = json.loads(args.records.read_text(encoding="utf-8"))
    selected = random_accepted_records(records, args.count, args.seed)
    loaded = load_selected_images(selected, args.image_root, args.mask_root)

    baseline_model = load_detector(args.repo, args.baseline_config, args.baseline_checkpoint)
    baseline_features = capture_s16(baseline_model, loaded)
    del baseline_model
    torch.cuda.empty_cache()
    guided_model = load_detector(args.repo, args.guided_config, args.guided_checkpoint)
    guided_features = capture_s16(guided_model, loaded)
    del guided_model
    torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, provenance = [], []
    filenames = ("01_original.png", "02_sam_mask_overlay.png", "03_a00_s16.png", "04_sam_guided_s16.png", "05_guided_minus_a00.png")
    for index, (item, baseline, guided) in enumerate(zip(loaded, baseline_features, guided_features), start=1):
        images = crop_assets(item, baseline, guided, args.output_size)
        folder = args.output_dir / f"sample_{index:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        for filename, image in zip(filenames, images):
            image.save(folder / filename)
        rows.append(images)
        provenance.append(
            {
                "sample": index,
                "image_id": item["record"]["image_id"],
                "annotation_id": item["record"]["annotation_id"],
                "file_name": item["record"]["file_name"],
                "pred_iou": item["record"].get("pred_iou"),
                "stability_score": item["record"].get("stability_score"),
            }
        )
    overview = args.output_dir / "随机3例_SAM与S16特征对比总览.png"
    make_overview(rows, overview, args.output_size)
    metadata = {
        "selection": "random among accepted SAM records, distinct source images, no feature or metric filtering",
        "seed": args.seed,
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "guided_checkpoint": str(args.guided_checkpoint),
        "samples": provenance,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
