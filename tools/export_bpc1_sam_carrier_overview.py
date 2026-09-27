#!/usr/bin/env python3
"""Export a PPT-ready overview of the trained BPC1 SAM boundary carrier."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_random_multistage_overlays import (  # noqa: E402
    activation_overlay,
    joint_normalize_maps,
    resize_heatmap_smooth,
)
from export_s8_s16_edge_loss_examples import feature_crop  # noqa: E402
from visualize_s8_s16_edge_loss import edge_energy, square_crop  # noqa: E402


PRESENTATION_HEADERS = (
    "完整原图",
    "局部裁剪",
    "SAM边界教师",
    "S8压缩前",
    "标准S16",
    "学生门控",
    "BPC补偿增量",
)


def font(size):
    for candidate in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def robust_unit(values, lower=2.0, upper=98.0):
    values = np.asarray(values, dtype=np.float32)
    low, high = np.percentile(values, [lower, upper], method="nearest")
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def random_records(records, count, seed):
    candidates = [row for row in records if row.get("accepted", False)]
    random.Random(seed).shuffle(candidates)
    chosen, image_ids = [], set()
    for row in candidates:
        if row["image_id"] in image_ids:
            continue
        chosen.append(row)
        image_ids.add(row["image_id"])
        if len(chosen) == count:
            return chosen
    raise RuntimeError(f"Only {len(chosen)} distinct accepted samples found")


def records_from_manifest(records, manifest):
    """Select records in the exact order recorded by a prior visualization."""
    selected = []
    for sample in manifest["samples"]:
        matches = [
            row
            for row in records
            if int(row["image_id"]) == int(sample["image_id"])
            and (
                "annotation_id" not in sample
                or int(row.get("annotation_id", -1)) == int(sample["annotation_id"])
            )
        ]
        if not matches:
            raise RuntimeError(
                f"Manifest sample image_id={sample['image_id']} was not found"
            )
        selected.append(matches[0])
    return selected


def mask_name(record):
    recorded = record.get("mask_path")
    if recorded:
        return str(recorded).replace("\\", "/").rsplit("/", 1)[-1]
    return f"{int(record['image_id']):06d}.png"


def load_samples(records, image_root, mask_root):
    result = []
    for record in records:
        image = Image.open(image_root / record["file_name"]).convert("RGB")
        mask = Image.open(mask_root / mask_name(record)).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        tensor = (
            torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        )
        result.append({"record": record, "image": image, "mask": mask, "tensor": tensor})
    return result


def load_model(repo, config_path, checkpoint_path):
    sys.path.insert(0, str(repo))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(config_path))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model.cuda().eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = checkpoint.get("ema", {}).get("module")
    source = "ema.module"
    if weights is None:
        weights = checkpoint.get("model", checkpoint)
        source = "model_or_raw"
    model.load_state_dict(weights, strict=True)
    return model, source


class Capture:
    def __init__(self, backbone):
        self.s8 = None
        self.standard_s16 = None
        self.enhanced_s16 = None
        self.handles = [
            backbone.stages[1].register_forward_hook(self._s8),
            backbone.stages[2].register_forward_hook(self._standard),
            backbone.bpc_branch.register_forward_hook(self._enhanced),
        ]

    def _s8(self, module, inputs, output):
        self.s8 = output.detach().float()

    def _standard(self, module, inputs, output):
        self.standard_s16 = output.detach().float()

    def _enhanced(self, module, inputs, output):
        self.enhanced_s16 = output.detach().float()

    def close(self):
        for handle in self.handles:
            handle.remove()


def capture_features(model, samples):
    capture = Capture(model.backbone)
    results = []
    try:
        with torch.inference_mode():
            for item in samples:
                tensor = item["tensor"].cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    model.backbone(tensor)
                branch = model.backbone.bpc_branch
                results.append(
                    {
                        "s8": capture.s8.cpu(),
                        "standard_s16": capture.standard_s16.cpu(),
                        "enhanced_s16": capture.enhanced_s16.cpu(),
                        "gate": branch.last_gate.detach().float().cpu()[0, 0].numpy(),
                        "scale": float(branch.last_scale),
                        "residual_rms_ratio": float(branch.last_residual_rms_ratio),
                    }
                )
    finally:
        capture.close()
    return results


def full_image_with_crop(image, crop, size):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle(crop, outline=(255, 196, 0), width=max(3, min(result.size) // 128))
    contained = ImageOps.contain(result, (size, size), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size, size), (238, 240, 244))
    panel.paste(contained, ((size - contained.width) // 2, (size - contained.height) // 2))
    return panel


def draw_gt(image, box, crop):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    left, top, right, bottom = crop
    sx, sy = result.width / (right - left), result.height / (bottom - top)
    coords = (
        (box[0] - left) * sx,
        (box[1] - top) * sy,
        (box[2] - left) * sx,
        (box[3] - top) * sy,
    )
    draw.rectangle(coords, outline=(0, 229, 255), width=max(3, result.width // 128))
    return result


def boundary_overlay(crop_image, mask_crop):
    binary = np.asarray(mask_crop) > 0
    mask_image = Image.fromarray((binary * 255).astype(np.uint8), mode="L")
    dilated = np.asarray(mask_image.filter(ImageFilter.MaxFilter(9))) > 0
    eroded = np.asarray(mask_image.filter(ImageFilter.MinFilter(9))) > 0
    boundary = dilated ^ eroded
    image = np.asarray(crop_image, dtype=np.uint8).astype(np.float32)
    color = np.array([255, 210, 20], dtype=np.float32)
    image[boundary] = 0.25 * image[boundary] + 0.75 * color
    return Image.fromarray(np.clip(np.rint(image), 0, 255).astype(np.uint8), mode="RGB")


def overlay_map(crop_image, native_map, size, max_alpha=0.72):
    normalized = robust_unit(native_map)
    heat = resize_heatmap_smooth(normalized, size, size)
    return Image.fromarray(
        activation_overlay(np.asarray(crop_image), heat, max_alpha=max_alpha), mode="RGB"
    )


def make_row(item, features, size):
    image = item["image"]
    mask = item["mask"]
    width, height = image.size
    box = item["record"]["bbox_xyxy"]
    crop = square_crop(box, width, height, scale=5.5)
    crop_image = image.crop(crop).resize((size, size), Image.Resampling.BICUBIC)
    crop_gt = draw_gt(crop_image, box, crop)
    mask_crop = mask.crop(crop).resize((size, size), Image.Resampling.NEAREST)
    sam_boundary = draw_gt(boundary_overlay(crop_image, mask_crop), box, crop)

    s8_native = feature_crop(edge_energy(features["s8"]), crop, width, height)
    standard_native = feature_crop(edge_energy(features["standard_s16"]), crop, width, height)
    gate_native = feature_crop(features["gate"], crop, width, height)

    s8_view = draw_gt(overlay_map(crop_image, s8_native, size), box, crop)
    standard_s16 = draw_gt(overlay_map(crop_image, standard_native, size), box, crop)
    gate_view = draw_gt(overlay_map(crop_image, gate_native, size, max_alpha=0.82), box, crop)

    delta = (
        (features["enhanced_s16"] - features["standard_s16"])
        .square()
        .mean(1)
        .sqrt()[0]
        .numpy()
    )
    delta_native = feature_crop(delta, crop, width, height)
    delta_view = draw_gt(overlay_map(crop_image, delta_native, size, max_alpha=0.82), box, crop)
    return [
        full_image_with_crop(image, crop, size),
        crop_gt,
        sam_boundary,
        s8_view,
        standard_s16,
        gate_view,
        delta_view,
    ]


def make_overview(rows, output_path, panel_size):
    headers = PRESENTATION_HEADERS
    header_height, label_width, gap = 72, 92, 10
    width = label_width + len(headers) * panel_size + (len(headers) - 1) * gap
    height = header_height + len(rows) * panel_size + (len(rows) - 1) * gap
    canvas = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(canvas)
    header_font, row_font = font(25), font(22)
    for column, title in enumerate(headers):
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
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        help="Reuse samples, in order, from a previous visualization JSON file.",
    )
    parser.add_argument("--panel-size", type=int, default=320)
    return parser.parse_args()


def main():
    args = parse_args()
    records = json.loads(args.records.read_text(encoding="utf-8"))
    if args.sample_manifest:
        manifest = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
        selected = records_from_manifest(records, manifest)
    else:
        selected = random_records(records, args.count, args.seed)
    samples = load_samples(selected, args.image_root, args.mask_root)
    model, weight_source = load_model(args.repo, args.config, args.checkpoint)
    features = capture_features(model, samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, sample_metadata = [], []
    names = (
        "00_完整原图与裁剪框.png",
        "01_局部裁剪与GT框.png",
        "02_SAM边界教师.png",
        "03_S8压缩前特征.png",
        "04_标准S16特征.png",
        "05_学生门控.png",
        "06_BPC补偿增量_归一化放大.png",
    )
    for index, (item, feature) in enumerate(zip(samples, features), start=1):
        row = make_row(item, feature, args.panel_size)
        folder = args.output_dir / f"sample_{index:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        for name, panel in zip(names, row):
            panel.save(folder / name)
        rows.append(row)
        sample_metadata.append(
            {
                "sample": index,
                "image_id": item["record"]["image_id"],
                "annotation_id": item["record"]["annotation_id"],
                "file_name": item["record"]["file_name"],
                "bbox_xyxy": item["record"]["bbox_xyxy"],
                "pred_iou": item["record"].get("pred_iou"),
                "learned_scale": feature["scale"],
                "carrier_to_standard_rms": feature["residual_rms_ratio"],
            }
        )

    overview = args.output_dir / f"固定{len(rows)}例_BPC1_SAM边界补偿总览.png"
    make_overview(rows, overview, args.panel_size)
    metadata = {
        "model": "S-BPC1 SAM-supervised boundary polyphase carrier",
        "checkpoint": str(args.checkpoint),
        "weight_source": weight_source,
        "selection": (
            "samples reused in manifest order"
            if args.sample_manifest
            else "random accepted SAM records, distinct images, no feature/metric filtering"
        ),
        "seed": args.seed,
        "notes": {
            "standard_s16": "real Stage2 output before BPC residual",
            "student_gate": "single-channel learned boundary gate predicted from S8; SAM is not read at inference",
            "bpc_compensation_delta": "RMS norm of enhanced S16 minus standard S16, independently normalized for visibility",
            "warning": "the BPC compensation-delta panel is magnified by independent normalization and does not represent absolute amplitude",
        },
        "samples": sample_metadata,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
