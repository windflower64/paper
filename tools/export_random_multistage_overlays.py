#!/usr/bin/env python3
"""Export five reproducibly random S4/S8/S16 feature-activation overlays."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_s8_s16_edge_loss_examples import feature_crop  # noqa: E402
from visualize_s8_s16_edge_loss import edge_energy, separation, side_points, square_crop  # noqa: E402


def activation_overlay(image, heat, max_alpha=0.68):
    image = np.asarray(image, dtype=np.uint8)
    heat = np.clip(np.asarray(heat, dtype=np.float32), 0.0, 1.0)
    color = matplotlib.colormaps["magma"](heat, bytes=True)[..., :3].astype(np.float32)
    alpha = (max_alpha * np.power(heat, 0.7))[..., None]
    blended = image.astype(np.float32) * (1.0 - alpha) + color * alpha
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def resize_heatmap_smooth(heat, width, height):
    heat = np.clip(np.asarray(heat, dtype=np.float32), 0.0, 1.0)
    image = Image.fromarray(heat, mode="F").resize((width, height), Image.Resampling.BICUBIC)
    return np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)


def full_context_with_crop_box(image, crop):
    result = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(result)
    width = max(3, min(result.size) // 128)
    draw.rectangle(crop, outline=(255, 196, 0), width=width)
    return result


def random_distinct_records(records, count, seed):
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    selected, used_images = [], set()
    for record in shuffled:
        if record["image_id"] in used_images:
            continue
        selected.append(record)
        used_images.add(record["image_id"])
        if len(selected) == count:
            return selected
    raise RuntimeError(f"Only {len(selected)} distinct images are available; requested {count}.")


def joint_normalize_maps(maps, lower=2.0, upper=98.0):
    joined = np.concatenate([np.asarray(values, dtype=np.float32).ravel() for values in maps])
    low, high = np.percentile(joined, [lower, upper], method="nearest")
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return [np.zeros_like(values, dtype=np.float32) for values in maps]
    return [np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32) for values in maps]


class MultistageCapture:
    def __init__(self, backbone):
        self.s4 = self.s8 = self.s16 = None
        s4_to_s8 = backbone.stages[1].downsample
        s8_to_s16 = backbone.stages[2].downsample
        self.handles = [
            s4_to_s8.register_forward_pre_hook(self._capture_s4),
            s4_to_s8.register_forward_hook(self._capture_s8),
            s8_to_s16.register_forward_hook(self._capture_s16),
        ]

    def _capture_s4(self, module, inputs):
        self.s4 = inputs[0].detach().float()

    def _capture_s8(self, module, inputs, output):
        self.s8 = output.detach().float()

    def _capture_s16(self, module, inputs, output):
        self.s16 = output.detach().float()

    def close(self):
        for handle in self.handles:
            handle.remove()


def collect_eligible_targets(loader):
    eligible = []
    for samples, targets in loader:
        for object_index, box_tensor in enumerate(targets[0]["boxes"]):
            box = box_tensor.detach().float().cpu().tolist()
            geometric_size = math.sqrt(max(0.0, (box[2] - box[0]) * (box[3] - box[1])))
            if 16 <= geometric_size < 32:
                eligible.append(
                    {
                        "image_id": int(targets[0]["image_id"].item()),
                        "image_path": str(targets[0]["image_path"]),
                        "object_index": object_index,
                        "box": box,
                        "geometric_size": geometric_size,
                    }
                )
    return eligible


def capture_selected(model, loader, selected):
    wanted = {row["image_id"]: row for row in selected}
    captured = {}
    hooks = MultistageCapture(model.backbone)
    try:
        with torch.inference_mode():
            for samples, targets in loader:
                image_id = int(targets[0]["image_id"].item())
                if image_id not in wanted:
                    continue
                samples = samples.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    model.backbone(samples)
                record = dict(wanted[image_id])
                record.update(
                    {
                        "input": samples.detach().float().cpu(),
                        "s4": hooks.s4.detach().float().cpu(),
                        "s8": hooks.s8.detach().float().cpu(),
                        "s16": hooks.s16.detach().float().cpu(),
                    }
                )
                captured[image_id] = record
                if len(captured) == len(wanted):
                    break
    finally:
        hooks.close()
    if len(captured) != len(wanted):
        raise RuntimeError("Not all randomly selected images were captured.")
    return [captured[row["image_id"]] for row in selected]


def draw_gt(image, box, crop):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    left, top, right, bottom = crop
    sx, sy = result.width / (right - left), result.height / (bottom - top)
    rectangle = [(box[0] - left) * sx, (box[1] - top) * sy, (box[2] - left) * sx, (box[3] - top) * sy]
    draw.rectangle(rectangle, outline=(0, 229, 255), width=max(3, result.width // 128))
    return result


def make_stage_overlays(record, output_size):
    image = (record["input"][0].permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
    height, width = image.shape[:2]
    crop = square_crop(record["box"], width, height, scale=6.0)
    full_context = full_context_with_crop_box(image, crop)
    original = Image.fromarray(image, mode="RGB").crop(crop).resize((output_size, output_size), Image.Resampling.BICUBIC)

    native_maps = [feature_crop(edge_energy(record[stage]), crop, width, height) for stage in ("s4", "s8", "s16")]
    normalized_maps = joint_normalize_maps(native_maps)
    overlays = {}
    for stage, heat in zip(("S4", "S8", "S16"), normalized_maps):
        heat_resized = resize_heatmap_smooth(heat, output_size, output_size)
        blended = Image.fromarray(activation_overlay(np.asarray(original), heat_resized), mode="RGB")
        overlays[stage] = draw_gt(blended, record["box"], crop)
    return full_context, draw_gt(original, record["box"], crop), overlays


def default_font(size):
    for path in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def square_panel(image, panel_size):
    contained = ImageOps.contain(image, (panel_size, panel_size), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (panel_size, panel_size), (238, 240, 244))
    panel.paste(contained, ((panel_size - contained.width) // 2, (panel_size - contained.height) // 2))
    return panel


def make_overview(all_rows, output_path, panel_size):
    header, label_width, gap = 68, 92, 10
    width = label_width + 5 * panel_size + 4 * gap
    height = header + 5 * panel_size + 4 * gap
    canvas = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(canvas)
    title_font, row_font = default_font(30), default_font(24)
    columns = ("完整原图", "局部裁剪", "S4", "S8", "S16")
    for column, stage in enumerate(columns):
        x = label_width + column * (panel_size + gap) + panel_size // 2
        draw.text((x, header // 2), stage, fill=(23, 35, 60), font=title_font, anchor="mm")
    for row, images in enumerate(all_rows):
        y = header + row * (panel_size + gap)
        draw.text((label_width // 2, y + panel_size // 2), f"样本{row + 1}", fill=(55, 68, 86), font=row_font, anchor="mm")
        for column, image in enumerate(images):
            x = label_width + column * (panel_size + gap)
            canvas.paste(square_panel(image, panel_size), (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output-size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.repo))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = 1
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    eligible = collect_eligible_targets(cfg.val_dataloader)
    selected = random_distinct_records(eligible, args.count, args.seed)

    model = cfg.model.cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = checkpoint.get("ema", {}).get("module")
    weight_source = "ema.module"
    if weights is None:
        weights = checkpoint.get("model", checkpoint)
        weight_source = "model_or_raw"
    model.load_state_dict(weights, strict=True)
    records = capture_selected(model, cfg.val_dataloader, selected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata, all_rows = [], []
    for index, record in enumerate(records, start=1):
        folder = args.output_dir / f"sample_{index:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        full_context, crop_gt, overlays = make_stage_overlays(record, args.output_size)
        full_context.save(folder / "00_full_image_with_crop_region.png")
        crop_gt.save(folder / "01_crop_with_gt.png")
        for stage in ("S4", "S8", "S16"):
            overlays[stage].save(folder / f"02_{stage}_overlay.png")
        all_rows.append([full_context, crop_gt, overlays["S4"], overlays["S8"], overlays["S16"]])

        points = side_points(record["box"], 512, 640)
        separations = {
            stage: float(np.mean([separation(record[stage.lower()], inside, outside) for inside, outside in points]))
            for stage in ("S4", "S8", "S16")
        }
        metadata.append(
            {
                "sample": index,
                "image_id": record["image_id"],
                "image_path": record["image_path"],
                "object_index": record["object_index"],
                "box_xyxy_at_640x512": record["box"],
                "geometric_size_px": record["geometric_size"],
                "boundary_separation": separations,
            }
        )

    make_overview(
        all_rows,
        args.output_dir / "随机5例_原图_裁剪_S4_S8_S16平滑叠加总览.png",
        args.output_size,
    )
    summary = {
        "model": "A00 visible D-FINE-N",
        "checkpoint": str(args.checkpoint),
        "weight_source": weight_source,
        "split": "Val",
        "eligible_population": "all 16-32 px GT targets on Val",
        "selection": "uniform pseudo-random shuffle followed by five distinct image IDs; no metric filtering",
        "random_seed": args.seed,
        "overlay": "original RGB background plus activation-weighted feature edge-energy heatmap; bicubic display interpolation; GT box in cyan",
        "color_scale": "S4/S8/S16 share one scale within each sample",
        "context_box": "yellow rectangle on the full image marks the displayed crop region",
        "samples": metadata,
    }
    (args.output_dir / "README_随机抽样与数值.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
