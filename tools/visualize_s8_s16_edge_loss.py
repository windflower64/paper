#!/usr/bin/env python3
"""Create a PPT-ready, real-weight visualization of S8->S16 edge attenuation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def robust_unit_interval(values, lower=2.0, upper=98.0):
    values = np.asarray(values, dtype=np.float32)
    low, high = np.percentile(values, [lower, upper], method="nearest")
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def edge_energy(feature):
    """Channel-normalized spatial gradient RMS, returned as one HxW map."""
    feature = feature.detach().float()
    mean = feature.mean(dim=(-2, -1), keepdim=True)
    std = feature.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    normalized = (feature - mean) / std
    gx = F.pad(normalized[..., 1:] - normalized[..., :-1], (0, 1, 0, 0))
    gy = F.pad(normalized[..., 1:, :] - normalized[..., :-1, :], (0, 0, 0, 1))
    return (gx.square() + gy.square()).mean(dim=1).sqrt()[0].cpu().numpy()


def square_crop(box, image_width, image_height, scale=5.0):
    x1, y1, x2, y2 = [float(v) for v in box]
    side = int(math.ceil(max(x2 - x1, y2 - y1) * scale))
    side = max(48, min(side, image_width, image_height))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    left = min(max(left, 0), image_width - side)
    top = min(max(top, 0), image_height - side)
    return left, top, left + side, top + side


def side_points(box, image_height, image_width):
    x1, y1, x2, y2 = [float(v) for v in box]
    dx = min(4.0, max(1.0, 0.20 * (x2 - x1)))
    dy = min(4.0, max(1.0, 0.20 * (y2 - y1)))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    entries = [
        ((x1 + dx, cy), (x1 - dx, cy)),
        ((x2 - dx, cy), (x2 + dx, cy)),
        ((cx, y1 + dy), (cx, y1 - dy)),
        ((cx, y2 - dy), (cx, y2 + dy)),
    ]
    result = []
    for inside, outside in entries:
        points = []
        for x, y in (inside, outside):
            points.append((x / image_width, y / image_height))
        if all(0 <= value <= 1 for point in points for value in point):
            result.append(tuple(points))
    return result


def sample_vectors(feature, points):
    grid = feature.new_tensor([[[[2 * x - 1, 2 * y - 1] for x, y in points]]])
    sampled = F.grid_sample(feature, grid, mode="bilinear", align_corners=False)
    return sampled[0, :, 0, :].transpose(0, 1)


def separation(feature, inside, outside):
    vectors = sample_vectors(feature, [inside, outside])
    numerator = (vectors[0] - vectors[1]).square().mean().sqrt()
    denominator = (0.5 * (vectors[0].square().mean() + vectors[1].square().mean())).sqrt()
    return float((numerator / denominator.clamp_min(1e-8)).cpu())


class S8S16Capture:
    def __init__(self, backbone):
        module = backbone.stages[2].downsample
        self.before = None
        self.after = None
        self.handles = [
            module.register_forward_pre_hook(self._pre),
            module.register_forward_hook(self._post),
        ]

    def _pre(self, module, inputs):
        self.before = inputs[0].detach().float()

    def _post(self, module, inputs, output):
        self.after = output.detach().float()

    def close(self):
        for handle in self.handles:
            handle.remove()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--scan-images", type=int, default=120)
    return parser.parse_args()


def set_chinese_font():
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(candidate)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def select_representative(model, loader, scan_images):
    capture = S8S16Capture(model.backbone)
    best = None
    target_retention = 0.65
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
                    retention = math.exp(np.mean(np.log(np.maximum(after_values, 1e-8))) - np.mean(np.log(np.maximum(before_values, 1e-8))))
                    if retention >= 1.0:
                        continue
                    score = abs(math.log(max(retention, 1e-8)) - math.log(target_retention))
                    if best is None or score < best["score"]:
                        best = {
                            "score": score,
                            "image_id": int(targets[0]["image_id"].item()),
                            "image_path": str(targets[0]["image_path"]),
                            "object_index": object_index,
                            "box": box,
                            "input": samples.detach().float().cpu(),
                            "s8": capture.before.detach().float().cpu(),
                            "s16": capture.after.detach().float().cpu(),
                            "before_values": before_values,
                            "after_values": after_values,
                            "retention": retention,
                            "geometric_size": geometric_size,
                            "scanned_images": min(batch_index + 1, scan_images),
                        }
    finally:
        capture.close()
    if best is None:
        raise RuntimeError("No 16-32 px target with S8->S16 attenuation was found.")
    return best


def draw_figure(record, output):
    set_chinese_font()
    image = record["input"][0].permute(1, 2, 0).numpy().clip(0, 1)
    height, width = image.shape[:2]
    box = record["box"]
    crop = square_crop(box, width, height)
    left, top, right, bottom = crop

    s8_map = edge_energy(record["s8"])
    s16_map = edge_energy(record["s16"])
    s8_up = F.interpolate(torch.from_numpy(s8_map)[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0].numpy()
    s16_up = F.interpolate(torch.from_numpy(s16_map)[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0].numpy()
    s8_crop = robust_unit_interval(s8_up[top:bottom, left:right])
    s16_crop = robust_unit_interval(s16_up[top:bottom, left:right])
    image_crop = image[top:bottom, left:right]

    fig = plt.figure(figsize=(16, 9), dpi=160, facecolor="#F7F8FA")
    grid = fig.add_gridspec(2, 4, height_ratios=[0.17, 0.83], width_ratios=[1, 1, 1, 1.18], hspace=0.03, wspace=0.10)
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.72, "问题：连续下采样削弱微小目标的精确定位依据", fontsize=25, weight="bold", color="#17233C", transform=title_ax.transAxes)
    title_ax.text(0.0, 0.23, "A00真实权重 · Val代表样本 · S8→S16（仅作直观展示，整体结论来自全验证集）", fontsize=13, color="#526174", transform=title_ax.transAxes)

    axes = [fig.add_subplot(grid[1, i]) for i in range(4)]
    for ax in axes:
        ax.set_facecolor("white")

    axes[0].imshow(image_crop)
    axes[0].add_patch(Rectangle((box[0] - left, box[1] - top), box[2] - box[0], box[3] - box[1], fill=False, edgecolor="#00E5FF", linewidth=2.2))
    axes[0].set_title("输入图像与GT框", fontsize=15, weight="bold", pad=12)
    axes[0].text(0.5, -0.07, f"目标约 {box[2]-box[0]:.0f}×{box[3]-box[1]:.0f} 像素", ha="center", transform=axes[0].transAxes, fontsize=11, color="#526174")

    for ax, heatmap, stage, cells in [
        (axes[1], s8_crop, "S8：压缩前", ((box[2] - box[0]) / 8, (box[3] - box[1]) / 8)),
        (axes[2], s16_crop, "S16：压缩后", ((box[2] - box[0]) / 16, (box[3] - box[1]) / 16)),
    ]:
        ax.imshow(image_crop, cmap="gray", alpha=0.20)
        ax.imshow(heatmap, cmap="magma", alpha=0.88, interpolation="nearest")
        ax.add_patch(Rectangle((box[0] - left, box[1] - top), box[2] - box[0], box[3] - box[1], fill=False, edgecolor="#00E5FF", linewidth=2.2))
        ax.set_title(stage, fontsize=15, weight="bold", pad=12)
        ax.text(0.5, -0.07, f"目标覆盖约 {cells[0]:.1f}×{cells[1]:.1f} 个网格", ha="center", transform=ax.transAxes, fontsize=11, color="#526174")

    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    evidence = axes[3]
    before = float(np.mean(record["before_values"]))
    after = float(np.mean(record["after_values"]))
    evidence.bar([0, 1], [before, after], color=["#2878B5", "#E45756"], width=0.58)
    evidence.set_xticks([0, 1], ["S8压缩前", "S16压缩后"], fontsize=12)
    evidence.set_ylabel("框边内外特征差异（归一化）", fontsize=11)
    evidence.set_title("边界区分能力下降", fontsize=15, weight="bold", pad=12)
    evidence.grid(axis="y", color="#DCE1E8", linewidth=0.8, alpha=0.8)
    evidence.set_axisbelow(True)
    for x, value in enumerate([before, after]):
        evidence.text(x, value + max(before, after) * 0.035, f"{value:.3f}", ha="center", fontsize=12, weight="bold")
    evidence.text(0.5, 0.69, f"该样本保持率：{record['retention']*100:.1f}%", ha="center", transform=evidence.transAxes, fontsize=13, color="#C83E4D", weight="bold")
    evidence.text(0.5, 0.57, "全Val因果干预", ha="center", transform=evidence.transAxes, fontsize=12, color="#526174")
    evidence.text(0.5, 0.49, "移除S8→S16目标边缘细节", ha="center", transform=evidence.transAxes, fontsize=12, color="#17233C")
    evidence.text(0.5, 0.39, "AP75 下降 3.292 个百分点", ha="center", transform=evidence.transAxes, fontsize=15, color="#C83E4D", weight="bold")
    evidence.text(0.5, 0.25, "结论：语义仍可能保留，\n但严格定位所需的空间证据被削弱。", ha="center", va="center", transform=evidence.transAxes, fontsize=12, color="#17233C", linespacing=1.5)
    evidence.text(0.5, 0.08, "热图：各Stage通道归一化后的空间梯度能量", ha="center", transform=evidence.transAxes, fontsize=9.5, color="#6F7B8B")
    evidence.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.025, "S系列核心问题：如何在不破坏原有语义主路的前提下，让目标相关、位置正确的细节跨越关键下采样阶段？", ha="center", fontsize=14, weight="bold", color="#17233C")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


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
    record = select_representative(model, cfg.val_dataloader, args.scan_images)
    draw_figure(record, args.output)

    metadata = {
        "model": "A00 visible D-FINE-N",
        "checkpoint": str(args.checkpoint),
        "weight_source": weight_source,
        "split": "Val",
        "selection_rule": "Among the first scanned validation images, choose a 16-32 px target with attenuation whose retention is closest to 0.65.",
        "image_id": record["image_id"],
        "image_path": record["image_path"],
        "object_index": record["object_index"],
        "box_xyxy_at_640x512": record["box"],
        "geometric_size_px": record["geometric_size"],
        "s8_side_separation_mean": float(np.mean(record["before_values"])),
        "s16_side_separation_mean": float(np.mean(record["after_values"])),
        "sample_retention": record["retention"],
        "whole_val_causal_delta_ap75": -0.03292023598175098,
        "heatmap_definition": "RMS spatial gradient energy after per-channel spatial z-normalization; maps are percentile-normalized per stage for visualization only.",
        "scope_warning": "The selected sample is illustrative. The whole-validation causal intervention is the aggregate evidence.",
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
