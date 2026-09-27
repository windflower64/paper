#!/usr/bin/env python3
"""S-BOX0：验证SAM掩码的四向极值是否适合作为检测框定位提示。

脚本不更新参数。它同时检查掩码外接框与标注框的几何一致性，以及A00框回归
损失对S8特征的梯度是否更集中在SAM四向极值，而不是完整轮廓或错位位置。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


BOX_LOSS_PREFIXES = ("loss_bbox", "loss_giou")
REGIONS = (
    "sam_extrema",
    "sam_left",
    "sam_right",
    "sam_top",
    "sam_bottom",
    "sam_contour",
    "sam_non_extreme_contour",
    "sam_tight_box_boundary",
    "gt_box_boundary",
    "gt_box_corners",
    "shifted_sam_extrema",
)
FIELDS = ("box_gradient", "box_feature_gradient_value")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a00-config",
        type=Path,
        default=ROOT / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument(
        "--a00-checkpoint",
        type=Path,
        default=WORKSPACE / "outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=WORKSPACE
        / "reports/20_spatial_importance/S_DIAG3_TRUE_CONTOUR/masks_val",
    )
    parser.add_argument(
        "--selection",
        choices=("manual_audit", "heldout_accepted"),
        default="manual_audit",
    )
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--extreme-band", type=int, default=3)
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE
        / "reports/23_sam_box_alignment/S_BOX0_MANUAL200/report.json",
    )
    return parser.parse_args()


def checkpoint_weights(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("ema", {}).get("module") is not None:
        return state["ema"]["module"], "ema.module"
    if "model" in state:
        return state["model"], "model"
    return state, "raw"


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def dilate(x, radius):
    return F.max_pool2d(x, 2 * radius + 1, stride=1, padding=radius)


def erode(x, radius):
    return 1.0 - dilate(1.0 - x, radius)


def translate_no_wrap(x, dx):
    result = torch.zeros_like(x)
    width = x.shape[-1]
    src_x1, src_x2 = max(0, -dx), min(width, width - dx)
    dst_x1, dst_x2 = max(0, dx), min(width, width + dx)
    if src_x2 > src_x1:
        result[..., dst_x1:dst_x2] = x[..., src_x1:src_x2]
    return result


def rectangle_mask(height, width, box, device):
    x1, y1, x2, y2 = box
    ix1 = max(0, min(width - 1, int(np.floor(x1))))
    iy1 = max(0, min(height - 1, int(np.floor(y1))))
    ix2 = max(ix1 + 1, min(width, int(np.ceil(x2))))
    iy2 = max(iy1 + 1, min(height, int(np.ceil(y2))))
    result = torch.zeros((1, 1, height, width), device=device)
    result[..., iy1:iy2, ix1:ix2] = 1.0
    return result


def corner_mask(height, width, box, radius, device):
    x1, y1, x2, y2 = box
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    result = torch.zeros((height, width), dtype=torch.bool, device=device)
    for x, y in ((x1, y1), (x1, y2), (x2, y1), (x2, y2)):
        result |= (xx - x).square() + (yy - y).square() <= radius**2
    return result.float()[None, None]


def box_iou(left, right):
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    return intersection / max(left_area + right_area - intersection, 1e-12)


def build_regions(record, mask_path, output_size, extreme_band, boundary_radius, device):
    mask_numpy = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 127
    ys, xs = np.nonzero(mask_numpy)
    if xs.size == 0:
        return None
    height, width = mask_numpy.shape
    union = torch.from_numpy(mask_numpy.copy()).to(device=device, dtype=torch.float32)[
        None, None
    ]
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    mask_box = [float(min_x), float(min_y), float(max_x + 1), float(max_y + 1)]
    gt_box = [float(value) for value in record["bbox_xyxy"]]

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    body = union[0, 0] > 0.5
    sides = {
        "sam_left": body & (xx <= min_x + extreme_band),
        "sam_right": body & (xx >= max_x - extreme_band),
        "sam_top": body & (yy <= min_y + extreme_band),
        "sam_bottom": body & (yy >= max_y - extreme_band),
    }
    side_tensors = {
        name: dilate(value.float()[None, None], 1) for name, value in sides.items()
    }
    extrema = torch.stack(list(side_tensors.values()), dim=0).amax(0)
    contour = (dilate(union, boundary_radius) - erode(union, boundary_radius)).clamp(0, 1)
    non_extreme = contour * (1.0 - dilate(extrema, boundary_radius)).clamp(0, 1)

    gt_rectangle = rectangle_mask(height, width, gt_box, device)
    mask_rectangle = rectangle_mask(height, width, mask_box, device)
    gt_boundary = (dilate(gt_rectangle, boundary_radius) - erode(gt_rectangle, boundary_radius)).clamp(0, 1)
    mask_box_boundary = (
        dilate(mask_rectangle, boundary_radius) - erode(mask_rectangle, boundary_radius)
    ).clamp(0, 1)
    gt_corners = corner_mask(height, width, gt_box, boundary_radius * 2, device)
    object_width = max(1.0, gt_box[2] - gt_box[0])
    shift = int(round(max(16.0, 1.5 * object_width)))
    center_x = 0.5 * (gt_box[0] + gt_box[2])
    shifted = translate_no_wrap(extrema, shift if center_x < width / 2 else -shift)

    raw = {
        "sam_extrema": extrema,
        **side_tensors,
        "sam_contour": contour,
        "sam_non_extreme_contour": non_extreme,
        "sam_tight_box_boundary": mask_box_boundary,
        "gt_box_boundary": gt_boundary,
        "gt_box_corners": gt_corners,
        "shifted_sam_extrema": shifted,
    }
    regions = {
        name: F.interpolate(value, size=output_size, mode="area")
        for name, value in raw.items()
    }
    side_offsets = [
        abs(mask_box[0] - gt_box[0]),
        abs(mask_box[2] - gt_box[2]),
        abs(mask_box[1] - gt_box[1]),
        abs(mask_box[3] - gt_box[3]),
    ]
    geometry = {
        "mask_box_iou_recomputed": box_iou(mask_box, gt_box),
        "mean_side_offset_pixels": float(np.mean(side_offsets)),
        "max_side_offset_pixels": float(np.max(side_offsets)),
        "mean_side_offset_s8_cells": float(np.mean(side_offsets) / 8.0),
        "mask_box": mask_box,
        "gt_box": gt_box,
    }
    return regions, geometry


def weighted_mean(value, weight):
    denominator = weight.sum()
    if float(denominator) <= 1e-8:
        return None
    return float((value * weight).sum() / denominator)


def summarize(values):
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def bootstrap_mean_ci(values, rng, samples=3000):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return [None, None]
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_summary(left, right, rng):
    pairs = [
        (l, r)
        for l, r in zip(left, right)
        if l is not None and r is not None and np.isfinite(l) and np.isfinite(r)
    ]
    left_array = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    right_array = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    difference = left_array - right_array
    ratio = left_array / np.maximum(right_array, 1e-12)
    if not pairs:
        return {
            "count": 0,
            "left": summarize([]),
            "right": summarize([]),
            "difference": {
                **summarize([]),
                "bootstrap_mean_95ci": [None, None],
                "positive_fraction": None,
            },
            "ratio": summarize([]),
        }
    return {
        "count": len(pairs),
        "left": summarize(left_array),
        "right": summarize(right_array),
        "difference": {
            **summarize(difference),
            "bootstrap_mean_95ci": bootstrap_mean_ci(difference, rng),
            "positive_fraction": float((difference > 0).mean()),
        },
        "ratio": summarize(ratio),
    }


def select_records(mask_root, selection, maximum):
    audit_ids = {int(path.stem) for path in (mask_root / "audit_overlays").glob("*.webp")}
    records = json.loads((mask_root / "records.json").read_text(encoding="utf-8"))
    if selection == "manual_audit":
        candidates = [record for record in records if int(record["image_id"]) in audit_ids]
    else:
        candidates = [
            record
            for record in records
            if int(record["image_id"]) not in audit_ids and bool(record.get("accepted"))
        ]
    selected = {int(record["image_id"]): record for record in candidates}
    available = len(selected)
    return dict(list(sorted(selected.items()))[:maximum]), len(audit_ids), available


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("S-BOX0需要CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    selected, audit_count, available = select_records(
        args.mask_root, args.selection, args.max_samples
    )
    if not selected:
        raise RuntimeError("没有可用样本")

    cfg = YAMLConfig(str(args.a00_config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = 1
    loader = cfg.val_dataloader
    model = cfg.model.to(device).train()
    criterion = cfg.criterion.to(device).train()
    weights, checkpoint_source = checkpoint_weights(args.a00_checkpoint)
    model.load_state_dict(weights, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    cache = {}

    def capture_s8(_module, inputs):
        cache["s8"] = inputs[0]
        cache["s8"].retain_grad()

    hook = model.backbone.stages[2].register_forward_pre_hook(capture_s8)
    raw = defaultdict(list)
    sample_results = []
    processed = set()
    peak_memory = 0.0

    try:
        for samples, targets in loader:
            image_id = int(targets[0]["image_id"].item())
            if image_id not in selected:
                continue
            record = selected[image_id]
            mask_path = Path(record["mask_path"])
            if not mask_path.exists():
                mask_path = args.mask_root / "masks" / f"{image_id:06d}.png"

            samples = samples.to(device).requires_grad_(True)
            targets = move_targets(targets, device)
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(samples, targets=targets)
            with torch.autocast("cuda", enabled=False):
                losses = criterion(outputs, targets)
            box_loss_keys = [key for key in losses if key.startswith(BOX_LOSS_PREFIXES)]
            box_loss = sum(losses[key] for key in box_loss_keys)
            s8 = cache["s8"]
            gradient = torch.autograd.grad(box_loss, s8, retain_graph=False)[0].float()
            fields = {
                "box_gradient": gradient.square().mean(1, keepdim=True).sqrt(),
                "box_feature_gradient_value": (s8.float() * gradient).abs().mean(1, keepdim=True),
            }
            built = build_regions(
                record,
                mask_path,
                s8.shape[-2:],
                args.extreme_band,
                args.boundary_radius,
                device,
            )
            if built is None:
                continue
            regions, geometry = built
            sample = {"image_id": image_id, **geometry}
            for key in (
                "mask_box_iou_recomputed",
                "mean_side_offset_pixels",
                "max_side_offset_pixels",
                "mean_side_offset_s8_cells",
            ):
                raw[key].append(geometry[key])
            for field_name, field in fields.items():
                for region_name in REGIONS:
                    value = weighted_mean(field, regions[region_name])
                    raw[f"{field_name}_{region_name}"].append(value)
                    sample[f"{field_name}_{region_name}"] = value
            sample_results.append(sample)
            processed.add(image_id)
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 2**20)
            del samples, targets, outputs, losses, box_loss, gradient, fields, s8
            if len(processed) >= len(selected):
                break
    finally:
        hook.remove()

    rng = np.random.default_rng(args.seed)
    comparisons = {}
    for field in FIELDS:
        for control in (
            "shifted_sam_extrema",
            "sam_contour",
            "sam_non_extreme_contour",
            "gt_box_boundary",
            "gt_box_corners",
        ):
            comparisons[f"{field}_extrema_vs_{control}"] = paired_summary(
                raw[f"{field}_sam_extrema"], raw[f"{field}_{control}"], rng
            )

    geometry_iou = summarize(raw["mask_box_iou_recomputed"])
    primary_shift = comparisons[
        "box_feature_gradient_value_extrema_vs_shifted_sam_extrema"
    ]
    primary_contour = comparisons[
        "box_feature_gradient_value_extrema_vs_sam_contour"
    ]
    primary_box = comparisons[
        "box_feature_gradient_value_extrema_vs_gt_box_boundary"
    ]
    gate = {
        "mask_box_geometry_is_compatible": bool(
            geometry_iou["median"] >= 0.70 and geometry_iou["q25"] >= 0.60
        ),
        "extrema_beats_shifted_control": bool(
            primary_shift["ratio"]["mean"] > 1.50
            and primary_shift["difference"]["bootstrap_mean_95ci"][0] > 0
        ),
        "extrema_selects_box_relevant_contour": bool(
            primary_contour["ratio"]["mean"] > 1.05
            and primary_contour["difference"]["positive_fraction"] > 0.60
        ),
        "extrema_adds_beyond_box_boundary": bool(
            primary_box["ratio"]["mean"] > 1.05
            and primary_box["difference"]["positive_fraction"] > 0.60
        ),
    }
    gate["pass"] = all(gate.values())

    report = {
        "protocol": {
            "diagnostic": "S-BOX0 SAM四向极值与检测框对齐诊断",
            "updates": 0,
            "selection": args.selection,
            "manual_audit_count": audit_count,
            "available_count": available,
            "selected_count": len(selected),
            "processed_count": len(processed),
            "a00_checkpoint": str(args.a00_checkpoint),
            "checkpoint_source": checkpoint_source,
            "extreme_band_input_pixels": args.extreme_band,
            "boundary_radius_input_pixels": args.boundary_radius,
            "box_loss_keys": box_loss_keys if processed else [],
            "peak_cuda_memory_mb": peak_memory,
        },
        "summary": {key: summarize(values) for key, values in sorted(raw.items())},
        "paired_tests": comparisons,
        "entry_gate": gate,
        "interpretation_boundary": (
            "通过只说明SAM极值适合作为训练期框定位提示，不证明新增分支会提升AP；"
            "发现集通过后必须在未参与发现的样本上复核。"
        ),
        "samples": sample_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                **report["protocol"],
                "geometry_iou": geometry_iou,
                "entry_gate": gate,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
