#!/usr/bin/env python3
"""Frozen A00 audit of target-boundary retention at every spatial reduction.

This script does not train a model and does not claim that a generic feature
change is information loss.  For every GT box it samples paired points just
inside and outside each of the four box sides, measures normalized feature
separability before and after a reduction, and repeats the same measurement on
a non-overlapping random box of identical size.  The paired target-minus-
background change is the main statistic.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


class TransitionCapture:
    def __init__(self, backbone):
        self.data = {}
        self.handles = []
        transitions = {
            "input_to_s2": backbone.stem.stem1,
            "s2_to_s4": backbone.stem.stem3,
            "s4_to_s8": backbone.stages[1].downsample,
            "s8_to_s16": backbone.stages[2].downsample,
            "s16_to_s32": backbone.stages[3].downsample,
        }
        for name, module in transitions.items():
            self.handles.append(module.register_forward_pre_hook(self._pre(name)))
            self.handles.append(module.register_forward_hook(self._post(name)))

    def _pre(self, name):
        def hook(module, inputs):
            self.data.setdefault(name, {})["before"] = inputs[0].detach().float()

        return hook

    def _post(self, name):
        def hook(module, inputs, output):
            self.data.setdefault(name, {})["after"] = output.detach().float()

        return hook

    def clear(self):
        self.data.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def normalized_boxes(target, input_h, input_w):
    boxes = target["boxes"].detach().float().cpu()
    if not boxes.numel():
        return boxes.reshape(0, 4)
    if float(boxes.max()) <= 2.0:
        cx, cy, bw, bh = boxes.unbind(1)
        return torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), 1).clamp(0, 1)
    x1, y1, x2, y2 = boxes.unbind(1)
    return torch.stack((x1 / input_w, y1 / input_h, x2 / input_w, y2 / input_h), 1).clamp(0, 1)


def box_iou_one_to_many(box, boxes):
    if not boxes.numel():
        return torch.zeros(0)
    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)
    area2 = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    return inter / (area1 + area2 - inter).clamp_min(1e-12)


def random_control_box(box, all_boxes, image_id, object_index, seed):
    width = float(box[2] - box[0])
    height = float(box[3] - box[1])
    generator = torch.Generator().manual_seed(seed + image_id * 1009 + object_index * 9176)
    for _ in range(128):
        cx = width / 2 + float(torch.rand((), generator=generator)) * max(1e-6, 1 - width)
        cy = height / 2 + float(torch.rand((), generator=generator)) * max(1e-6, 1 - height)
        candidate = torch.tensor([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])
        if not all_boxes.numel() or float(box_iou_one_to_many(candidate, all_boxes).max()) < 0.01:
            return candidate.clamp(0, 1)
    # Deterministic fallback: half-image translation, clipped to the image.
    cx = ((float((box[0] + box[2]) / 2) + 0.5) % 1.0)
    cy = ((float((box[1] + box[3]) / 2) + 0.5) % 1.0)
    cx = min(max(cx, width / 2), 1 - width / 2)
    cy = min(max(cy, height / 2), 1 - height / 2)
    return torch.tensor([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]).clamp(0, 1)


def size_bin(box, input_h, input_w):
    width = float(box[2] - box[0]) * input_w
    height = float(box[3] - box[1]) * input_h
    edge = math.sqrt(max(0.0, width * height))
    if edge < 8:
        return "lt8"
    if edge < 16:
        return "8to16"
    if edge < 32:
        return "16to32"
    if edge < 48:
        return "32to48"
    return "ge48"


def side_points(box, input_h, input_w):
    x1, y1, x2, y2 = [float(value) for value in box]
    width_px = max(1e-6, (x2 - x1) * input_w)
    height_px = max(1e-6, (y2 - y1) * input_h)
    dx = min(4.0, max(1.0, 0.20 * width_px)) / input_w
    dy = min(4.0, max(1.0, 0.20 * height_px)) / input_h
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    entries = {
        "left": ((x1 + dx, cy), (x1 - dx, cy), "lr"),
        "right": ((x2 - dx, cy), (x2 + dx, cy), "lr"),
        "top": ((cx, y1 + dy), (cx, y1 - dy), "tb"),
        "bottom": ((cx, y2 - dy), (cx, y2 + dy), "tb"),
    }
    result = []
    for side, (inside, outside, orientation) in entries.items():
        if min(*inside, *outside) < 0 or max(*inside, *outside) > 1:
            continue
        result.append((side, orientation, inside, outside))
    return result


def sample_vectors(feature, batch_index, points):
    # points are in image-normalized [0, 1] coordinates.
    grid = feature.new_tensor([[[[2 * x - 1, 2 * y - 1] for x, y in points]]])
    sampled = F.grid_sample(
        feature[batch_index : batch_index + 1], grid, mode="bilinear", align_corners=False
    )
    return sampled[0, :, 0, :].transpose(0, 1)


def separation(feature, batch_index, inside, outside):
    vectors = sample_vectors(feature, batch_index, [inside, outside])
    numerator = (vectors[0] - vectors[1]).square().mean().sqrt()
    denominator = (0.5 * (vectors[0].square().mean() + vectors[1].square().mean())).sqrt()
    return float((numerator / denominator.clamp_min(1e-8)).cpu())


def bootstrap_mean_ci(values, samples, seed):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return None
    rng = np.random.default_rng(seed)
    if len(array) == 1:
        ci = [float(array[0]), float(array[0])]
    else:
        means = np.empty(samples, dtype=np.float64)
        for index in range(samples):
            means[index] = rng.choice(array, size=len(array), replace=True).mean()
        ci = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci95_mean": ci,
        "fraction_negative": float((array < 0).mean()),
    }


def summarize(records, bootstrap, seed):
    grouped = defaultdict(list)
    for record in records:
        keys = [
            (record["transition"], record["orientation"], "all"),
            (record["transition"], record["orientation"], record["size_bin"]),
        ]
        for key in keys:
            grouped[key].append(record)
    output = {}
    for (transition, orientation, group), rows in sorted(grouped.items()):
        target_log = [row["target_log_change"] for row in rows]
        control_log = [row["control_log_change"] for row in rows]
        paired = [row["target_minus_control_log_change"] for row in rows]
        target_ratio = [math.exp(value) for value in target_log]
        control_ratio = [math.exp(value) for value in control_log]
        node = output.setdefault(transition, {}).setdefault(orientation, {})
        node[group] = {
            "target_retention_geometric_mean": float(math.exp(np.mean(target_log))),
            "control_retention_geometric_mean": float(math.exp(np.mean(control_log))),
            "target_log_change": bootstrap_mean_ci(target_log, bootstrap, seed + 1),
            "control_log_change": bootstrap_mean_ci(control_log, bootstrap, seed + 2),
            "target_minus_control_log_change": bootstrap_mean_ci(paired, bootstrap, seed + 3),
            "fraction_target_retention_below_one": float((np.asarray(target_ratio) < 1).mean()),
            "fraction_control_retention_below_one": float((np.asarray(control_ratio) < 1).mean()),
        }
    return output


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = args.batch_size
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    model = cfg.model.cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = checkpoint.get("ema", {}).get("module")
    weight_source = "ema.module"
    if weights is None:
        weights = checkpoint.get("model", checkpoint)
        weight_source = "model_or_raw"
    model.load_state_dict(weights, strict=True)

    capture = TransitionCapture(model.backbone)
    records = []
    loader = cfg.val_dataloader
    images_seen = 0
    objects_seen = 0
    started = time.time()
    try:
        with torch.inference_mode():
            for batch_index, (samples, targets) in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                samples = samples.cuda(non_blocking=True)
                capture.clear()
                with torch.autocast("cuda", dtype=torch.float16):
                    model.backbone(samples)
                input_h, input_w = samples.shape[-2:]
                for image_index, target in enumerate(targets):
                    boxes = normalized_boxes(target, input_h, input_w)
                    image_id = int(target["image_id"].item())
                    for object_index, box in enumerate(boxes):
                        control = random_control_box(box, boxes, image_id, object_index, args.seed)
                        target_sides = side_points(box, input_h, input_w)
                        control_sides = side_points(control, input_h, input_w)
                        if len(target_sides) != 4 or len(control_sides) != 4:
                            continue
                        control_by_side = {side: (orientation, inside, outside) for side, orientation, inside, outside in control_sides}
                        for transition, pair in capture.data.items():
                            before, after = pair["before"], pair["after"]
                            for side, orientation, inside, outside in target_sides:
                                c_orientation, c_inside, c_outside = control_by_side[side]
                                target_before = separation(before, image_index, inside, outside)
                                target_after = separation(after, image_index, inside, outside)
                                control_before = separation(before, image_index, c_inside, c_outside)
                                control_after = separation(after, image_index, c_inside, c_outside)
                                eps = 1e-6
                                target_log_change = math.log(target_after + eps) - math.log(target_before + eps)
                                control_log_change = math.log(control_after + eps) - math.log(control_before + eps)
                                records.append(
                                    {
                                        "image_id": image_id,
                                        "object_index": object_index,
                                        "transition": transition,
                                        "side": side,
                                        "orientation": c_orientation,
                                        "size_bin": size_bin(box, input_h, input_w),
                                        "target_before": target_before,
                                        "target_after": target_after,
                                        "control_before": control_before,
                                        "control_after": control_after,
                                        "target_log_change": target_log_change,
                                        "control_log_change": control_log_change,
                                        "target_minus_control_log_change": target_log_change - control_log_change,
                                    }
                                )
                        objects_seen += 1
                images_seen += len(targets)
                if batch_index == 0 or (batch_index + 1) % 25 == 0:
                    print(f"batch={batch_index + 1}/{len(loader)} images={images_seen} objects={objects_seen}", flush=True)
    finally:
        capture.close()

    summary = {
        "protocol": {
            "split": "Val",
            "training": False,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "images_seen": images_seen,
            "objects_seen": objects_seen,
            "records": len(records),
            "elapsed_seconds": time.time() - started,
            "main_statistic": "paired target-minus-same-size-random-control log separability change",
            "negative_interpretation": "target box sides are attenuated more than random background box sides",
            "scope_warning": "This establishes stage-local boundary attenuation, not yet causal AP75 benefit.",
        },
        "model": {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "weight_source": weight_source,
        },
        "summary": summarize(records, args.bootstrap, args.seed),
    }
    (args.output_dir / "multistage_boundary_retention_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "multistage_boundary_retention_records.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary["protocol"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
