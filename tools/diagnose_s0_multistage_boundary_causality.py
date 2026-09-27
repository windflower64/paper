#!/usr/bin/env python3
"""Frozen A00 causal screen of boundary-aligned detail at every reduction.

For each reduction, remove the native horizontal detail at GT left/right box
sides and native vertical detail at GT top/bottom sides immediately before the
reduction.  The detector and weights are frozen.  The screen ranks stages by
AP75 loss and records the actual perturbation energy; it does not use the GT
mask as a proposed inference component.
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


TRANSITIONS = ("input_to_s2", "s2_to_s4", "s4_to_s8", "s8_to_s16", "s16_to_s32")
MODES = ("baseline",) + tuple(f"remove::{name}" for name in TRANSITIONS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def normalized_boxes(target, input_h, input_w, device):
    boxes = target["boxes"].detach().float().to(device)
    if not boxes.numel():
        return boxes.reshape(0, 4)
    if float(boxes.max()) <= 2.0:
        cx, cy, bw, bh = boxes.unbind(1)
        return torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), 1).clamp(0, 1)
    x1, y1, x2, y2 = boxes.unbind(1)
    return torch.stack((x1 / input_w, y1 / input_h, x2 / input_w, y2 / input_h), 1).clamp(0, 1)


def directional_detail(x):
    padded_x = F.pad(x.float(), (1, 1, 0, 0), mode="replicate")
    padded_y = F.pad(x.float(), (0, 0, 1, 1), mode="replicate")
    blur_x = F.avg_pool2d(padded_x, kernel_size=(1, 3), stride=1)
    blur_y = F.avg_pool2d(padded_y, kernel_size=(3, 1), stride=1)
    return x.float() - blur_x, x.float() - blur_y


def side_masks(targets, height, width, input_h, input_w, device):
    lr = torch.zeros((len(targets), 1, height, width), dtype=torch.bool, device=device)
    tb = torch.zeros_like(lr)
    for batch_index, target in enumerate(targets):
        boxes = normalized_boxes(target, input_h, input_w, device)
        for box in boxes:
            x1, y1, x2, y2 = box
            fx1, fx2 = float(x1 * width), float(x2 * width)
            fy1, fy2 = float(y1 * height), float(y2 * height)
            ix1 = max(0, min(width - 1, int(round(fx1 - 0.5))))
            ix2 = max(0, min(width - 1, int(round(fx2 - 0.5))))
            iy1 = max(0, min(height - 1, int(math.floor(fy1))))
            iy2 = max(iy1 + 1, min(height, int(math.ceil(fy2))))
            jy1 = max(0, min(height - 1, int(round(fy1 - 0.5))))
            jy2 = max(0, min(height - 1, int(round(fy2 - 0.5))))
            jx1 = max(0, min(width - 1, int(math.floor(fx1))))
            jx2 = max(jx1 + 1, min(width, int(math.ceil(fx2))))
            lr[batch_index, 0, iy1:iy2, ix1] = True
            lr[batch_index, 0, iy1:iy2, ix2] = True
            tb[batch_index, 0, jy1, jx1:jx2] = True
            tb[batch_index, 0, jy2, jx1:jx2] = True
    return lr, tb


class BoundaryController:
    def __init__(self, backbone):
        self.mode = "baseline"
        self.targets = None
        self.input_h = None
        self.input_w = None
        self.last = {}
        modules = {
            "input_to_s2": backbone.stem.stem1,
            "s2_to_s4": backbone.stem.stem3,
            "s4_to_s8": backbone.stages[1].downsample,
            "s8_to_s16": backbone.stages[2].downsample,
            "s16_to_s32": backbone.stages[3].downsample,
        }
        self.handles = [module.register_forward_pre_hook(self._hook(name)) for name, module in modules.items()]

    def prepare(self, targets, input_h, input_w):
        self.targets = targets
        self.input_h = input_h
        self.input_w = input_w
        self.last = {}

    def _hook(self, name):
        def hook(module, inputs):
            x = inputs[0]
            active = self.mode == f"remove::{name}"
            if not active:
                return None
            lr, tb = side_masks(self.targets, x.shape[-2], x.shape[-1], self.input_h, self.input_w, x.device)
            detail_x, detail_y = directional_detail(x)
            # Corners are assigned once to avoid double perturbation.
            lr_only = lr & ~tb
            perturbation = detail_x * lr_only.to(detail_x.dtype) + detail_y * tb.to(detail_y.dtype)
            x_norm = x.float().flatten(1).norm(dim=1).clamp_min(1e-12)
            p_norm = perturbation.flatten(1).norm(dim=1)
            self.last[name] = {
                "relative_l2": (p_norm / x_norm).detach().cpu(),
                "lr_cells": lr.flatten(1).sum(1).detach().cpu(),
                "tb_cells": tb.flatten(1).sum(1).detach().cpu(),
            }
            modified = x.float() - perturbation
            return (modified.to(x.dtype),) + tuple(inputs[1:])

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def collect_detections(storage, targets, results, category_ids):
    for target, result in zip(targets, results):
        boxes = result["boxes"].detach().cpu().clone()
        boxes[:, 2:] -= boxes[:, :2]
        for box, score, label in zip(boxes.tolist(), result["scores"].tolist(), result["labels"].tolist()):
            storage.append(
                {
                    "image_id": int(target["image_id"].item()),
                    "category_id": int(category_ids[int(label)]),
                    "bbox": box,
                    "score": float(score),
                }
            )


def custom_coco_metrics(coco_gt, detections):
    from faster_coco_eval import COCOeval_faster

    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")
    labels = ["all", "lt8", "8to16", "16to32", "32to48", "ge48"]
    edges = [(0, 1e10), (0, 8**2), (8**2, 16**2), (16**2, 32**2), (32**2, 48**2), (48**2, 1e10)]
    evaluator.params.areaRng = [list(edge) for edge in edges]
    evaluator.params.areaRngLbl = labels
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]

    def mean_valid(array):
        valid = array[array > -1]
        return float(valid.mean()) if valid.size else None

    output = {}
    i75 = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.75)))
    for area_index, label in enumerate(labels):
        output[label] = {
            "AP50_95": mean_valid(precision[:, :, :, area_index, -1]),
            "AP50": mean_valid(precision[0, :, :, area_index, -1]),
            "AP75": mean_valid(precision[i75, :, :, area_index, -1]),
            "AR100": mean_valid(recall[:, :, area_index, -1]),
        }
    return output


def summarize_perturbation(storage):
    output = {}
    for transition, fields in storage.items():
        output[transition] = {}
        for name, values in fields.items():
            array = np.asarray(values, dtype=np.float64)
            output[transition][name] = {
                "n": int(len(array)),
                "mean": float(array.mean()),
                "median": float(np.median(array)),
                "q05": float(np.quantile(array, 0.05)),
                "q95": float(np.quantile(array, 0.95)),
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

    loader, postprocessor = cfg.val_dataloader, cfg.postprocessor
    coco = loader.dataset.coco
    category_ids = sorted(coco.getCatIds())
    detections = {mode: [] for mode in MODES}
    perturbation = defaultdict(lambda: defaultdict(list))
    controller = BoundaryController(model.backbone)
    images_seen = 0
    started = time.time()
    try:
        with torch.inference_mode():
            for batch_index, (samples, targets) in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                samples = samples.cuda(non_blocking=True)
                targets_gpu = [
                    {key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value for key, value in target.items()}
                    for target in targets
                ]
                controller.prepare(targets_gpu, samples.shape[-2], samples.shape[-1])
                sizes = torch.stack([target["orig_size"] for target in targets_gpu])
                for mode in MODES:
                    controller.mode = mode
                    controller.last = {}
                    with torch.autocast("cuda", dtype=torch.float16):
                        outputs = model(samples)
                    results = postprocessor(outputs, sizes)
                    collect_detections(detections[mode], targets_gpu, results, category_ids)
                    if mode != "baseline":
                        transition = mode.split("::", 1)[1]
                        for field, values in controller.last[transition].items():
                            perturbation[transition][field].extend(values.tolist())
                images_seen += len(targets)
                if batch_index == 0 or (batch_index + 1) % 20 == 0:
                    print(f"batch={batch_index + 1}/{len(loader)} images={images_seen}", flush=True)
    finally:
        controller.close()

    metrics = {mode: custom_coco_metrics(coco, values) for mode, values in detections.items()}
    delta = {}
    for mode in MODES[1:]:
        delta[mode] = {}
        for area, area_metrics in metrics[mode].items():
            delta[mode][area] = {
                metric: None if value is None else value - metrics["baseline"][area][metric]
                for metric, value in area_metrics.items()
            }
    summary = {
        "protocol": {
            "split": "Val",
            "training": False,
            "modes": list(MODES),
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "images_seen": images_seen,
            "elapsed_seconds": time.time() - started,
            "intervention": "remove native x-detail on GT left/right sides and y-detail on GT top/bottom sides immediately before each reduction",
            "ranking_rule": "larger AP75 loss indicates greater detector dependence, interpreted jointly with perturbation relative-L2",
            "warning": "GT-guided perturbation is a causal probe, not an inference design.",
        },
        "model": {"config": str(args.config), "checkpoint": str(args.checkpoint), "weight_source": weight_source},
        "metrics": metrics,
        "delta_from_baseline": delta,
        "perturbation": summarize_perturbation(perturbation),
    }
    (args.output_dir / "multistage_boundary_causality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"baseline": metrics["baseline"]["all"], "delta": {mode: delta[mode]["all"] for mode in MODES[1:]}, "elapsed_seconds": summary["protocol"]["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
