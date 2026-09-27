#!/usr/bin/env python3
"""Equal-energy controls for the winning boundary-sensitive reduction stage."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from diagnose_s0_multistage_boundary_causality import (
    TRANSITIONS,
    collect_detections,
    custom_coco_metrics,
    directional_detail,
    side_masks,
    summarize_perturbation,
)


MODES = ("baseline", "aligned", "crossed", "shifted")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transition", choices=TRANSITIONS, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


class SpecificityController:
    def __init__(self, backbone, transition):
        modules = {
            "input_to_s2": backbone.stem.stem1,
            "s2_to_s4": backbone.stem.stem3,
            "s4_to_s8": backbone.stages[1].downsample,
            "s8_to_s16": backbone.stages[2].downsample,
            "s16_to_s32": backbone.stages[3].downsample,
        }
        self.transition = transition
        self.mode = "baseline"
        self.targets = None
        self.input_h = None
        self.input_w = None
        self.last = {}
        self.handle = modules[transition].register_forward_pre_hook(self._hook)

    def prepare(self, targets, input_h, input_w):
        self.targets = targets
        self.input_h = input_h
        self.input_w = input_w
        self.last = {}

    def _hook(self, module, inputs):
        if self.mode == "baseline":
            return None
        x = inputs[0]
        lr, tb = side_masks(self.targets, x.shape[-2], x.shape[-1], self.input_h, self.input_w, x.device)
        dx, dy = directional_detail(x)
        lr_only = lr & ~tb
        aligned = dx * lr_only.to(dx.dtype) + dy * tb.to(dy.dtype)
        if self.mode == "aligned":
            perturbation = aligned
        elif self.mode == "crossed":
            perturbation = dy * lr_only.to(dy.dtype) + dx * tb.to(dx.dtype)
        elif self.mode == "shifted":
            shifted_lr = torch.roll(lr_only, shifts=(x.shape[-2] // 2, x.shape[-1] // 2), dims=(-2, -1))
            shifted_tb = torch.roll(tb, shifts=(x.shape[-2] // 2, x.shape[-1] // 2), dims=(-2, -1))
            perturbation = dx * shifted_lr.to(dx.dtype) + dy * shifted_tb.to(dy.dtype)
        else:
            raise ValueError(self.mode)

        aligned_norm = aligned.flatten(1).norm(dim=1)
        current_norm = perturbation.flatten(1).norm(dim=1).clamp_min(1e-12)
        scale = aligned_norm / current_norm
        perturbation = perturbation * scale[:, None, None, None]
        x_norm = x.float().flatten(1).norm(dim=1).clamp_min(1e-12)
        self.last = {
            "relative_l2": (perturbation.flatten(1).norm(dim=1) / x_norm).detach().cpu(),
            "scale": scale.detach().cpu(),
            "lr_cells": lr.flatten(1).sum(1).detach().cpu(),
            "tb_cells": tb.flatten(1).sum(1).detach().cpu(),
        }
        return ((x.float() - perturbation).to(x.dtype),) + tuple(inputs[1:])

    def close(self):
        self.handle.remove()


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
    controller = SpecificityController(model.backbone, args.transition)
    images_seen = 0
    started = time.time()
    try:
        with torch.inference_mode():
            for batch_index, (samples, targets) in enumerate(loader):
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
                        for field, values in controller.last.items():
                            perturbation[mode][field].extend(values.tolist())
                images_seen += len(targets)
                if batch_index == 0 or (batch_index + 1) % 20 == 0:
                    print(f"batch={batch_index + 1}/{len(loader)} images={images_seen}", flush=True)
    finally:
        controller.close()

    metrics = {mode: custom_coco_metrics(coco, values) for mode, values in detections.items()}
    delta = {
        mode: {
            area: {
                metric: None if value is None else value - metrics["baseline"][area][metric]
                for metric, value in area_metrics.items()
            }
            for area, area_metrics in metrics[mode].items()
        }
        for mode in MODES[1:]
    }
    summary = {
        "protocol": {
            "split": "Val",
            "training": False,
            "transition": args.transition,
            "images_seen": images_seen,
            "elapsed_seconds": time.time() - started,
            "controls": {
                "aligned": "x-detail at left/right and y-detail at top/bottom",
                "crossed": "orientation swapped at identical boundary cells",
                "shifted": "correct orientations at half-map translated cells",
            },
            "energy_control": "crossed and shifted are scaled per image to aligned perturbation L2",
        },
        "model": {"config": str(args.config), "checkpoint": str(args.checkpoint), "weight_source": weight_source},
        "metrics": metrics,
        "delta_from_baseline": delta,
        "perturbation": summarize_perturbation(perturbation),
    }
    (args.output_dir / "boundary_specificity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"baseline": metrics["baseline"]["all"], "delta": {mode: delta[mode]["all"] for mode in MODES[1:]}, "elapsed_seconds": summary["protocol"]["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
