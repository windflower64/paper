#!/usr/bin/env python3
"""S-BOX1：冻结A00、等能量移除S8局部细节，检验SAM极值点的因果价值。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "tools"))

from diagnose_true_contour_causality import (
    collect_detections,
    cosine,
    custom_coco_metrics,
    isotropic_detail,
    summarize,
    translate,
)


MODES = (
    "baseline",
    "sam_extrema",
    "shifted_extrema",
    "sam_contour",
    "gt_box_boundary",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=WORKSPACE / "outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=WORKSPACE
        / "reports/20_spatial_importance/S_DIAG3_TRUE_CONTOUR/masks_val",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--extreme-band", type=int, default=3)
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument(
        "--control",
        choices=("fixed_native", "matched_l2"),
        default="fixed_native",
    )
    return parser.parse_args()


def mask_modes(record, input_h, input_w, feature_h, feature_w, extreme_band, boundary_radius, device):
    image = Image.open(record["mask_path"]).convert("L")
    original_w, original_h = image.size
    resized = image.resize((input_w, input_h), Image.Resampling.NEAREST)
    mask_numpy = np.asarray(resized, dtype=np.uint8) > 127
    ys, xs = np.nonzero(mask_numpy)
    if xs.size == 0:
        return {
            name: torch.zeros((1, 1, feature_h, feature_w), device=device)
            for name in MODES[1:]
        }
    body = torch.from_numpy(mask_numpy.copy()).to(device=device)[None, None]
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    yy, xx = torch.meshgrid(
        torch.arange(input_h, device=device),
        torch.arange(input_w, device=device),
        indexing="ij",
    )
    body_2d = body[0, 0]
    extrema = (
        (body_2d & (xx <= min_x + extreme_band))
        | (body_2d & (xx >= max_x - extreme_band))
        | (body_2d & (yy <= min_y + extreme_band))
        | (body_2d & (yy >= max_y - extreme_band))
    ).float()[None, None]
    extrema = F.max_pool2d(extrema, 3, stride=1, padding=1)

    body_float = body.float()
    dilated = F.max_pool2d(
        body_float, 2 * boundary_radius + 1, stride=1, padding=boundary_radius
    )
    eroded = 1.0 - F.max_pool2d(
        1.0 - body_float,
        2 * boundary_radius + 1,
        stride=1,
        padding=boundary_radius,
    )
    contour = (dilated - eroded).clamp(0, 1)

    x1, y1, x2, y2 = [float(value) for value in record["bbox_xyxy"]]
    x1, x2 = x1 * input_w / original_w, x2 * input_w / original_w
    y1, y2 = y1 * input_h / original_h, y2 * input_h / original_h
    left, right = max(0, int(round(x1))), min(input_w - 1, int(round(x2)))
    top, bottom = max(0, int(round(y1))), min(input_h - 1, int(round(y2)))
    rectangle = torch.zeros_like(body_float)
    rectangle[..., top : bottom + 1, left : right + 1] = 1
    box_boundary = (
        F.max_pool2d(
            rectangle,
            2 * boundary_radius + 1,
            stride=1,
            padding=boundary_radius,
        )
        - (
            1.0
            - F.max_pool2d(
                1.0 - rectangle,
                2 * boundary_radius + 1,
                stride=1,
                padding=boundary_radius,
            )
        )
    ).clamp(0, 1)

    object_width = max(1.0, x2 - x1)
    shift = int(round(max(16.0, 1.5 * object_width)))
    center_x = 0.5 * (x1 + x2)
    shifted = translate(extrema, 0, shift if center_x < input_w / 2 else -shift)
    high_resolution = {
        "sam_extrema": extrema,
        "shifted_extrema": shifted,
        "sam_contour": contour,
        "gt_box_boundary": box_boundary,
    }
    return {
        name: F.adaptive_avg_pool2d(value, (feature_h, feature_w))
        for name, value in high_resolution.items()
    }


class ExtremeController:
    def __init__(self, backbone, records, extreme_band, boundary_radius, control):
        self.records = records
        self.extreme_band = extreme_band
        self.boundary_radius = boundary_radius
        self.control = control
        self.mode = "baseline"
        self.targets = None
        self.input_h = None
        self.input_w = None
        self.cached_masks = None
        self.last = {}
        self.handle = backbone.stages[2].downsample.register_forward_pre_hook(self._hook)

    def prepare(self, targets, input_h, input_w):
        self.targets = targets
        self.input_h = input_h
        self.input_w = input_w
        self.cached_masks = None
        self.last = {}

    def _build_masks(self, x):
        per_mode = {name: [] for name in MODES[1:]}
        for target in self.targets:
            image_id = int(target["image_id"].item())
            record = self.records.get(image_id)
            if record is None:
                sample = {
                    name: torch.zeros((1, 1, *x.shape[-2:]), device=x.device)
                    for name in MODES[1:]
                }
            else:
                sample = mask_modes(
                    record,
                    self.input_h,
                    self.input_w,
                    x.shape[-2],
                    x.shape[-1],
                    self.extreme_band,
                    self.boundary_radius,
                    x.device,
                )
            for name in MODES[1:]:
                per_mode[name].append(sample[name])
        return {name: torch.cat(values) for name, values in per_mode.items()}

    def _hook(self, _module, inputs):
        if self.mode == "baseline":
            return None
        x = inputs[0]
        if self.cached_masks is None:
            self.cached_masks = self._build_masks(x)
        detail = isotropic_detail(x)
        raw = {
            name: detail * mask.to(detail.dtype)
            for name, mask in self.cached_masks.items()
        }
        reference_norm = raw["sam_extrema"].flatten(1).norm(dim=1)
        current_norm = raw[self.mode].flatten(1).norm(dim=1).clamp_min(1e-12)
        if self.control == "matched_l2":
            scale = reference_norm / current_norm
        else:
            # 正确/错位极值掩码形状完全相同；保留原生振幅，避免把弱背景放大成异常扰动。
            scale = torch.ones_like(reference_norm)
        perturbation = raw[self.mode] * scale[:, None, None, None]
        x_norm = x.float().flatten(1).norm(dim=1).clamp_min(1e-12)
        self.last = {
            "relative_l2": (perturbation.flatten(1).norm(dim=1) / x_norm).detach().cpu(),
            "scale_to_extrema_energy": scale.detach().cpu(),
            "mask_mass": self.cached_masks[self.mode].flatten(1).sum(1).detach().cpu(),
            "cosine_extrema_shifted": cosine(
                self.cached_masks["sam_extrema"], self.cached_masks["shifted_extrema"]
            ).detach().cpu(),
            "cosine_extrema_contour": cosine(
                self.cached_masks["sam_extrema"], self.cached_masks["sam_contour"]
            ).detach().cpu(),
            "cosine_extrema_box": cosine(
                self.cached_masks["sam_extrema"], self.cached_masks["gt_box_boundary"]
            ).detach().cpu(),
        }
        return ((x.float() - perturbation).to(x.dtype),) + tuple(inputs[1:])

    def close(self):
        self.handle.remove()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records = json.loads((args.mask_root / "records.json").read_text(encoding="utf-8"))
    manual_ids = {
        int(path.stem) for path in (args.mask_root / "audit_overlays").glob("*.webp")
    }
    records = {
        int(record["image_id"]): record
        for record in all_records
        if bool(record.get("accepted")) or int(record["image_id"]) in manual_ids
    }

    sys.path.insert(0, str(ROOT))
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
    accepted_ids = set(records)
    detections = {mode: [] for mode in MODES}
    audit = defaultdict(lambda: defaultdict(list))
    controller = ExtremeController(
        model.backbone,
        records,
        args.extreme_band,
        args.boundary_radius,
        args.control,
    )
    images_seen = 0
    started = time.time()
    try:
        with torch.inference_mode():
            for batch_index, (samples, targets) in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                samples = samples.cuda(non_blocking=True)
                targets_gpu = [
                    {
                        key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
                        for key, value in target.items()
                    }
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
                    collect_detections(
                        detections[mode], targets_gpu, results, category_ids, accepted_ids
                    )
                    if mode != "baseline":
                        for field, values in controller.last.items():
                            audit[mode][field].extend(values.tolist())
                images_seen += len(targets)
                if batch_index == 0 or (batch_index + 1) % 20 == 0:
                    print(f"batch={batch_index + 1}/{len(loader)} images={images_seen}", flush=True)
    finally:
        controller.close()

    evaluated_ids = accepted_ids
    if args.max_batches is not None:
        evaluated_ids = {
            item["image_id"] for item in detections["baseline"]
        }
    metrics = {
        mode: custom_coco_metrics(coco, values, evaluated_ids)
        for mode, values in detections.items()
    }
    delta = {
        mode: {
            area: {
                metric: value - metrics["baseline"][area][metric]
                if value is not None and metrics["baseline"][area][metric] is not None
                else None
                for metric, value in area_metrics.items()
            }
            for area, area_metrics in metrics[mode].items()
        }
        for mode in MODES[1:]
    }
    extreme_ap75 = delta["sam_extrema"]["all"]["AP75"]
    shifted_ap75 = delta["shifted_extrema"]["all"]["AP75"]
    extreme_ap = delta["sam_extrema"]["all"]["AP50_95"]
    shifted_ap = delta["shifted_extrema"]["all"]["AP50_95"]
    gate = {
        "extrema_removal_hurts_ap75": bool(extreme_ap75 < -0.002),
        "extrema_is_more_causal_than_shifted": bool(
            extreme_ap75 < shifted_ap75 - 0.002 and extreme_ap < shifted_ap - 0.001
        ),
    }
    gate["pass"] = all(gate.values())
    summary = {
        "protocol": {
            "diagnostic": "S-BOX1 SAM极值点等能量因果移除",
            "training": False,
            "images_seen": images_seen,
            "evaluated_image_ids": len(evaluated_ids),
            "elapsed_seconds": time.time() - started,
            "transition": "S8进入S16下采样前",
            "control": args.control,
            "control_explanation": (
                "正确/错位极值使用同形掩码和固定原生细节移除强度，不放大弱背景"
                if args.control == "fixed_native"
                else "所有位置对照逐图缩放到与SAM极值点完全相同的移除L2"
            ),
            "weight_source": weight_source,
        },
        "metrics": metrics,
        "delta_from_baseline": delta,
        "perturbation_and_geometry": summarize(audit),
        "entry_gate": gate,
        "interpretation_boundary": "GT/SAM引导干预只用于机制验证，不是推理模块。",
    }
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output), "delta": delta, "entry_gate": gate},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
