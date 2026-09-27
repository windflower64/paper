#!/usr/bin/env python3
"""Frozen A00 causal test of SAM2 pseudo-contours at S8 -> S16.

All interventions remove the same isotropic native S8 detail energy.  Only the
spatial support changes: pseudo-contour, rectangular box edge, object interior,
locally shifted contour, or matched remote background.  This is a mechanism
probe, never an inference-time use of masks.
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
from PIL import Image


MODES = ("baseline", "true_contour", "box_edge", "target_interior", "shifted_contour", "background")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contour-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def isotropic_detail(x: torch.Tensor) -> torch.Tensor:
    return x.float() - F.avg_pool2d(F.pad(x.float(), (1, 1, 1, 1), mode="replicate"), 3, stride=1)


def translate(mask: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
    output = torch.zeros_like(mask)
    height, width = mask.shape[-2:]
    source_y1, source_y2 = max(0, -shift_y), min(height, height - shift_y)
    source_x1, source_x2 = max(0, -shift_x), min(width, width - shift_x)
    target_y1, target_y2 = max(0, shift_y), min(height, height + shift_y)
    target_x1, target_x2 = max(0, shift_x), min(width, width + shift_x)
    if source_y2 > source_y1 and source_x2 > source_x1:
        output[..., target_y1:target_y2, target_x1:target_x2] = mask[..., source_y1:source_y2, source_x1:source_x2]
    return output


def mask_modes(record: dict, input_h: int, input_w: int, feature_h: int, feature_w: int, device) -> dict[str, torch.Tensor]:
    original_image = Image.open(record["mask_path"]).convert("L")
    original_w, original_h = original_image.size
    resized = original_image.resize((input_w, input_h), Image.Resampling.NEAREST)
    object_mask = torch.from_numpy(np.asarray(resized, dtype=np.float32).copy() / 255.0)[None, None].to(device=device)
    object_mask = (object_mask >= 0.5).float()
    dilated = F.max_pool2d(object_mask, 3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - object_mask, 3, stride=1, padding=1)
    contour = (dilated - eroded).clamp(0, 1)
    interior = eroded

    x1, y1, x2, y2 = record["bbox_xyxy"]
    x1, x2 = x1 * input_w / original_w, x2 * input_w / original_w
    y1, y2 = y1 * input_h / original_h, y2 * input_h / original_h
    left, right = max(0, int(round(x1))), min(input_w - 1, int(round(x2)))
    top, bottom = max(0, int(round(y1))), min(input_h - 1, int(round(y2)))
    box_edge = torch.zeros_like(object_mask)
    box_edge[..., top : bottom + 1, left] = 1
    box_edge[..., top : bottom + 1, right] = 1
    box_edge[..., top, left : right + 1] = 1
    box_edge[..., bottom, left : right + 1] = 1

    # One S8 cell tests precise spatial alignment without moving to unrelated background.
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    local_shift_x = 8 if center_x < input_w / 2 else -8
    local_shift_y = 8 if center_y < input_h / 2 else -8
    shifted = translate(contour, local_shift_y, local_shift_x)
    # Remote background uses the same shape but moves by roughly half the image.
    remote_shift_x = input_w // 2 if center_x < input_w / 2 else -(input_w // 2)
    remote_shift_y = input_h // 2 if center_y < input_h / 2 else -(input_h // 2)
    background = translate(contour, remote_shift_y, remote_shift_x)

    high_resolution = {
        "true_contour": contour,
        "box_edge": box_edge,
        "target_interior": interior,
        "shifted_contour": shifted,
        "background": background,
    }
    return {
        name: F.adaptive_avg_pool2d(value, (feature_h, feature_w)).to(device=device)
        for name, value in high_resolution.items()
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    flat_a, flat_b = a.flatten(1), b.flatten(1)
    return F.cosine_similarity(flat_a, flat_b, dim=1, eps=1e-12)


class ContourController:
    def __init__(self, backbone, records: dict[int, dict]):
        self.records = records
        self.mode = "baseline"
        self.targets = None
        self.input_h = None
        self.input_w = None
        self.cached_masks = None
        self.last = {}
        self.handle = backbone.stages[2].downsample.register_forward_pre_hook(self._hook)

    def prepare(self, targets, input_h: int, input_w: int):
        self.targets = targets
        self.input_h = input_h
        self.input_w = input_w
        self.cached_masks = None
        self.last = {}

    def _build_masks(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        per_mode = {name: [] for name in MODES[1:]}
        for target in self.targets:
            image_id = int(target["image_id"].item())
            record = self.records.get(image_id)
            if record is None:
                sample = {name: torch.zeros((1, 1, x.shape[-2], x.shape[-1]), device=x.device) for name in MODES[1:]}
            else:
                sample = mask_modes(record, self.input_h, self.input_w, x.shape[-2], x.shape[-1], x.device)
            for name in MODES[1:]:
                per_mode[name].append(sample[name])
        return {name: torch.cat(values, dim=0) for name, values in per_mode.items()}

    def _hook(self, module, inputs):
        if self.mode == "baseline":
            return None
        x = inputs[0]
        if self.cached_masks is None:
            self.cached_masks = self._build_masks(x)
        detail = isotropic_detail(x)
        raw = {name: detail * mask.to(detail.dtype) for name, mask in self.cached_masks.items()}
        reference_norm = raw["true_contour"].flatten(1).norm(dim=1)
        current_norm = raw[self.mode].flatten(1).norm(dim=1).clamp_min(1e-12)
        scale = reference_norm / current_norm
        perturbation = raw[self.mode] * scale[:, None, None, None]
        x_norm = x.float().flatten(1).norm(dim=1).clamp_min(1e-12)
        self.last = {
            "relative_l2": (perturbation.flatten(1).norm(dim=1) / x_norm).detach().cpu(),
            "scale_to_contour_energy": scale.detach().cpu(),
            "mask_mass": self.cached_masks[self.mode].flatten(1).sum(1).detach().cpu(),
            "cosine_contour_box": cosine(self.cached_masks["true_contour"], self.cached_masks["box_edge"]).detach().cpu(),
            "cosine_contour_interior": cosine(self.cached_masks["true_contour"], self.cached_masks["target_interior"]).detach().cpu(),
            "cosine_contour_shifted": cosine(self.cached_masks["true_contour"], self.cached_masks["shifted_contour"]).detach().cpu(),
        }
        return ((x.float() - perturbation).to(x.dtype),) + tuple(inputs[1:])

    def close(self):
        self.handle.remove()


def collect_detections(storage, targets, results, category_ids, accepted_ids):
    for target, result in zip(targets, results):
        image_id = int(target["image_id"].item())
        if image_id not in accepted_ids:
            continue
        boxes = result["boxes"].detach().cpu().clone()
        boxes[:, 2:] -= boxes[:, :2]
        for box, score, label in zip(boxes.tolist(), result["scores"].tolist(), result["labels"].tolist()):
            storage.append({"image_id": image_id, "category_id": int(category_ids[int(label)]), "bbox": box, "score": float(score)})


def custom_coco_metrics(coco_gt, detections, image_ids):
    from faster_coco_eval import COCOeval_faster

    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")
    labels = ["all", "lt8", "8to16", "16to32", "32to48", "ge48"]
    edges = [(0, 1e10), (0, 8**2), (8**2, 16**2), (16**2, 32**2), (32**2, 48**2), (48**2, 1e10)]
    evaluator.params.areaRng = [list(edge) for edge in edges]
    evaluator.params.areaRngLbl = labels
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.params.imgIds = sorted(image_ids)
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


def summarize(values):
    output = {}
    for mode, fields in values.items():
        output[mode] = {}
        for name, observations in fields.items():
            array = np.asarray(observations, dtype=np.float64)
            finite = array[np.isfinite(array)]
            output[mode][name] = {
                "n": int(len(finite)),
                "mean": float(finite.mean()),
                "median": float(np.median(finite)),
                "q05": float(np.quantile(finite, 0.05)),
                "q95": float(np.quantile(finite, 0.95)),
            }
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records = json.loads(args.contour_records.read_text(encoding="utf-8"))
    records = {int(item["image_id"]): item for item in all_records if item["accepted"]}
    if len(records) < 300:
        raise RuntimeError(f"Only {len(records)} accepted pseudo-contours; at least 300 are required for a meaningful causal test.")

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
    accepted_ids = set(records)
    detections = {mode: [] for mode in MODES}
    audit = defaultdict(lambda: defaultdict(list))
    controller = ContourController(model.backbone, records)
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
                    collect_detections(detections[mode], targets_gpu, results, category_ids, accepted_ids)
                    if mode != "baseline":
                        for field, values in controller.last.items():
                            audit[mode][field].extend(values.tolist())
                images_seen += len(targets)
                if batch_index == 0 or (batch_index + 1) % 20 == 0:
                    print(f"batch={batch_index + 1}/{len(loader)} images={images_seen}", flush=True)
    finally:
        controller.close()

    metrics = {mode: custom_coco_metrics(coco, values, accepted_ids) for mode, values in detections.items()}
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
    audit_summary = summarize(audit)
    summary = {
        "protocol": {
            "split": "Val accepted-pseudo-contour subset",
            "training": False,
            "transition": "s8_to_s16",
            "images_seen": images_seen,
            "accepted_image_ids": len(accepted_ids),
            "elapsed_seconds": time.time() - started,
            "detail": "isotropic native S8 local detail",
            "energy_control": "every control is scaled per image to exact true-contour perturbation L2",
            "warning": "SAM2 masks are filtered pseudo-contours, not segmentation ground truth or an inference component.",
        },
        "model": {"config": str(args.config), "checkpoint": str(args.checkpoint), "weight_source": weight_source},
        "metrics": metrics,
        "delta_from_baseline": delta,
        "perturbation_and_geometry": audit_summary,
        "decision_rule": {
            "pass": "true_contour must cause clearly larger AP75 and 16to32 AP75 loss than box_edge, target_interior, and shifted_contour; geometry cosine and energy scales must remain interpretable",
            "fail": "weak/unstable ordering closes the contour-specific route; do not train a contour executor",
        },
    }
    (args.output_dir / "true_contour_causality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": metrics["baseline"], "delta": delta, "elapsed_seconds": summary["protocol"]["elapsed_seconds"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
