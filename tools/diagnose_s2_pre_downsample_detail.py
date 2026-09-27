#!/usr/bin/env python3
"""Causal diagnostic for task-relevant detail before A00 S8->S16 downsampling.

The detector is frozen.  A fixed binomial low-pass separates the feature that
enters HGNet stage index 2 into low-frequency content and a local detail
residual.  The residual is removed at GT target cells, a one-cell target ring,
or equal-count background controls.  This is a Val-only mechanism probe; GT is
never proposed as an inference input.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


MODES = (
    "baseline",
    "detail_target",
    "detail_target_ring",
    "detail_ring_full",
    "detail_ring_energy_matched",
    "detail_background_energy_matched",
    "detail_background_ring_energy_matched",
    "detail_background_random",
    "low_target_energy_control",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    p.add_argument("--max-batches", type=int)
    p.add_argument("--stage-index", type=int, default=2)
    return p.parse_args()


class Collector:
    def __init__(self):
        self.raw = defaultdict(lambda: defaultdict(list))

    def add(self, name, values, groups):
        values = values.detach().float().cpu().tolist()
        for value, labels in zip(values, groups):
            if not math.isfinite(float(value)):
                continue
            for label in labels:
                self.raw[name][label].append(float(value))

    def summary(self):
        result = {}
        for metric, grouped in sorted(self.raw.items()):
            result[metric] = {}
            for group, values in sorted(grouped.items()):
                a = np.asarray(values, dtype=np.float64)
                result[metric][group] = {
                    "n": int(a.size),
                    "mean": float(a.mean()),
                    "std": float(a.std()),
                    "median": float(np.median(a)),
                    "q25": float(np.quantile(a, 0.25)),
                    "q75": float(np.quantile(a, 0.75)),
                }
        return result

    def serializable_raw(self):
        return {
            metric: {group: values for group, values in grouped.items()}
            for metric, grouped in self.raw.items()
        }


def batch_pair_metrics(before, after):
    before = before.float().flatten(1)
    after = after.float().flatten(1)
    delta = after - before
    return {
        "relative_l2": delta.norm(dim=1) / before.norm(dim=1).clamp_min(1e-12),
        "cosine": F.cosine_similarity(before, after, dim=1),
        "rms_ratio": after.norm(dim=1) / before.norm(dim=1).clamp_min(1e-12),
    }


def make_target_mask(targets, height, width, image_height, image_width, device):
    mask = torch.zeros((len(targets), 1, height, width), dtype=torch.bool, device=device)
    for batch_index, target in enumerate(targets):
        boxes = target["boxes"]
        absolute_xyxy = bool(boxes.numel() and boxes.max() > 2)
        for box in boxes:
            if absolute_xyxy:
                x1, y1, x2, y2 = box
                x1, x2 = x1 / image_width * width, x2 / image_width * width
                y1, y2 = y1 / image_height * height, y2 / image_height * height
            else:
                cx, cy, bw, bh = box
                x1, x2 = (cx - bw / 2) * width, (cx + bw / 2) * width
                y1, y2 = (cy - bh / 2) * height, (cy + bh / 2) * height
            ix1 = max(0, min(width - 1, int(torch.floor(x1).item())))
            iy1 = max(0, min(height - 1, int(torch.floor(y1).item())))
            ix2 = max(ix1 + 1, min(width, int(torch.ceil(x2).item())))
            iy2 = max(iy1 + 1, min(height, int(torch.ceil(y2).item())))
            mask[batch_index, 0, iy1:iy2, ix1:ix2] = True
    return mask


def region_mean(field, mask):
    field = field.float()
    values = []
    for item, area in zip(field, mask):
        area = area.expand_as(item)
        values.append(item[area].mean() if area.any() else item.new_tensor(float("nan")))
    return torch.stack(values)


def image_groups_from_coco(coco, targets):
    groups = []
    for target in targets:
        image_id = int(target["image_id"].item())
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
        labels = {"all"}
        for annotation in annotations:
            edge = math.sqrt(float(annotation.get("area", 0.0)))
            if edge < 8:
                labels.add("lt8")
            elif edge < 16:
                labels.add("8to16")
            elif edge < 32:
                labels.add("16to32")
            elif edge < 48:
                labels.add("32to48")
            else:
                labels.add("ge48")
        groups.append(labels)
    return groups


def binomial_blur(x):
    kernel = x.new_tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel, groups=x.shape[1])


def closest_energy_mask(candidate_mask, reference_mask, energy):
    """Select equal-count candidate cells closest to target detail energies."""
    selected = torch.zeros_like(candidate_mask)
    for batch_index in range(candidate_mask.shape[0]):
        candidate = candidate_mask[batch_index, 0].flatten().nonzero().flatten()
        reference = reference_mask[batch_index, 0].flatten().nonzero().flatten()
        if not candidate.numel() or not reference.numel():
            continue
        candidate_energy = energy[batch_index, 0].flatten()[candidate]
        reference_energy = energy[batch_index, 0].flatten()[reference]
        # Tiny UAV boxes occupy few S8 cells, so a greedy distinct match is
        # both tractable and closer in perturbation energy than top-k controls.
        order = torch.argsort(reference_energy, descending=True)
        available = torch.ones(candidate.numel(), dtype=torch.bool, device=candidate.device)
        chosen = []
        for ref_index in order[: candidate.numel()]:
            distance = (candidate_energy - reference_energy[ref_index]).abs()
            distance = distance.masked_fill(~available, float("inf"))
            local = int(distance.argmin().item())
            chosen.append(candidate[local])
            available[local] = False
        if chosen:
            flat = selected[batch_index, 0].flatten()
            flat[torch.stack(chosen)] = True
    return selected


def random_equal_count_mask(candidate_mask, reference_mask, image_ids):
    selected = torch.zeros_like(candidate_mask)
    for batch_index in range(candidate_mask.shape[0]):
        candidate = candidate_mask[batch_index, 0].flatten().nonzero().flatten()
        count = min(int(reference_mask[batch_index].sum().item()), int(candidate.numel()))
        if not count:
            continue
        generator = torch.Generator(device="cpu").manual_seed(20260812 + int(image_ids[batch_index]))
        order = torch.randperm(int(candidate.numel()), generator=generator)[:count]
        flat = selected[batch_index, 0].flatten()
        flat[candidate[order.to(candidate.device)]] = True
    return selected


class DetailIntervention:
    def __init__(self, stage):
        self.stage = stage
        self.mode = "baseline"
        self.targets = None
        self.image_height = None
        self.image_width = None
        self.image_ids = None
        self.masks = None
        self.last = {}
        self.handle = stage.register_forward_pre_hook(self._hook)

    def prepare(self, targets, image_height, image_width):
        self.targets = targets
        self.image_height = image_height
        self.image_width = image_width
        self.image_ids = [int(target["image_id"].item()) for target in targets]
        self.masks = None
        self.last = {}

    def _build_masks(self, x, detail_energy):
        target = make_target_mask(
            self.targets,
            x.shape[-2],
            x.shape[-1],
            self.image_height,
            self.image_width,
            x.device,
        )
        dilated1 = F.max_pool2d(target.float(), 3, 1, 1).bool()
        dilated2 = F.max_pool2d(target.float(), 5, 1, 2).bool()
        ring = dilated1 & ~target
        background = ~dilated2
        self.masks = {
            "target": target,
            "target_ring": dilated1,
            "ring_energy_matched": closest_energy_mask(ring, target, detail_energy),
            "background_energy_matched": closest_energy_mask(
                background, target, detail_energy
            ),
            "background_ring_energy_matched": closest_energy_mask(
                background, ring, detail_energy
            ),
            "background_random": random_equal_count_mask(
                background, target, self.image_ids
            ),
            "ring_full": ring,
            "background_full": background,
        }

    def _hook(self, module, inputs):
        x = inputs[0]
        low = binomial_blur(x.float()).to(x.dtype)
        detail = x - low
        detail_energy = detail.float().square().mean(dim=1, keepdim=True).sqrt()
        low_energy = low.float().square().mean(dim=1, keepdim=True).sqrt()
        if self.masks is None:
            self._build_masks(x, detail_energy)

        target = self.masks["target"]
        ring = self.masks["ring_full"]
        background = self.masks["background_full"]
        local_contrast = detail_energy / low_energy.clamp_min(1e-6)
        gx = F.pad(x.float()[..., 1:] - x.float()[..., :-1], (0, 1, 0, 0))
        gy = F.pad(x.float()[..., 1:, :] - x.float()[..., :-1, :], (0, 0, 0, 1))
        gradient = (gx.square() + gy.square()).mean(dim=1, keepdim=True).sqrt()

        if self.mode == "baseline":
            selected = torch.zeros_like(target)
            perturbation = torch.zeros_like(x)
            raw_down = module.downsample(x)
            low_down = module.downsample(low)
            for name, values in batch_pair_metrics(raw_down, low_down).items():
                self.last[f"downsample_all_detail::{name}"] = values
        else:
            mask_name = {
                "detail_target": "target",
                "detail_target_ring": "target_ring",
                "detail_ring_full": "ring_full",
                "detail_ring_energy_matched": "ring_energy_matched",
                "detail_background_energy_matched": "background_energy_matched",
                "detail_background_ring_energy_matched": "background_ring_energy_matched",
                "detail_background_random": "background_random",
                "low_target_energy_control": "target",
            }[self.mode]
            selected = self.masks[mask_name]
            if self.mode == "low_target_energy_control":
                detail_reference = detail.float() * selected
                low_reference = low.float() * selected
                scale = detail_reference.flatten(1).norm(dim=1) / low_reference.flatten(1).norm(
                    dim=1
                ).clamp_min(1e-12)
                perturbation = low * selected.to(low.dtype) * scale[:, None, None, None].to(low.dtype)
            else:
                perturbation = detail * selected.to(detail.dtype)

        modified = x - perturbation
        x_norm = x.float().flatten(1).norm(dim=1).clamp_min(1e-12)
        self.last.update(
            {
                "selected_cells": selected.flatten(1).sum(dim=1).float(),
                "input_perturbation_relative_l2": perturbation.float().flatten(1).norm(dim=1)
                / x_norm,
                "detail_rms::target": region_mean(detail_energy, target),
                "detail_rms::ring": region_mean(detail_energy, ring),
                "detail_rms::background": region_mean(detail_energy, background),
                "local_contrast::target": region_mean(local_contrast, target),
                "local_contrast::ring": region_mean(local_contrast, ring),
                "local_contrast::background": region_mean(local_contrast, background),
                "gradient_rms::target": region_mean(gradient, target),
                "gradient_rms::ring": region_mean(gradient, ring),
                "gradient_rms::background": region_mean(gradient, background),
            }
        )
        return (modified,) + tuple(inputs[1:])

    def close(self):
        self.handle.remove()


class FeatureCapture:
    def __init__(self, backbone, stage_index):
        self.data = {}
        self.handles = [
            backbone.stages[stage_index].register_forward_hook(self._make_hook("s16")),
            backbone.stages[stage_index + 1].register_forward_hook(self._make_hook("s32")),
        ]

    def _make_hook(self, name):
        def hook(module, inputs, output):
            self.data[name] = output.detach().float()

        return hook

    def clear(self):
        self.data.clear()

    def snapshot(self):
        return {name: value.clone() for name, value in self.data.items()}

    def close(self):
        for handle in self.handles:
            handle.remove()


def add_metrics(collector, prefix, metrics, groups):
    for name, values in metrics.items():
        collector.add(f"{prefix}/{name}", values, groups)


def collect_detections(storage, targets, results, category_ids):
    for target, result in zip(targets, results):
        boxes = result["boxes"].detach().cpu().clone()
        boxes[:, 2:] -= boxes[:, :2]
        for box, score, label in zip(
            boxes.tolist(), result["scores"].tolist(), result["labels"].tolist()
        ):
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
    edges = [
        (0, 1e10),
        (0, 8**2),
        (8**2, 16**2),
        (16**2, 32**2),
        (32**2, 48**2),
        (48**2, 1e10),
    ]
    evaluator.params.areaRng = [list(edge) for edge in edges]
    evaluator.params.areaRngLbl = labels
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]

    def mean_valid(array):
        valid = array[array > -1]
        return float(valid.mean()) if valid.size else None

    result = {}
    i75 = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.75)))
    for area_index, label in enumerate(labels):
        result[label] = {
            "AP50_95": mean_valid(precision[:, :, :, area_index, -1]),
            "AP50": mean_valid(precision[0, :, :, area_index, -1]),
            "AP75": mean_valid(precision[i75, :, :, area_index, -1]),
            "AR100": mean_valid(recall[:, :, area_index, -1]),
        }
    return result


def kernel_frequency_audit(stage):
    conv = stage.downsample.conv
    if isinstance(conv, torch.nn.Sequential):
        conv = next(module for module in conv if isinstance(module, torch.nn.Conv2d))
    weight = conv.weight.detach().float().cpu()[:, 0]
    patterns = {
        "dc": torch.ones((3, 3)),
        "horizontal_nyquist": torch.tensor([[1.0, -1.0, 1.0]]).repeat(3, 1),
        "vertical_nyquist": torch.tensor([[1.0], [-1.0], [1.0]]).repeat(1, 3),
        "checkerboard_nyquist": torch.tensor(
            [[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]]
        ),
    }
    response = {name: (weight * pattern).sum(dim=(-2, -1)).abs() for name, pattern in patterns.items()}
    dc = response["dc"].clamp_min(1e-8)
    output = {}
    for name, values in response.items():
        ratio = values / dc
        output[name] = {
            "mean_abs_response": float(values.mean()),
            "median_abs_response": float(values.median()),
            "mean_ratio_to_dc": float(ratio.mean()),
            "median_ratio_to_dc": float(ratio.median()),
            "fraction_channels_response_gt_dc": float((values > response["dc"]).float().mean()),
        }
    return output


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model.cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = checkpoint.get("ema", {}).get("module")
    weight_source = "ema.module"
    if weights is None:
        weights = checkpoint.get("model", checkpoint)
        weight_source = "model_or_raw"
    model.load_state_dict(weights, strict=True)

    if args.stage_index + 1 >= len(model.backbone.stages):
        raise ValueError("stage-index must have a following S32 stage")
    stage = model.backbone.stages[args.stage_index]
    controller = DetailIntervention(stage)
    capture = FeatureCapture(model.backbone, args.stage_index)
    loader, postprocessor = cfg.val_dataloader, cfg.postprocessor
    coco = loader.dataset.coco
    category_ids = sorted(coco.getCatIds())
    use_amp = args.precision == "fp16"
    collector = Collector()
    detections = {mode: [] for mode in MODES}
    images_seen = 0
    started = time.time()

    try:
        with torch.inference_mode():
            for batch_index, (samples, targets) in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                samples = samples.cuda(non_blocking=True)
                targets = [
                    {
                        key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
                        for key, value in target.items()
                    }
                    for target in targets
                ]
                groups = image_groups_from_coco(coco, targets)
                controller.prepare(targets, samples.shape[-2], samples.shape[-1])
                baseline_features = None
                baseline_outputs = None

                for mode in MODES:
                    controller.mode = mode
                    capture.clear()
                    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                        outputs = model(samples)
                    features = capture.snapshot()
                    sizes = torch.stack([target["orig_size"] for target in targets])
                    results = postprocessor(outputs, sizes)
                    collect_detections(detections[mode], targets, results, category_ids)

                    if mode == "baseline":
                        baseline_features = features
                        baseline_outputs = {
                            "pred_logits": outputs["pred_logits"].detach().float().clone(),
                            "pred_boxes": outputs["pred_boxes"].detach().float().clone(),
                        }
                        add_metrics(collector, "baseline_s8", controller.last, groups)
                    else:
                        add_metrics(
                            collector,
                            f"{mode}_intervention",
                            {
                                "selected_cells": controller.last["selected_cells"],
                                "input_perturbation_relative_l2": controller.last[
                                    "input_perturbation_relative_l2"
                                ],
                            },
                            groups,
                        )
                        add_metrics(
                            collector,
                            f"{mode}_s16",
                            batch_pair_metrics(baseline_features["s16"], features["s16"]),
                            groups,
                        )
                        add_metrics(
                            collector,
                            f"{mode}_s32",
                            batch_pair_metrics(baseline_features["s32"], features["s32"]),
                            groups,
                        )
                        add_metrics(
                            collector,
                            f"{mode}_pred_logits",
                            batch_pair_metrics(baseline_outputs["pred_logits"], outputs["pred_logits"]),
                            groups,
                        )
                        add_metrics(
                            collector,
                            f"{mode}_pred_boxes",
                            batch_pair_metrics(baseline_outputs["pred_boxes"], outputs["pred_boxes"]),
                            groups,
                        )

                images_seen += len(targets)
                if batch_index == 0 or (batch_index + 1) % 5 == 0:
                    print(
                        f"batch={batch_index + 1}/{len(loader)} images={images_seen}",
                        flush=True,
                    )
    finally:
        controller.close()
        capture.close()

    detection_metrics = {
        mode: custom_coco_metrics(coco, mode_detections)
        for mode, mode_detections in detections.items()
    }
    baseline_ap = detection_metrics["baseline"]["all"]["AP50_95"]
    detection_delta = {}
    for mode in MODES[1:]:
        detection_delta[mode] = {}
        for area in detection_metrics[mode]:
            detection_delta[mode][area] = {}
            for metric, value in detection_metrics[mode][area].items():
                base = detection_metrics["baseline"][area][metric]
                detection_delta[mode][area][metric] = (
                    None if value is None or base is None else value - base
                )

    summary = {
        "protocol": {
            "split": "Val only",
            "precision": args.precision,
            "max_batches": args.max_batches,
            "images_seen": images_seen,
            "elapsed_seconds": time.time() - started,
            "stage_index": args.stage_index,
            "detail_definition": "X8 - fixed 3x3 binomial low-pass(X8)",
            "warning": "GT-guided interventions are mechanism probes, not inference components.",
        },
        "model": {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "weight_source": weight_source,
        },
        "modes": list(MODES),
        "kernel_frequency_audit": kernel_frequency_audit(stage),
        "detection_metrics": detection_metrics,
        "detection_delta_from_baseline": detection_delta,
        "metrics": collector.summary(),
    }
    (args.output_dir / "s2_diag_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with gzip.open(args.output_dir / "s2_diag_raw_metrics.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(collector.serializable_raw(), handle, ensure_ascii=False)
    print(
        json.dumps(
            {
                "images_seen": images_seen,
                "baseline_ap": baseline_ap,
                "elapsed_seconds": summary["protocol"]["elapsed_seconds"],
                "output": str(args.output_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
