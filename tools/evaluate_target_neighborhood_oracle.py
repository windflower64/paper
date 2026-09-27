#!/usr/bin/env python3
"""Frozen A00 oracle test for mask-guided pre-downsample detail preservation.

The standard S8->S16 path is unchanged.  Before its depthwise stride-2
operation, the test adds a fixed amount of native isotropic S8 detail inside a
SAM2-derived target-neighborhood gate.  Shifted and remote-background gates
are matched per image to the target perturbation L2.
"""

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

from diagnose_true_contour_causality import (
    collect_detections,
    custom_coco_metrics,
    isotropic_detail,
    summarize,
    translate,
)


MODES = ("baseline", "target_neighborhood", "shifted_neighborhood", "background_neighborhood")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contour-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def target_neighborhood_gate(
    record: dict,
    input_h: int,
    input_w: int,
    feature_h: int,
    feature_w: int,
    device: torch.device,
) -> torch.Tensor:
    image = Image.open(record["mask_path"]).convert("L")
    image = image.resize((input_w, input_h), Image.Resampling.NEAREST)
    mask = torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255.0)[None, None]
    mask = (mask >= 0.5).float().to(device)
    projected = F.adaptive_avg_pool2d(mask, (feature_h, feature_w))
    # One S8-cell expansion covers object interior, boundary and tight context.
    expanded = F.max_pool2d(projected, kernel_size=3, stride=1, padding=1)
    # A local average makes a soft transition instead of a brittle binary rim.
    return F.avg_pool2d(F.pad(expanded, (1, 1, 1, 1), mode="replicate"), 3, stride=1).clamp(0, 1)


class NeighborhoodOracleController:
    def __init__(self, backbone, records: dict[int, dict], gamma: float):
        self.records = records
        self.gamma = float(gamma)
        self.mode = "baseline"
        self.targets = None
        self.input_h = None
        self.input_w = None
        self.cached_gates = None
        self.last = {}
        self.handle = backbone.stages[2].downsample.register_forward_pre_hook(self._hook)

    def prepare(self, targets, input_h: int, input_w: int) -> None:
        self.targets = targets
        self.input_h = input_h
        self.input_w = input_w
        self.cached_gates = None
        self.last = {}

    def _build_gates(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        target_gates = []
        for target in self.targets:
            image_id = int(target["image_id"].item())
            record = self.records.get(image_id)
            if record is None:
                gate = torch.zeros((1, 1, x.shape[-2], x.shape[-1]), device=x.device)
            else:
                gate = target_neighborhood_gate(
                    record,
                    self.input_h,
                    self.input_w,
                    x.shape[-2],
                    x.shape[-1],
                    x.device,
                )
            target_gates.append(gate)
        target_gate = torch.cat(target_gates, dim=0)
        shifted = translate(target_gate, 1, 1)
        # Periodic half-grid roll keeps the complete support and therefore
        # permits an exact per-image energy match. A zero-filled translation
        # can crop small gates completely when the source lies near an edge.
        background = torch.roll(
            target_gate,
            shifts=(x.shape[-2] // 2, x.shape[-1] // 2),
            dims=(-2, -1),
        )
        return {
            "target_neighborhood": target_gate,
            "shifted_neighborhood": shifted,
            "background_neighborhood": background,
        }

    def _hook(self, module, inputs):
        if self.mode == "baseline":
            return None
        x = inputs[0]
        if self.cached_gates is None:
            self.cached_gates = self._build_gates(x)
        detail = isotropic_detail(x)
        raw = {
            name: self.gamma * detail * gate.to(detail.dtype)
            for name, gate in self.cached_gates.items()
        }
        reference_norm = raw["target_neighborhood"].flatten(1).norm(dim=1)
        current_norm_raw = raw[self.mode].flatten(1).norm(dim=1)
        invalid = (reference_norm > 1e-12) & (current_norm_raw <= 1e-12)
        if invalid.any():
            raise RuntimeError(f"{self.mode} has zero energy for a non-zero target reference")
        current_norm = current_norm_raw.clamp_min(1e-12)
        scale = reference_norm / current_norm
        perturbation = raw[self.mode] * scale[:, None, None, None]
        x_norm = x.float().flatten(1).norm(dim=1).clamp_min(1e-12)
        self.last = {
            "relative_l2": (perturbation.flatten(1).norm(dim=1) / x_norm).detach().cpu(),
            "scale_to_target_energy": scale.detach().cpu(),
            "gate_mass": self.cached_gates[self.mode].flatten(1).sum(1).detach().cpu(),
        }
        return ((x.float() + perturbation).to(x.dtype),) + tuple(inputs[1:])

    def close(self) -> None:
        self.handle.remove()


def metric_delta(metrics: dict) -> dict:
    return {
        mode: {
            area: {
                name: None if value is None else value - metrics["baseline"][area][name]
                for name, value in values.items()
            }
            for area, values in metrics[mode].items()
        }
        for mode in MODES[1:]
    }


def oracle_decision(metrics: dict) -> dict:
    baseline = metrics["baseline"]
    target = metrics["target_neighborhood"]
    controls = (metrics["shifted_neighborhood"], metrics["background_neighborhood"])
    checks = {
        "all_ap_improves": target["all"]["AP50_95"] > baseline["all"]["AP50_95"],
        "all_ap75_improves": target["all"]["AP75"] > baseline["all"]["AP75"],
        "16to32_ap75_non_decreasing": target["16to32"]["AP75"] >= baseline["16to32"]["AP75"],
        "target_beats_controls_all_ap": all(
            target["all"]["AP50_95"] > control["all"]["AP50_95"] for control in controls
        ),
        "target_beats_controls_all_ap75": all(
            target["all"]["AP75"] > control["all"]["AP75"] for control in controls
        ),
    }
    return {"pass": all(checks.values()), "checks": checks}


def main() -> None:
    args = parse_args()
    if args.gamma <= 0:
        raise ValueError("gamma must be positive for the preservation oracle")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records = json.loads(args.contour_records.read_text(encoding="utf-8"))
    records = {int(item["image_id"]): item for item in all_records if item["accepted"]}
    if len(records) < 300:
        raise RuntimeError(f"Only {len(records)} accepted masks; at least 300 are required")

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
    controller = NeighborhoodOracleController(model.backbone, records, args.gamma)
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
    decision = oracle_decision(metrics)
    summary = {
        "protocol": {
            "training": False,
            "transition": "s8_to_s16",
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "images_seen": images_seen,
            "accepted_image_ids": len(accepted_ids),
            "elapsed_seconds": time.time() - started,
            "detail": "additive isotropic native S8 detail inside a soft target-neighborhood gate",
            "energy_control": "shifted/background controls are per-image L2 matched to target-neighborhood",
        },
        "model": {"config": str(args.config), "checkpoint": str(args.checkpoint), "weight_source": weight_source},
        "metrics": metrics,
        "delta_from_baseline": metric_delta(metrics),
        "perturbation": summarize(audit),
        "decision": decision,
    }
    output_path = args.output_dir / "target_neighborhood_oracle_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"delta": summary["delta_from_baseline"], "decision": decision}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
