#!/usr/bin/env python3
"""Measure aligned and shifted SAM retention error for an SIBR checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "experiments/phase_s/s_sibr1_sam_incoherent_boundary_retention_ft6_lr02_local.yml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def move_target(target, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }


def checkpoint_weights(checkpoint):
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"], "ema.module"
    model = checkpoint.get("model")
    if isinstance(model, dict):
        return model, "model"
    return checkpoint, "root"


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SIBR1 retention validation requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    cfg = YAMLConfig(str(args.config))
    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights, weight_source = checkpoint_weights(checkpoint)
    incompatible = model.load_state_dict(weights, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")

    # Enable only the training-time feature taps; freeze every BN statistic.
    model.backbone.train()
    for module in model.backbone.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

    aligned_losses = []
    shifted_losses = []
    processed_batches = 0
    skipped_samples = 0
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for samples, targets in cfg.train_dataloader:
            samples = samples.to(device)
            targets = [move_target(target, device) for target in targets]
            model.backbone(samples)
            features = model.backbone.sbrd_stage_features
            if features is None:
                raise RuntimeError("SIBR feature taps did not run")
            for index, target in enumerate(targets):
                masks = target.get("masks")
                if masks is None or masks.numel() == 0 or not bool(masks.any()):
                    skipped_samples += 1
                    continue
                sample_features = tuple(
                    feature[index : index + 1] for feature in features
                )
                shifted_target = dict(target)
                shifted_target["masks"] = torch.roll(
                    masks, shifts=masks.shape[-1] // 2, dims=-1
                )
                aligned_losses.append(
                    float(criterion._sbrd_loss(sample_features, [target]))
                )
                shifted_losses.append(
                    float(criterion._sbrd_loss(sample_features, [shifted_target]))
                )
            processed_batches += 1
            if processed_batches >= args.batches:
                break

    if not aligned_losses:
        raise RuntimeError("no accepted SAM samples were evaluated")
    aligned = np.asarray(aligned_losses, dtype=np.float64)
    shifted = np.asarray(shifted_losses, dtype=np.float64)
    report = {
        "status": "PASS",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("last_epoch", -1)),
        "weight_source": weight_source,
        "region_mode": criterion.sbrd_region_mode,
        "batches": processed_batches,
        "accepted_samples": int(aligned.size),
        "skipped_samples": skipped_samples,
        "aligned_retention_error_mean": float(aligned.mean()),
        "shifted_retention_error_mean": float(shifted.mean()),
        "aligned_minus_shifted_mean": float((aligned - shifted).mean()),
        "aligned_higher_error_fraction": float((aligned > shifted).mean()),
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
