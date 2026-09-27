#!/usr/bin/env python3
"""Validate whether a trained REP1-SPAR branch prefers aligned SAM masks."""

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
        / "experiments/phase_s/s_rep1_spar_a00ft10_local.yml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=20)
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
        raise RuntimeError("REP1 alignment validation requires CUDA")
    if args.batches <= 0:
        raise ValueError("--batches must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    cfg = YAMLConfig(str(args.config))
    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights, weight_source = checkpoint_weights(checkpoint)
    incompatible = model.load_state_dict(weights, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")

    # SPAR is a training-only branch. Keep it active while freezing batch-norm
    # statistics so this read-only validation cannot alter the loaded model.
    model.backbone.train()
    for module in model.backbone.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    criterion.eval()

    aligned_losses = []
    box_losses = []
    shifted_losses = []
    processed_batches = 0
    skipped_samples = 0
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for samples, targets in cfg.train_dataloader:
            samples = samples.to(device)
            targets = [move_target(target, device) for target in targets]
            model.backbone(samples)
            fused = model.backbone.spar_fused_features
            if fused is None:
                raise RuntimeError("SPAR fusion did not run during validation")

            for index, target in enumerate(targets):
                masks = target.get("masks")
                if masks is None or masks.numel() == 0 or not bool(masks.any()):
                    skipped_samples += 1
                    continue
                criterion.spar_target_mode = "mask"
                criterion.spar_mask_shift_fraction = 0.0
                aligned = criterion._spar_loss(fused[index : index + 1], [target])
                criterion.spar_target_mode = "box"
                criterion.spar_mask_shift_fraction = 0.0
                box = criterion._spar_loss(fused[index : index + 1], [target])
                criterion.spar_target_mode = "mask"
                criterion.spar_mask_shift_fraction = 0.5
                shifted = criterion._spar_loss(
                    fused[index : index + 1], [target]
                )
                aligned_losses.append(float(aligned))
                box_losses.append(float(box))
                shifted_losses.append(float(shifted))

            processed_batches += 1
            if processed_batches >= args.batches:
                break

    if not aligned_losses:
        raise RuntimeError("no accepted SAM samples were evaluated")

    aligned_array = np.asarray(aligned_losses, dtype=np.float64)
    box_array = np.asarray(box_losses, dtype=np.float64)
    shifted_array = np.asarray(shifted_losses, dtype=np.float64)
    margins = shifted_array - aligned_array
    box_margins = box_array - aligned_array
    report = {
        "status": "PASS",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("last_epoch", -1)),
        "weight_source": weight_source,
        "batches": processed_batches,
        "accepted_samples": int(aligned_array.size),
        "skipped_samples": skipped_samples,
        "aligned_loss_mean": float(aligned_array.mean()),
        "box_loss_mean": float(box_array.mean()),
        "shifted_loss_mean": float(shifted_array.mean()),
        "box_minus_aligned_mean": float(box_margins.mean()),
        "shifted_minus_aligned_mean": float(margins.mean()),
        "aligned_better_than_box_fraction": float((box_margins > 0).mean()),
        "aligned_better_fraction": float((margins > 0).mean()),
        "interpretation": (
            "ALIGNED_PREFERRED" if margins.mean() > 0 else "NO_ALIGNED_PREFERENCE"
        ),
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
