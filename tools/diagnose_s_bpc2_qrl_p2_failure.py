#!/usr/bin/env python3
"""Trace why QRL learned/shifted/uniform controls are nearly equivalent."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_s/s_bpc2_qrl_s8_s16_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT.parent
        / "reports/20_spatial_importance/S_BPC2_QRL/p2_smoke/smoke_step100.pth",
    )
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent
        / "reports/20_spatial_importance/S_BPC2_QRL/p2_smoke/failure_diagnostic.json",
    )
    return parser.parse_args()


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def rms(tensor):
    return float(tensor.detach().float().square().mean().sqrt())


def relative_rms(left, right):
    denominator = left.detach().float().square().mean().sqrt().clamp_min(1e-12)
    numerator = (left.detach().float() - right.detach().float()).square().mean().sqrt()
    return float(numerator / denominator)


def localization_loss(losses):
    return float(
        sum(
            value
            for key, value in losses.items()
            if any(part in key for part in ("loss_bbox", "loss_giou", "loss_fgl", "loss_ddf"))
        )
    )


def main():
    args = parse_args()
    device = torch.device("cuda")
    seed = 20260815
    cfg = YAMLConfig(str(args.config))
    base_loader = cfg.train_dataloader
    dataset_size = len(base_loader.dataset)
    holdout_indices = torch.randperm(
        dataset_size, generator=torch.Generator().manual_seed(seed)
    ).tolist()[: 32 * args.batches]
    loader = DataLoader(
        Subset(base_loader.dataset, holdout_indices),
        batch_size=32,
        shuffle=False,
        num_workers=base_loader.num_workers,
        collate_fn=base_loader.collate_fn,
        pin_memory=base_loader.pin_memory,
        persistent_workers=False,
    )

    model = cfg.model.to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"smoke checkpoint mismatch: {incompatible}")
    criterion = cfg.criterion.to(device)
    model.train()
    criterion.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    qrl = model.decoder.qrl
    qrl.capture_diagnostics = True

    rows = []
    modes = ("learned", "shifted", "uniform")
    with torch.no_grad():
        for batch_index, (samples, targets) in enumerate(loader):
            if batch_index >= args.batches:
                break
            samples = samples.to(device)
            targets = move_targets(targets, device)
            captured = {}
            for mode in modes:
                torch.manual_seed(seed + 10_000 + batch_index)
                torch.cuda.manual_seed_all(seed + 10_000 + batch_index)
                qrl.region_mode = mode
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = model(samples, targets=targets)
                with torch.autocast("cuda", enabled=False):
                    losses = criterion(outputs, targets, epoch=0)
                attention = qrl.last_attention.float()
                entropy = -(
                    attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()
                ).sum(dim=(-2, -1)) / math.log(36.0)
                captured[mode] = {
                    "attention": attention.cpu(),
                    "side_tokens": qrl.last_side_tokens.float().cpu(),
                    "delta": qrl.last_delta.detach().float().cpu(),
                    "delta_without_detail": qrl.last_delta_without_detail.float().cpu(),
                    "delta_without_context": qrl.last_delta_without_context.float().cpu(),
                    "boxes": outputs["pred_boxes"].detach().float().cpu(),
                    "localization_loss": localization_loss(losses),
                    "attention_entropy": float(entropy.mean()),
                    "attention_std": float(attention.std()),
                }
            learned = captured["learned"]
            row = {
                "batch": batch_index,
                "localization_loss": {
                    mode: captured[mode]["localization_loss"] for mode in modes
                },
                "attention_entropy": {
                    mode: captured[mode]["attention_entropy"] for mode in modes
                },
                "attention_std": {
                    mode: captured[mode]["attention_std"] for mode in modes
                },
                "learned_rms": {
                    "side_tokens": rms(learned["side_tokens"]),
                    "delta": rms(learned["delta"]),
                    "boxes": rms(learned["boxes"]),
                },
                "learned_detail_contribution_relative_rms": relative_rms(
                    learned["delta"], learned["delta_without_detail"]
                ),
                "learned_context_contribution_relative_rms": relative_rms(
                    learned["delta"], learned["delta_without_context"]
                ),
                "learned_vs_shifted_relative_rms": {
                    key: relative_rms(learned[key], captured["shifted"][key])
                    for key in ("attention", "side_tokens", "delta", "boxes")
                },
                "learned_vs_uniform_relative_rms": {
                    key: relative_rms(learned[key], captured["uniform"][key])
                    for key in ("attention", "side_tokens", "delta", "boxes")
                },
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    def mean(path):
        values = []
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            values.append(value)
        return float(np.mean(values))

    summary = {
        "checkpoint": str(args.checkpoint),
        "batches": args.batches,
        "mean_attention_entropy": {
            mode: mean(("attention_entropy", mode)) for mode in modes
        },
        "mean_attention_std": {mode: mean(("attention_std", mode)) for mode in modes},
        "mean_learned_detail_contribution_relative_rms": mean(
            ("learned_detail_contribution_relative_rms",)
        ),
        "mean_learned_context_contribution_relative_rms": mean(
            ("learned_context_contribution_relative_rms",)
        ),
        "mean_learned_vs_shifted_relative_rms": {
            key: mean(("learned_vs_shifted_relative_rms", key))
            for key in ("attention", "side_tokens", "delta", "boxes")
        },
        "mean_learned_vs_uniform_relative_rms": {
            key: mean(("learned_vs_uniform_relative_rms", key))
            for key in ("attention", "side_tokens", "delta", "boxes")
        },
        "rows": rows,
    }
    qrl.region_mode = "learned"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
