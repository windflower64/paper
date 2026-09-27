#!/usr/bin/env python3
"""Run the 100-step S-BPC2-QRL smoke, causal controls, and full validation."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from src.solver.det_engine import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_s/s_bpc2_qrl_s8_s16_local.yml",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--control-batches", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT.parent / "reports/20_spatial_importance/S_BPC2_QRL/p2_smoke",
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


def selected_loss(losses, fragments):
    return sum(
        value
        for key, value in losses.items()
        if any(fragment in key for fragment in fragments)
    )


def make_loader(base_loader, indices, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(base_loader.dataset, list(indices)),
        batch_size=32,
        shuffle=shuffle,
        num_workers=base_loader.num_workers,
        collate_fn=base_loader.collate_fn,
        pin_memory=base_loader.pin_memory,
        drop_last=shuffle,
        persistent_workers=False,
        generator=generator,
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("QRL P2 smoke requires CUDA")
    if args.steps <= 0 or args.control_batches <= 0:
        raise ValueError("steps and control-batches must be positive")
    seed = 20260815
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")

    cfg = YAMLConfig(str(args.config))
    base_loader = cfg.train_dataloader
    dataset_size = len(base_loader.dataset)
    holdout_count = 32 * args.control_batches
    if holdout_count >= dataset_size:
        raise RuntimeError("control holdout is larger than the training dataset")
    all_indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(seed)).tolist()
    holdout_indices = all_indices[:holdout_count]
    update_indices = all_indices[holdout_count:]
    update_loader = make_loader(base_loader, update_indices, True, seed + 1)
    control_loader = make_loader(base_loader, holdout_indices, False, seed + 2)

    model = cfg.model.to(device).train()
    criterion = cfg.criterion.to(device).train()
    optimizer = cfg.optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    qrl = model.decoder.qrl
    if qrl is None:
        raise RuntimeError("QRL module is missing")

    train_records = []
    iterator = iter(update_loader)
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        try:
            samples, targets = next(iterator)
        except StopIteration:
            iterator = iter(update_loader)
            samples, targets = next(iterator)
        samples = samples.to(device, non_blocking=True)
        targets = move_targets(targets, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(samples, targets=targets)
        with torch.autocast("cuda", enabled=False):
            losses = criterion(
                outputs,
                targets,
                epoch=0,
                step=step,
                global_step=step,
                epoch_step=args.steps,
            )
            total_loss = sum(losses.values())
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite loss at smoke step {step}: {losses}")
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step == 0 or (step + 1) % 10 == 0:
            train_records.append(
                {
                    "step": step + 1,
                    "total_loss": float(total_loss.detach()),
                    "region_loss": float(losses.get("loss_qrl_region", torch.nan).detach()),
                    "delta_rms": float(qrl.last_delta_rms),
                    "region_probability_mean": float(
                        qrl.last_region_logits.detach().sigmoid().mean()
                    ),
                    "region_probability_std": float(
                        qrl.last_region_logits.detach().sigmoid().std()
                    ),
                }
            )
            print(json.dumps(train_records[-1], ensure_ascii=False), flush=True)

    peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
    if float(qrl.last_delta_rms) <= 0:
        raise RuntimeError("QRL delta remained zero after 100 smoke steps")

    # Keep train-style decoder outputs for localization losses, while freezing
    # BN statistics.  Reset RNG for every mode so denoising queries are equal.
    model.train()
    criterion.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    modes = ("learned", "zero", "shifted", "uniform")
    controls = {mode: [] for mode in modes}
    control_delta_rms = {mode: [] for mode in modes}
    control_region_std = []
    with torch.no_grad():
        for batch_index, (samples, targets) in enumerate(control_loader):
            if batch_index >= args.control_batches:
                break
            samples = samples.to(device, non_blocking=True)
            targets = move_targets(targets, device)
            for mode in modes:
                torch.manual_seed(seed + 10_000 + batch_index)
                torch.cuda.manual_seed_all(seed + 10_000 + batch_index)
                qrl.region_mode = mode
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = model(samples, targets=targets)
                with torch.autocast("cuda", enabled=False):
                    losses = criterion(outputs, targets, epoch=0)
                    localization = selected_loss(
                        losses, ("loss_bbox", "loss_giou", "loss_fgl", "loss_ddf")
                    )
                controls[mode].append(float(localization))
                control_delta_rms[mode].append(float(qrl.last_delta_rms))
                if mode == "learned":
                    control_region_std.append(
                        float(qrl.last_region_logits.detach().sigmoid().std())
                    )
    qrl.region_mode = "learned"

    control_summary = {
        mode: {
            "localization_loss_mean": float(np.mean(values)),
            "localization_loss_std": float(np.std(values)),
            "delta_rms_mean": float(np.mean(control_delta_rms[mode])),
            "per_batch": values,
        }
        for mode, values in controls.items()
    }
    learned_better_shifted = (
        control_summary["learned"]["localization_loss_mean"]
        < control_summary["shifted"]["localization_loss_mean"]
    )
    learned_nonconstant = float(np.mean(control_region_std)) > 0.0
    learned_nonzero = control_summary["learned"]["delta_rms_mean"] > 0.0

    # Full Val verifies mask-free inference and emits the normal COCO AP table.
    val_stats, _ = evaluate(
        model,
        criterion,
        cfg.postprocessor,
        cfg.val_dataloader,
        cfg.evaluator,
        device,
        epoch=0,
        use_wandb=False,
        output_dir=str(args.output_dir),
        num_visualization_sample_batch=0,
    )

    report = {
        "config": str(args.config),
        "seed": seed,
        "steps": args.steps,
        "batch_size": 32,
        "update_dataset_size": len(update_indices),
        "heldout_dataset_size": len(holdout_indices),
        "heldout_disjoint": not bool(set(update_indices) & set(holdout_indices)),
        "peak_cuda_memory_mib": peak_memory_mib,
        "train_records": train_records,
        "controls": control_summary,
        "learned_region_probability_std_mean": float(np.mean(control_region_std)),
        "learned_better_than_shifted": learned_better_shifted,
        "learned_delta_nonzero": learned_nonzero,
        "learned_region_nonconstant": learned_nonconstant,
        "p2_mechanism_pass": bool(
            learned_better_shifted and learned_nonzero and learned_nonconstant
        ),
        "val_stats": val_stats,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "p2_smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": args.steps},
        args.output_dir / "smoke_step100.pth",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["p2_mechanism_pass"]:
        raise RuntimeError("QRL failed the preregistered P2 mechanism gate")


if __name__ == "__main__":
    main()
