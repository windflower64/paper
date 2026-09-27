#!/usr/bin/env python3
"""Preflight TNDP2 data synchronization, invariance, loss, and gradients."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from src.nn.backbone.hgnetv2 import HGNetv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_s/s_tndp2_retention_distill_s8_s16_local.yml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent
        / "reports/20_spatial_importance/S_TNDP2_RETENTION_DISTILLATION/preflight.json",
    )
    return parser.parse_args()


def gradient_l2(module):
    values = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("TNDP2 preflight requires CUDA")
    device = torch.device("cuda")

    common = dict(
        name="B0",
        return_idx=[1, 2, 3],
        freeze_at=-1,
        freeze_norm=True,
        pretrained=False,
    )
    torch.manual_seed(20260814)
    baseline = HGNetv2(**common).eval().to(device)
    candidate = HGNetv2(**common, tndp_stage=2, tndp_target_max=3.0).eval().to(device)
    incompatible = candidate.load_state_dict(baseline.state_dict(), strict=False)
    if incompatible.unexpected_keys or incompatible.missing_keys != [
        "tndp_head.reconstructor.weight",
        "tndp_head.reconstructor.bias",
    ]:
        raise RuntimeError(f"unexpected baseline transfer mismatch: {incompatible}")
    sample = torch.randn(1, 3, 512, 640, device=device)
    with torch.no_grad():
        base_features = baseline(sample)
        candidate_features = candidate(sample)
    max_invariance_error = max(
        float((base - test).abs().max())
        for base, test in zip(base_features, candidate_features)
    )
    if max_invariance_error != 0.0:
        raise RuntimeError(f"TNDP changed inference features: {max_invariance_error}")
    del baseline, candidate, base_features, candidate_features, sample
    gc.collect()
    torch.cuda.empty_cache()

    cfg = YAMLConfig(str(args.config))
    loader = cfg.train_dataloader
    samples, targets = next(iter(loader))
    positive_masks = sum(int(target["masks"].numel() > 0 and target["masks"].any()) for target in targets)
    if positive_masks == 0:
        raise RuntimeError("preflight batch contains no accepted SAM mask; choose another batch/seed")
    for target in targets:
        if "masks" not in target:
            raise RuntimeError("transformed SAM mask missing from target")
        if target["masks"].shape[-2:] != samples.shape[-2:]:
            raise RuntimeError(
                f"mask/image shape mismatch: {target['masks'].shape[-2:]} vs {samples.shape[-2:]}"
            )

    model = cfg.model.to(device).train()
    optimizer = cfg.optimizer
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(optimizer_parameters) != len({id(parameter) for parameter in optimizer_parameters}):
        raise RuntimeError("optimizer contains duplicate TNDP/model parameters")
    criterion = cfg.criterion.to(device).train()
    samples = samples.to(device)
    targets = [
        {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()}
        for target in targets
    ]
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(samples, targets=targets)
    with torch.autocast("cuda", enabled=False):
        losses = criterion(outputs, targets)
    total_loss = sum(losses.values())
    if not torch.isfinite(total_loss):
        raise RuntimeError(f"non-finite loss: {losses}")
    total_loss.backward()
    head_gradient = gradient_l2(model.backbone.tndp_head)
    downsample_gradient = gradient_l2(model.backbone.stages[2].downsample)
    if head_gradient <= 0 or downsample_gradient <= 0:
        raise RuntimeError(
            f"missing TNDP gradients: head={head_gradient}, downsample={downsample_gradient}"
        )

    report = {
        "config": str(args.config),
        "device": torch.cuda.get_device_name(0),
        "batch_size": int(samples.shape[0]),
        "image_shape": list(samples.shape),
        "positive_masks_in_batch": positive_masks,
        "inference_feature_max_abs_error": max_invariance_error,
        "tndp_prediction_shape": list(outputs["tndp_detail_prediction"].shape),
        "tndp_target_shape": list(outputs["tndp_detail_target"].shape),
        "loss_tndp_detail": float(losses["loss_tndp_detail"].detach()),
        "total_loss": float(total_loss.detach()),
        "head_gradient_l2": head_gradient,
        "stage2_downsample_gradient_l2": downsample_gradient,
        "optimizer_parameter_tensors": len(optimizer_parameters),
        "optimizer_unique_parameter_tensors": len(
            {id(parameter) for parameter in optimizer_parameters}
        ),
        "all_losses_finite": all(bool(torch.isfinite(value)) for value in losses.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
