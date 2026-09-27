#!/usr/bin/env python3
"""Diagnose what TNDP2 learned without modifying the trained checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_s/s_tndp2_retention_distill_s8_s16_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT.parent
        / "runs/20_spatial_importance/S_TNDP2_RETENTION_DISTILLATION/seed0/best_stg1.pth",
    )
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent
        / "reports/20_spatial_importance/S_TNDP2_RETENTION_DISTILLATION/failure_analysis.json",
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


def keep_norms_in_eval(module):
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()


def make_gate(prediction, targets, radius):
    gates = []
    for target in targets:
        masks = target["masks"]
        if masks.numel() == 0:
            union = prediction.new_zeros((1, *masks.shape[-2:]))
        else:
            union = masks.float().amax(dim=0, keepdim=True).to(prediction.device)
        gate = F.interpolate(
            union.unsqueeze(0), size=prediction.shape[-2:], mode="area"
        ).squeeze(0)
        if radius > 0:
            gate = F.max_pool2d(
                gate.unsqueeze(0),
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            ).squeeze(0)
        gates.append(gate.clamp(0.0, 1.0))
    return torch.stack(gates)


def weighted_smooth_l1(prediction, target, gate):
    regression = F.smooth_l1_loss(
        prediction.float(), target.float(), reduction="none", beta=0.25
    )
    denominator = (gate.sum() * prediction.shape[1]).clamp_min(1.0)
    return regression.mul(gate).sum() / denominator


def weighted_channel_mean(value, gate, per_image):
    weight = gate.expand(-1, value.shape[1], -1, -1)
    dims = (-2, -1) if per_image else (0, -2, -1)
    denominator = weight.sum(dims, keepdim=True).clamp_min(1e-8)
    return (value * weight).sum(dims, keepdim=True) / denominator


def weighted_correlation(prediction, target, gate, centered_per_image=False):
    weight = gate.expand(-1, prediction.shape[1], -1, -1)
    if centered_per_image:
        prediction = prediction - weighted_channel_mean(prediction, gate, per_image=True)
        target = target - weighted_channel_mean(target, gate, per_image=True)
    weight_sum = weight.sum().clamp_min(1e-8)
    pred_mean = (prediction * weight).sum() / weight_sum
    target_mean = (target * weight).sum() / weight_sum
    pred_centered = prediction - pred_mean
    target_centered = target - target_mean
    covariance = (weight * pred_centered * target_centered).sum() / weight_sum
    pred_var = (weight * pred_centered.square()).sum() / weight_sum
    target_var = (weight * target_centered.square()).sum() / weight_sum
    correlation = covariance / (pred_var * target_var).sqrt().clamp_min(1e-8)
    return float(correlation), float(pred_var), float(target_var)


def gradient_l2(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squares = [gradient.float().square().sum() for gradient in gradients if gradient is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def main():
    args = parse_args()
    if args.batches <= 0:
        raise ValueError("--batches must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("TNDP2 failure analysis requires CUDA")

    device = torch.device("cuda")
    cfg = YAMLConfig(str(args.config))
    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema", {}).get("module", checkpoint["model"])
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")

    model.train()
    criterion.train()
    keep_norms_in_eval(model)
    loader = cfg.train_dataloader
    radius = int(criterion.tndp_neighborhood_radius)

    aggregate = {
        "trained": [],
        "zero": [],
        "batch_channel_constant": [],
        "image_channel_constant": [],
        "half_map_shift": [],
        "correlation": [],
        "spatial_correlation_after_image_channel_centering": [],
        "prediction_variance": [],
        "target_variance": [],
        "gate_fraction": [],
        "valid_images": [],
        "batch_sizes": [],
    }
    gradient_batch = None
    for batch_index, (samples, targets) in enumerate(loader):
        if batch_index >= args.batches:
            break
        samples = samples.to(device)
        targets = move_targets(targets, device)
        if gradient_batch is None:
            gradient_batch = (samples, targets)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            outputs = model(samples, targets=targets)
        prediction = outputs["tndp_detail_prediction"].float()
        target = outputs["tndp_detail_target"].float()
        gate = make_gate(prediction, targets, radius)
        batch_constant = weighted_channel_mean(target, gate, per_image=False).expand_as(target)
        image_constant = weighted_channel_mean(target, gate, per_image=True).expand_as(target)
        shifted = torch.roll(
            prediction,
            shifts=(prediction.shape[-2] // 2, prediction.shape[-1] // 2),
            dims=(-2, -1),
        )
        correlation, pred_var, target_var = weighted_correlation(prediction, target, gate)
        spatial_correlation, _, _ = weighted_correlation(
            prediction, target, gate, centered_per_image=True
        )
        aggregate["trained"].append(float(weighted_smooth_l1(prediction, target, gate)))
        aggregate["zero"].append(float(weighted_smooth_l1(torch.zeros_like(target), target, gate)))
        aggregate["batch_channel_constant"].append(
            float(weighted_smooth_l1(batch_constant, target, gate))
        )
        aggregate["image_channel_constant"].append(
            float(weighted_smooth_l1(image_constant, target, gate))
        )
        aggregate["half_map_shift"].append(float(weighted_smooth_l1(shifted, target, gate)))
        aggregate["correlation"].append(correlation)
        aggregate["spatial_correlation_after_image_channel_centering"].append(spatial_correlation)
        aggregate["prediction_variance"].append(pred_var)
        aggregate["target_variance"].append(target_var)
        aggregate["gate_fraction"].append(float(gate.mean()))
        aggregate["valid_images"].append(
            sum(int(target_item["masks"].numel() > 0 and target_item["masks"].any()) for target_item in targets)
        )
        aggregate["batch_sizes"].append(int(samples.shape[0]))

    if gradient_batch is None:
        raise RuntimeError("training loader produced no batches")
    samples, targets = gradient_batch
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(samples, targets=targets)
    with torch.autocast("cuda", enabled=False):
        losses = criterion(outputs, targets)
    detection_loss = sum(value for key, value in losses.items() if key != "loss_tndp_detail")
    tndp_loss = losses["loss_tndp_detail"]
    downsample_parameters = [
        parameter
        for parameter in model.backbone.stages[2].downsample.parameters()
        if parameter.requires_grad
    ]
    detection_gradient = gradient_l2(detection_loss, downsample_parameters, retain_graph=True)
    tndp_gradient = gradient_l2(tndp_loss, downsample_parameters, retain_graph=False)

    means = {
        key: float(sum(values) / len(values))
        for key, values in aggregate.items()
        if key != "valid_images"
    }
    report = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_last_epoch": int(checkpoint.get("last_epoch", -1)),
        "device": torch.cuda.get_device_name(0),
        "batches": len(aggregate["trained"]),
        "images": int(sum(aggregate["batch_sizes"])),
        "valid_mask_images": int(sum(aggregate["valid_images"])),
        "raw_smooth_l1": {
            key: means[key]
            for key in (
                "trained",
                "zero",
                "batch_channel_constant",
                "image_channel_constant",
                "half_map_shift",
            )
        },
        "trained_improvement_over_zero": 1.0 - means["trained"] / means["zero"],
        "trained_improvement_over_batch_channel_constant": 1.0
        - means["trained"] / means["batch_channel_constant"],
        "shift_penalty_relative_to_trained": means["half_map_shift"] / means["trained"] - 1.0,
        "weighted_correlation": means["correlation"],
        "spatial_correlation_after_image_channel_centering": means[
            "spatial_correlation_after_image_channel_centering"
        ],
        "prediction_to_target_variance_ratio": means["prediction_variance"]
        / max(means["target_variance"], 1e-12),
        "mean_gate_fraction": means["gate_fraction"],
        "weighted_tndp_loss_on_gradient_batch": float(tndp_loss.detach()),
        "detection_loss_on_gradient_batch": float(detection_loss.detach()),
        "stage2_downsample_gradient_l2": {
            "detection": detection_gradient,
            "tndp": tndp_gradient,
            "tndp_to_detection_ratio": tndp_gradient / max(detection_gradient, 1e-12),
        },
    }
    if not all(math.isfinite(value) for value in report["raw_smooth_l1"].values()):
        raise RuntimeError("non-finite diagnostic metrics")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
