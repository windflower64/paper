#!/usr/bin/env python3
"""Fairness, identity, gradient, and batch-16 preflight for SIBR1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

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
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=ROOT / "experiments/phase_s/s_rep1_1_a00ft6_lr02_control_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT.parent
        / "outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent
        / "reports/21_public_reproduction/S_SIBR1/preflight_batch16.json",
    )
    parser.add_argument("--gradient-batch", type=int, default=16)
    return parser.parse_args()


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def gradient_l2(loss, parameters, retain_graph=True):
    gradients = torch.autograd.grad(
        loss,
        list(parameters),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squares = [
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def load_ema(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema", {}).get("module", checkpoint["model"])
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "A00 checkpoint is incompatible: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SBRD1 preflight requires CUDA")
    device = torch.device("cuda")

    # SBRD contains no module, so identical seeds must give a bitwise-identical
    # detector before loading the common A00 checkpoint.
    torch.manual_seed(20260815)
    baseline_cfg = YAMLConfig(str(args.baseline_config))
    baseline_state = baseline_cfg.model.state_dict()
    torch.manual_seed(20260815)
    cfg = YAMLConfig(str(args.config))
    model = cfg.model
    model_state = model.state_dict()
    missing = [key for key in baseline_state if key not in model_state]
    extra = [key for key in model_state if key not in baseline_state]
    max_error = 0.0
    for key in baseline_state.keys() & model_state.keys():
        left, right = baseline_state[key], model_state[key]
        if left.dtype == torch.bool:
            error = 0.0 if torch.equal(left, right) else 1.0
        else:
            error = float((left - right).abs().max())
        max_error = max(max_error, error)
    if missing or extra or max_error != 0.0:
        raise RuntimeError(
            "SBRD changed detector initialization: "
            f"missing={missing}, extra={extra}, max_error={max_error}"
        )
    baseline_parameter_count = sum(
        parameter.numel() for parameter in baseline_cfg.model.parameters()
    )
    del baseline_cfg, baseline_state

    load_ema(model, args.checkpoint)
    samples, targets = next(iter(cfg.train_dataloader))
    if samples.shape[0] != 16:
        raise RuntimeError(f"SBRD1 config must use batch16, got {samples.shape[0]}")
    accepted = [
        index
        for index, target in enumerate(targets)
        if "masks" in target and bool(target["masks"].any())
    ]
    if args.gradient_batch == samples.shape[0]:
        chosen = list(range(samples.shape[0]))
    elif len(accepted) >= args.gradient_batch:
        chosen = accepted[: args.gradient_batch]
    else:
        raise RuntimeError("first batch lacks enough accepted SAM masks")
    train_samples = samples[chosen].to(device)
    train_targets = move_targets([targets[index] for index in chosen], device)

    model = model.to(device)
    criterion = cfg.criterion.to(device)
    if not model.backbone.sbrd_enabled:
        raise RuntimeError("SBRD flag is disabled")
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if model_parameter_count != baseline_parameter_count:
        raise RuntimeError(
            f"SBRD added parameters: {model_parameter_count - baseline_parameter_count}"
        )

    # Eval must never create the auxiliary tensors and must be identical when
    # the flag is toggled off.
    model.eval()
    eval_samples = train_samples[:1]
    with torch.no_grad():
        enabled_outputs = model(eval_samples)
        if model.backbone.sbrd_stage_features is not None:
            raise RuntimeError("SBRD unexpectedly ran during evaluation")
        model.backbone.sbrd_enabled = False
        disabled_outputs = model(eval_samples)
        model.backbone.sbrd_enabled = True
    box_error = float(
        (enabled_outputs["pred_boxes"] - disabled_outputs["pred_boxes"]).abs().max()
    )
    logit_error = float(
        (enabled_outputs["pred_logits"] - disabled_outputs["pred_logits"]).abs().max()
    )
    if box_error != 0.0 or logit_error != 0.0:
        raise RuntimeError(
            f"SBRD changed eval predictions: boxes={box_error}, logits={logit_error}"
        )

    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    criterion.train()
    outputs = model(train_samples, targets=train_targets)
    features = outputs.get("sbrd_stage_features")
    if features is None:
        raise RuntimeError("model did not expose S8/S16 features")
    losses = criterion(outputs, train_targets, epoch=0)
    loss = losses.get("loss_sbrd")
    if loss is None or not bool(torch.isfinite(loss)) or float(loss.detach()) <= 0:
        raise RuntimeError(f"invalid SBRD loss: {loss}")
    raw_loss = criterion._sbrd_loss(features, train_targets)
    expected = criterion.sbrd_aux_weight * raw_loss
    if not torch.allclose(loss, expected, rtol=1e-6, atol=1e-7):
        raise RuntimeError("SBRD weight was applied incorrectly")

    shifted_targets = []
    for target in train_targets:
        shifted = dict(target)
        shifted["masks"] = torch.roll(
            target["masks"], shifts=target["masks"].shape[-1] // 2, dims=-1
        )
        shifted_targets.append(shifted)
    shifted_loss = criterion._sbrd_loss(features, shifted_targets)
    sensitivity = float((raw_loss - shifted_loss).abs().detach())
    if sensitivity <= 1e-8:
        raise RuntimeError("SBRD is insensitive to mask location")

    stage_gradient_l2 = {
        f"stage{index + 1}": gradient_l2(
            loss,
            model.backbone.stages[index].parameters(),
            retain_graph=True,
        )
        for index in range(4)
    }
    encoder_gradient = gradient_l2(
        loss, model.encoder.parameters(), retain_graph=False
    )
    if stage_gradient_l2["stage2"] <= 0 or stage_gradient_l2["stage3"] <= 0:
        raise RuntimeError(f"SBRD missed the S8->S16 backbone path: {stage_gradient_l2}")
    if encoder_gradient != 0.0:
        raise RuntimeError(f"SBRD leaked into the encoder: {encoder_gradient}")

    report = {
        "status": "PASS",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "configured_train_batch": int(samples.shape[0]),
        "gradient_batch": int(args.gradient_batch),
        "accepted_masks_in_batch": len(accepted),
        "model_parameters": model_parameter_count,
        "added_parameters": model_parameter_count - baseline_parameter_count,
        "shared_parameter_max_error_same_seed": max_error,
        "identity_box_error": box_error,
        "identity_logit_error": logit_error,
        "s8_shape": list(features[0].shape),
        "s16_shape": list(features[1].shape),
        "sbrd_aux_weight": float(criterion.sbrd_aux_weight),
        "region_mode": criterion.sbrd_region_mode,
        "raw_aligned_loss_before_training": float(raw_loss.detach()),
        "shifted_mask_loss_before_training": float(shifted_loss.detach()),
        "absolute_mask_sensitivity": sensitivity,
        "stage_gradient_l2": stage_gradient_l2,
        "encoder_gradient_l2": encoder_gradient,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
