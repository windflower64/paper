#!/usr/bin/env python3
"""Preflight BPC1 identity, supervision isolation, gradients, and batch32."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from src.nn.backbone.hgnetv2 import BoundaryPolyphaseCarrier, HGNetv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_s/s_bpc1_sam_boundary_polyphase_s8_s16_local.yml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent
        / "reports/20_spatial_importance/S_BPC1_SAM_BOUNDARY_POLYPHASE/preflight.json",
    )
    return parser.parse_args()


def gradient_l2(parameters):
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def autograd_l2(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    squares = [gradient.float().square().sum() for gradient in gradients if gradient is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("BPC1 preflight requires CUDA")
    device = torch.device("cuda")

    common = dict(
        name="B0",
        return_idx=[1, 2, 3],
        freeze_at=-1,
        freeze_norm=True,
        pretrained=False,
    )
    torch.manual_seed(20260815)
    baseline = HGNetv2(**common).eval().to(device)
    candidate = HGNetv2(
        **common,
        bpc_stage=2,
        bpc_detail_channels=32,
        bpc_project_groups=8,
        bpc_max_scale=0.25,
        bpc_init_scale=0.05,
    ).eval().to(device)
    incompatible = candidate.load_state_dict(baseline.state_dict(), strict=False)
    if incompatible.unexpected_keys or not incompatible.missing_keys:
        raise RuntimeError(f"unexpected baseline transfer mismatch: {incompatible}")
    if any(not key.startswith("bpc_branch.") for key in incompatible.missing_keys):
        raise RuntimeError(f"non-BPC missing keys: {incompatible.missing_keys}")
    sample = torch.randn(1, 3, 512, 640, device=device)
    with torch.no_grad():
        baseline_features = baseline(sample)
        candidate_features = candidate(sample)
    identity_error = max(
        float((base - test).abs().max())
        for base, test in zip(baseline_features, candidate_features)
    )
    if identity_error != 0.0:
        raise RuntimeError(f"zero-initialized BPC changed detector features: {identity_error}")

    phase_test = torch.randn(2, 7, 9, 11, device=device)
    phase_residual = BoundaryPolyphaseCarrier.phase_residual(phase_test)
    phase_zero_sum_error = float(phase_residual.sum(dim=2).abs().max())
    if phase_zero_sum_error > 1e-5:
        raise RuntimeError(f"BPC phase residual is not zero sum: {phase_zero_sum_error}")
    del baseline, candidate, baseline_features, candidate_features, sample, phase_test, phase_residual
    gc.collect()
    torch.cuda.empty_cache()

    cfg = YAMLConfig(str(args.config))
    loader = cfg.train_dataloader
    samples, targets = next(iter(loader))
    if samples.shape[0] != 32:
        raise RuntimeError(f"BPC1 requires batch32 preflight, got {samples.shape[0]}")
    samples = samples.to(device)
    targets = move_targets(targets, device)
    model = cfg.model.to(device).train()
    criterion = cfg.criterion.to(device).train()
    optimizer = cfg.optimizer
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    if len(optimizer_parameters) != len({id(parameter) for parameter in optimizer_parameters}):
        raise RuntimeError("optimizer contains duplicate BPC/model parameters")

    branch = model.backbone.bpc_branch
    branch_parameters = sum(parameter.numel() for parameter in branch.parameters())
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    baseline_parameters = model_parameters - branch_parameters

    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(samples, targets=targets)
    with torch.autocast("cuda", enabled=False):
        losses = criterion(outputs, targets)
    if "loss_bpc_boundary" not in losses:
        raise RuntimeError("BPC boundary loss missing")
    logits = outputs["bpc_boundary_logits"]
    boundary_target, valid = criterion._bpc_boundary_targets(logits.float(), targets)
    if logits.shape != (32, 1, 64, 80):
        raise RuntimeError(f"unexpected BPC logits shape: {tuple(logits.shape)}")
    if not bool(boundary_target.sum() > 0):
        raise RuntimeError("BPC preflight batch contains no valid boundary target")

    detection_loss = sum(value for key, value in losses.items() if key != "loss_bpc_boundary")
    locator_parameters = list(branch.locator.parameters())
    boundary_only_locator_gradient = autograd_l2(
        losses["loss_bpc_boundary"], locator_parameters, retain_graph=True
    )
    detection_to_locator_gradient = autograd_l2(
        detection_loss, locator_parameters, retain_graph=True
    )
    if boundary_only_locator_gradient <= 0:
        raise RuntimeError("BPC boundary supervision did not reach the locator")
    if detection_to_locator_gradient != 0.0:
        raise RuntimeError(
            "BPC detection gradient leaked into the locator gate: "
            f"{detection_to_locator_gradient}"
        )

    optimizer.zero_grad(set_to_none=True)
    total_loss = sum(losses.values())
    if not torch.isfinite(total_loss):
        raise RuntimeError(f"non-finite BPC loss: {losses}")
    total_loss.backward()
    gradients = {
        "reduce": gradient_l2(branch.reduce.parameters()),
        "locator": gradient_l2(branch.locator.parameters()),
        "project": gradient_l2(branch.project.parameters()),
        "standard_downsample": gradient_l2(model.backbone.stages[2].downsample.parameters()),
    }
    if any(value <= 0 for value in gradients.values()):
        raise RuntimeError(f"missing BPC gradients: {gradients}")
    initial_residual_ratio = float(branch.last_residual_rms_ratio)
    if initial_residual_ratio != 0.0:
        raise RuntimeError(
            "zero-initialized BPC residual must be exactly zero, got "
            f"{initial_residual_ratio}"
        )
    optimizer.step()

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        inference_outputs = model(samples)
    post_step_residual_ratio = float(branch.last_residual_rms_ratio)
    if post_step_residual_ratio <= 0:
        raise RuntimeError("BPC carrier did not leave zero after one optimizer step")
    if "pred_logits" not in inference_outputs or "pred_boxes" not in inference_outputs:
        raise RuntimeError("BPC inference without masks did not return detector outputs")

    report = {
        "config": str(args.config),
        "device": torch.cuda.get_device_name(0),
        "batch_size": int(samples.shape[0]),
        "image_shape": list(samples.shape),
        "identity_max_abs_error": identity_error,
        "phase_zero_sum_max_abs_error": phase_zero_sum_error,
        "bpc_logits_shape": list(logits.shape),
        "valid_boundary_samples": int(valid.sum().item()),
        "ignored_rejected_positive_samples": int((1 - valid).sum().item()),
        "boundary_target_mean": float(boundary_target.mean()),
        "loss_bpc_boundary": float(losses["loss_bpc_boundary"].detach()),
        "total_loss": float(total_loss.detach()),
        "boundary_only_locator_gradient_l2": boundary_only_locator_gradient,
        "detection_to_locator_gradient_l2": detection_to_locator_gradient,
        "total_gradients_l2": gradients,
        "initial_residual_rms_ratio": initial_residual_ratio,
        "post_step_residual_rms_ratio": post_step_residual_ratio,
        "branch_parameters": branch_parameters,
        "baseline_parameters": baseline_parameters,
        "branch_parameter_fraction": branch_parameters / baseline_parameters,
        "model_parameters": model_parameters,
        "optimizer_parameter_tensors": len(optimizer_parameters),
        "optimizer_unique_parameter_tensors": len({id(parameter) for parameter in optimizer_parameters}),
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "inference_without_masks": True,
        "all_losses_finite": all(bool(torch.isfinite(value)) for value in losses.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
