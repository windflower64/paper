#!/usr/bin/env python3
"""Static, target, and gradient preflight for SABR1 JointInit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from preflight_s_rep1_spar import (
    gradient_cosine,
    gradient_l2,
    gradient_list,
    gradients_l2,
    load_compatible_tuning_weights,
    move_targets,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_s/s_sabr1_jointinit_cocotune_w025_b32_60e_local.yml",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=ROOT / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gradient-batch", type=int, default=2)
    parser.add_argument("--expected-batch", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--max-shared-gradient-ratio", type=float, default=-1.0)
    parser.add_argument("--min-shared-gradient-cosine", type=float, default=-1.1)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SABR1 preflight requires CUDA")
    device = torch.device("cuda")
    seed = 20260816

    torch.manual_seed(seed)
    baseline_cfg = YAMLConfig(str(args.baseline_config))
    baseline_cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    baseline_model = baseline_cfg.model

    torch.manual_seed(seed)
    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    baseline_matched = load_compatible_tuning_weights(baseline_model, args.checkpoint)
    model_matched = load_compatible_tuning_weights(model, args.checkpoint)

    baseline_state = baseline_model.state_dict()
    model_state = model.state_dict()
    missing_shared = [key for key in baseline_state if key not in model_state]
    shared_max_error = max(
        (
            0.0
            if baseline_state[key].dtype == torch.bool
            and torch.equal(baseline_state[key], model_state[key])
            else (
                1.0
                if baseline_state[key].dtype == torch.bool
                else float((baseline_state[key] - model_state[key]).abs().max())
            )
        )
        for key in baseline_state
        if key in model_state
    )
    if missing_shared or shared_max_error != 0.0:
        raise RuntimeError(
            f"SABR changed shared initialization: missing={missing_shared}, error={shared_max_error}"
        )
    del baseline_model, baseline_cfg, baseline_state

    samples, targets = next(iter(cfg.train_dataloader))
    if samples.shape[0] != args.expected_batch:
        raise RuntimeError(
            f"configured batch mismatch: expected {args.expected_batch}, got {samples.shape[0]}"
        )
    valid_indices = [
        index
        for index, target in enumerate(targets)
        if "masks" in target and bool(target["masks"].any())
    ]
    if len(valid_indices) < args.gradient_batch:
        raise RuntimeError("first batch lacks enough accepted SAM masks")
    chosen = valid_indices[: args.gradient_batch]
    train_samples = samples[chosen].to(device)
    train_targets = move_targets([targets[index] for index in chosen], device)

    model = model.to(device)
    criterion = cfg.criterion.to(device)
    model.set_training_epoch(args.epoch)
    if model.backbone.spar_enabled or model.backbone.spar_fusion is not None:
        raise RuntimeError("SABR config accidentally retained SPAR fusion")
    heads = model.backbone.sabr_heads
    if not model.backbone.sabr_enabled or heads is None:
        raise RuntimeError("SABR heads were not constructed")
    head_parameters = list(heads.parameters())
    optimizer_parameter_ids = {
        id(parameter)
        for group in cfg.optimizer.param_groups
        for parameter in group["params"]
    }
    if not set(map(id, head_parameters)).issubset(optimizer_parameter_ids):
        raise RuntimeError("optimizer omitted SABR head parameters")

    model.eval()
    with torch.no_grad():
        with_sabr = model(train_samples[:1])
        if model.backbone.sabr_outputs is not None:
            raise RuntimeError("SABR unexpectedly ran during evaluation")
        model.backbone.sabr_heads = None
        without_sabr = model(train_samples[:1])
        model.backbone.sabr_heads = heads
    box_error = float((with_sabr["pred_boxes"] - without_sabr["pred_boxes"]).abs().max())
    logit_error = float(
        (with_sabr["pred_logits"] - without_sabr["pred_logits"]).abs().max()
    )
    if box_error != 0.0 or logit_error != 0.0:
        raise RuntimeError(f"SABR changed eval outputs: boxes={box_error}, logits={logit_error}")

    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    criterion.train()
    outputs = model(train_samples, targets=train_targets)
    required = {"sabr_boundary_logits", "sabr_body_logits"}
    missing = required.difference(outputs)
    if missing:
        raise RuntimeError(f"model omitted SABR outputs: {sorted(missing)}")
    boundary_logits = outputs["sabr_boundary_logits"]
    body_logits = outputs["sabr_body_logits"]
    expected_boundary_size = tuple(value // 8 for value in train_samples.shape[-2:])
    expected_body_size = tuple(value // 16 for value in train_samples.shape[-2:])
    if boundary_logits.shape[-2:] != expected_boundary_size:
        raise RuntimeError(f"boundary head is not S8: {tuple(boundary_logits.shape)}")
    if body_logits.shape[-2:] != expected_body_size:
        raise RuntimeError(f"body head is not S16: {tuple(body_logits.shape)}")

    components = criterion._sabr_loss_components(
        boundary_logits, body_logits, train_targets
    )
    boundary_fraction = float(components["boundary_target"].mean())
    body_fraction = float(components["body_target"].mean())
    if not (0.001 < boundary_fraction < 0.75):
        raise RuntimeError(f"implausible boundary target fraction: {boundary_fraction}")
    if not (0.0001 < body_fraction < 0.75):
        raise RuntimeError(f"implausible body target fraction: {body_fraction}")
    if not bool(components["valid"].all()):
        raise RuntimeError("selected SABR samples unexpectedly became invalid")

    losses = criterion(outputs, train_targets, epoch=args.epoch)
    if "loss_sabr" not in losses:
        raise RuntimeError("criterion omitted loss_sabr")
    sabr_loss = losses["loss_sabr"]
    schedule = criterion._sabr_schedule(args.epoch)
    expected_loss = criterion.sabr_aux_weight * schedule * components["combined"]
    if not torch.allclose(sabr_loss, expected_loss, rtol=1e-6, atol=1e-7):
        raise RuntimeError("criterion applied an unexpected SABR weight")
    if not bool(torch.isfinite(sabr_loss)) or float(sabr_loss.detach()) <= 0:
        raise RuntimeError(f"invalid SABR loss: {float(sabr_loss.detach())}")

    head_gradient = gradient_l2(sabr_loss, head_parameters, retain_graph=True)
    stage_gradients = {
        f"stage{index + 1}": gradient_l2(
            sabr_loss, model.backbone.stages[index].parameters(), retain_graph=True
        )
        for index in (1, 2, 3)
    }
    encoder_gradient = gradient_l2(
        sabr_loss, model.encoder.parameters(), retain_graph=True
    )
    gradient_scale = float(model.backbone._spar_backbone_gradient_scale())
    if head_gradient <= 0:
        raise RuntimeError("SABR heads received no gradient")
    if gradient_scale == 0 and any(value != 0 for value in stage_gradients.values()):
        raise RuntimeError(f"SABR warmup leaked into backbone: {stage_gradients}")
    if gradient_scale > 0:
        if stage_gradients["stage2"] <= 0 or stage_gradients["stage3"] <= 0:
            raise RuntimeError(f"SABR missed S8/S16 paths: {stage_gradients}")
        if stage_gradients["stage4"] != 0:
            raise RuntimeError(f"SABR incorrectly supervised S32: {stage_gradients}")
    if encoder_gradient != 0:
        raise RuntimeError(f"SABR leaked into encoder: {encoder_gradient}")

    shared_parameters = list(model.backbone.stages[1].parameters()) + list(
        model.backbone.stages[2].parameters()
    )
    sabr_shared = gradient_list(sabr_loss, shared_parameters, retain_graph=True)
    detection_loss = sum(value for name, value in losses.items() if name != "loss_sabr")
    detection_shared = gradient_list(detection_loss, shared_parameters, retain_graph=False)
    sabr_l2 = gradients_l2(sabr_shared)
    detection_l2 = gradients_l2(detection_shared)
    ratio = sabr_l2 / detection_l2 if detection_l2 > 0 else float("inf")
    cosine = gradient_cosine(sabr_shared, detection_shared) if sabr_l2 > 0 else float("nan")
    if args.max_shared_gradient_ratio > 0 and ratio > args.max_shared_gradient_ratio:
        raise RuntimeError(f"SABR shared-gradient ratio too high: {ratio}")
    if sabr_l2 > 0 and cosine < args.min_shared_gradient_cosine:
        raise RuntimeError(f"SABR gradient is too opposed to detection: {cosine}")

    report = {
        "status": "PASS",
        "config": str(args.config),
        "tuning_checkpoint": str(args.checkpoint),
        "baseline_matched_tensors": baseline_matched,
        "model_matched_tensors": model_matched,
        "configured_train_batch": int(samples.shape[0]),
        "gradient_batch": int(train_samples.shape[0]),
        "tested_epoch": int(args.epoch),
        "shared_parameter_max_error": shared_max_error,
        "identity_box_error": box_error,
        "identity_logit_error": logit_error,
        "head_parameters": sum(parameter.numel() for parameter in head_parameters),
        "optimizer_covers_heads": True,
        "boundary_shape": list(boundary_logits.shape),
        "body_shape": list(body_logits.shape),
        "boundary_target_fraction": boundary_fraction,
        "body_target_fraction": body_fraction,
        "raw_boundary_loss": float(components["boundary"].detach()),
        "raw_body_loss": float(components["body"].detach()),
        "raw_combined_loss": float(components["combined"].detach()),
        "weighted_sabr_loss": float(sabr_loss.detach()),
        "sabr_schedule": schedule,
        "backbone_gradient_scale": gradient_scale,
        "head_gradient_l2": head_gradient,
        "stage_gradient_l2": stage_gradients,
        "encoder_gradient_l2": encoder_gradient,
        "sabr_shared_gradient_l2": sabr_l2,
        "detection_shared_gradient_l2": detection_l2,
        "shared_gradient_ratio": ratio,
        "shared_gradient_cosine": cosine if math.isfinite(cosine) else None,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "s32_receives_sabr_gradient": False,
        "inference_branch_removed_by_deploy": True,
    }
    model.deploy()
    if model.backbone.sabr_heads is not None or model.backbone.sabr_enabled:
        raise RuntimeError("deploy did not physically remove SABR")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
