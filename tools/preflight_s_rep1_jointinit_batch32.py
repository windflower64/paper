#!/usr/bin/env python3
"""Full-batch AMP forward/backward/update smoke test for REP1.1 JointInit."""

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "experiments/phase_s/s_rep1_1_jointinit_progressive_w025_b32_60e_local.yml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent
        / "reports/21_public_reproduction/S_REP1_1_JOINTINIT/preflight_full_batch32.json",
    )
    parser.add_argument("--expected-batch", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--amp-init-scale", type=float, default=1024.0)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def gradient_l2(parameters):
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def load_compatible_tuning_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("ema", {}).get("module")
    source_name = "ema.module"
    if not isinstance(source, dict):
        source = checkpoint.get("model")
        source_name = "model"
    if not isinstance(source, dict):
        raise RuntimeError(f"no model weights in {checkpoint_path}")
    own = model.state_dict()
    matched = {
        key: value
        for key, value in source.items()
        if key in own and own[key].shape == value.shape
    }
    required = (
        "encoder.input_proj.0.conv.weight",
        "decoder.dec_bbox_head.0.layers.0.weight",
    )
    missing_required = [key for key in required if key not in matched]
    if missing_required:
        raise RuntimeError(
            f"COCO tuning checkpoint missed detector tensors: {missing_required}"
        )
    incompatible = model.load_state_dict(matched, strict=False)
    max_required_error = max(
        float((model.state_dict()[key] - source[key]).abs().max()) for key in required
    )
    if max_required_error != 0.0:
        raise RuntimeError(
            f"COCO detector tensors were not loaded exactly: {max_required_error}"
        )
    return {
        "path": str(checkpoint_path),
        "weight_source": source_name,
        "matched_tensor_count": len(matched),
        "missing_tensor_count": len(incompatible.missing_keys),
        "unexpected_tensor_count": len(incompatible.unexpected_keys),
        "required_detector_tensor_max_error": max_required_error,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("REP1.1 JointInit full-batch preflight requires CUDA")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    cfg = YAMLConfig(str(args.config))
    if args.checkpoint is not None:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    tuning_report = (
        load_compatible_tuning_weights(model, args.checkpoint)
        if args.checkpoint is not None
        else None
    )
    model = model.to(device).train()
    criterion = cfg.criterion.to(device).train()
    optimizer = cfg.optimizer
    model.set_training_epoch(args.epoch)

    samples, targets = next(iter(cfg.train_dataloader))
    if samples.shape[0] != args.expected_batch:
        raise RuntimeError(
            f"configured batch mismatch: expected {args.expected_batch}, got {samples.shape[0]}"
        )
    samples = samples.to(device)
    targets = move_targets(targets, device)

    if getattr(model.backbone, "sabr_enabled", False):
        auxiliary_name = "SABR"
        auxiliary_module = model.backbone.sabr_heads
        auxiliary_loss_name = "loss_sabr"
    else:
        auxiliary_name = "SPAR"
        auxiliary_module = model.backbone.spar_fusion
        auxiliary_loss_name = "loss_spar"
    fusion_parameters = list(auxiliary_module.parameters())
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if not set(map(id, fusion_parameters)).issubset(optimizer_parameters):
        raise RuntimeError(f"optimizer omitted {auxiliary_name} head parameters")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler(
        enabled=True, init_scale=float(args.amp_init_scale)
    )

    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(samples, targets=targets)
    with torch.autocast("cuda", enabled=False):
        losses = criterion(
            outputs,
            targets,
            epoch=args.epoch,
            step=0,
            global_step=0,
            epoch_step=len(cfg.train_dataloader),
        )
        total_loss = sum(losses.values())

    if auxiliary_loss_name not in losses:
        raise RuntimeError(f"criterion omitted {auxiliary_loss_name}")

    if not torch.isfinite(total_loss):
        raise RuntimeError(f"non-finite loss: {float(total_loss.detach())}")
    if not torch.isfinite(outputs["pred_boxes"]).all():
        raise RuntimeError("non-finite predicted boxes")
    if not torch.isfinite(outputs["pred_logits"]).all():
        raise RuntimeError("non-finite predicted logits")

    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    trainable_with_grad = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    nonfinite_gradient_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
    ]
    gradients_finite = not nonfinite_gradient_names
    if not gradients_finite:
        raise RuntimeError(
            "non-finite gradient in the full batch: "
            f"amp_init_scale={args.amp_init_scale}, "
            f"parameters={nonfinite_gradient_names[:20]}"
        )

    fusion_gradient = gradient_l2(fusion_parameters)
    if fusion_gradient <= 0:
        raise RuntimeError(f"{auxiliary_name} head received no gradient")
    stage_spar_scale = float(model.backbone._spar_backbone_gradient_scale())

    # The first AdamW step allocates optimizer states, so the peak reflects an
    # actual training iteration rather than forward/backward alone.
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)

    report = {
        "status": "PASS",
        "config": str(args.config),
        "tuning_checkpoint": tuning_report,
        "tested_epoch": int(args.epoch),
        "batch": int(samples.shape[0]),
        "sample_shape": list(samples.shape),
        "accepted_sam_masks_in_batch": sum(
            int("masks" in target and bool(target["masks"].any()))
            for target in targets
        ),
        "total_loss": float(total_loss.detach()),
        "losses": {
            name: float(value.detach()) for name, value in losses.items()
        },
        "all_losses_finite": all(
            math.isfinite(float(value.detach())) for value in losses.values()
        ),
        "trainable_gradient_tensors": len(trainable_with_grad),
        "all_gradients_finite": gradients_finite,
        "auxiliary_method": auxiliary_name,
        "auxiliary_loss_name": auxiliary_loss_name,
        "auxiliary_head_gradient_l2": fusion_gradient,
        "spar_backbone_gradient_scale": stage_spar_scale,
        "optimizer_covers_auxiliary_head": True,
        "optimizer_step_completed": True,
        "amp_initial_scale": float(args.amp_init_scale),
        "amp_scale_after_step": float(scaler.get_scale()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "device": torch.cuda.get_device_name(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
