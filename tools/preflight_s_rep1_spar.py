#!/usr/bin/env python3
"""Static and gradient preflight for the FALCON-SPAR reproduction."""

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
        / "experiments/phase_s/s_rep1_spar_sam_feature_regularization_local.yml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent
        / "reports/21_public_reproduction/S_REP1_SPAR/preflight.json",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=ROOT / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument("--gradient-batch", type=int, default=2)
    parser.add_argument("--expected-batch", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--max-shared-gradient-ratio", type=float, default=-1.0)
    parser.add_argument("--min-shared-gradient-cosine", type=float, default=-1.1)
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


def gradient_l2(loss, parameters, retain_graph=True):
    parameters = list(parameters)
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squares = [
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def gradient_list(loss, parameters, retain_graph=True):
    gradients = torch.autograd.grad(
        loss,
        list(parameters),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return [
        gradient.detach().float() if gradient is not None else None
        for gradient in gradients
    ]


def gradients_l2(gradients):
    squares = [gradient.square().sum() for gradient in gradients if gradient is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def gradient_cosine(left, right):
    pairs = [
        (left_gradient, right_gradient)
        for left_gradient, right_gradient in zip(left, right)
        if left_gradient is not None and right_gradient is not None
    ]
    if not pairs:
        return float("nan")
    dot = torch.stack([(left_value * right_value).sum() for left_value, right_value in pairs]).sum()
    left_norm = torch.stack([left_value.square().sum() for left_value, _ in pairs]).sum().sqrt()
    right_norm = torch.stack([right_value.square().sum() for _, right_value in pairs]).sum().sqrt()
    return float(dot / (left_norm * right_norm).clamp_min(1e-12))


def load_compatible_tuning_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("ema", {}).get("module")
    if not isinstance(source, dict):
        source = checkpoint.get("model")
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
    if any(key not in matched for key in required):
        raise RuntimeError("COCO tuning checkpoint missed encoder/decoder tensors")
    model.load_state_dict(matched, strict=False)
    error = max(
        float((model.state_dict()[key] - source[key]).abs().max()) for key in required
    )
    if error != 0.0:
        raise RuntimeError(f"COCO detector tuning load error: {error}")
    return len(matched)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("REP1-SPAR preflight requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(20260815)

    baseline_cfg = YAMLConfig(str(args.baseline_config))
    if args.checkpoint is not None:
        baseline_cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    baseline_model = baseline_cfg.model
    torch.manual_seed(20260815)
    cfg = YAMLConfig(str(args.config))
    if args.checkpoint is not None:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    baseline_matched_tensors = None
    model_matched_tensors = None
    if args.checkpoint is not None:
        baseline_matched_tensors = load_compatible_tuning_weights(
            baseline_model, args.checkpoint
        )
        model_matched_tensors = load_compatible_tuning_weights(model, args.checkpoint)
    model_state = model.state_dict()
    baseline_state = baseline_model.state_dict()
    shared_parameter_max_error = 0.0
    missing_shared_keys = []
    for key, baseline_value in baseline_state.items():
        if key not in model_state:
            missing_shared_keys.append(key)
            continue
        if baseline_value.dtype == torch.bool:
            error = 0.0 if torch.equal(model_state[key], baseline_value) else 1.0
        else:
            error = float((model_state[key] - baseline_value).abs().max())
        shared_parameter_max_error = max(shared_parameter_max_error, error)
    if missing_shared_keys or shared_parameter_max_error != 0.0:
        raise RuntimeError(
            "REP1 changed A00 initialization under the same seed: "
            f"missing={missing_shared_keys}, max_error={shared_parameter_max_error}"
        )
    del baseline_model, baseline_cfg, baseline_state

    samples, targets = next(iter(cfg.train_dataloader))
    if samples.shape[0] != args.expected_batch:
        raise RuntimeError(
            f"REP1 config must use batch{args.expected_batch}, got {samples.shape[0]}"
        )

    valid_indices = [
        index
        for index, target in enumerate(targets)
        if "masks" in target and bool(target["masks"].any())
    ]
    if args.gradient_batch == samples.shape[0]:
        chosen = list(range(samples.shape[0]))
    else:
        if len(valid_indices) < args.gradient_batch:
            raise RuntimeError("first training batch lacks enough accepted SAM masks")
        chosen = valid_indices[: args.gradient_batch]
    train_samples = samples[chosen].to(device)
    train_targets = move_targets([targets[index] for index in chosen], device)

    model = model.to(device)
    criterion = cfg.criterion.to(device)
    model.set_training_epoch(args.epoch)
    fusion = model.backbone.spar_fusion
    if not model.backbone.spar_enabled or fusion is None:
        raise RuntimeError("REP1 config did not construct SPAR fusion")
    fusion_parameters = list(fusion.parameters())
    fusion_parameter_count = sum(parameter.numel() for parameter in fusion_parameters)

    optimizer_parameters = {
        id(parameter)
        for group in cfg.optimizer.param_groups
        for parameter in group["params"]
    }
    if not set(map(id, fusion_parameters)).issubset(optimizer_parameters):
        raise RuntimeError("optimizer omitted SPAR fusion parameters")

    # Eval must skip the training-only branch and preserve detector outputs.
    model.eval()
    eval_samples = train_samples[:1]
    with torch.no_grad():
        with_spar = model(eval_samples)
        if model.backbone.spar_fused_features is not None:
            raise RuntimeError("SPAR fusion unexpectedly ran during eval")
        model.backbone.spar_fusion = None
        without_spar = model(eval_samples)
        model.backbone.spar_fusion = fusion
    identity_box_error = float(
        (with_spar["pred_boxes"] - without_spar["pred_boxes"]).abs().max()
    )
    identity_logit_error = float(
        (with_spar["pred_logits"] - without_spar["pred_logits"]).abs().max()
    )
    if identity_box_error != 0.0 or identity_logit_error != 0.0:
        raise RuntimeError(
            "training-only SPAR changed eval outputs: "
            f"boxes={identity_box_error}, logits={identity_logit_error}"
        )

    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    criterion.train()
    outputs = model(train_samples, targets=train_targets)
    if "spar_fused_features" not in outputs:
        raise RuntimeError("model did not expose SPAR fused features during training")
    losses = criterion(outputs, train_targets, epoch=args.epoch)
    if "loss_spar" not in losses:
        raise RuntimeError("criterion did not return loss_spar")
    spar_loss = losses["loss_spar"]
    raw_spar_loss = criterion._spar_loss(
        outputs["spar_fused_features"], train_targets
    )
    spar_schedule = criterion._spar_schedule(args.epoch)
    gradient_scale = model.backbone._spar_backbone_gradient_scale()
    expected_weighted_loss = (
        criterion.spar_aux_weight * spar_schedule * raw_spar_loss
    )
    if not torch.allclose(spar_loss, expected_weighted_loss, rtol=1e-6, atol=1e-7):
        raise RuntimeError(
            "criterion applied an unexpected SPAR weight: "
            f"actual={float(spar_loss.detach())}, "
            f"expected={float(expected_weighted_loss.detach())}"
        )
    if not bool(torch.isfinite(spar_loss)) or float(spar_loss.detach()) <= 0:
        raise RuntimeError(f"invalid SPAR loss: {float(spar_loss.detach())}")

    shifted_targets = []
    for target in train_targets:
        shifted = dict(target)
        shifted["masks"] = torch.roll(
            target["masks"], shifts=target["masks"].shape[-1] // 2, dims=-1
        )
        shifted_targets.append(shifted)
    shifted_loss = criterion._spar_loss(
        outputs["spar_fused_features"], shifted_targets
    )
    if float((raw_spar_loss - shifted_loss).abs().detach()) <= 1e-8:
        raise RuntimeError("SPAR loss is insensitive to a shifted mask")

    stage_gradient_l2 = {}
    for stage_index in (1, 2, 3):
        stage_gradient_l2[f"stage{stage_index + 1}"] = gradient_l2(
            spar_loss,
            model.backbone.stages[stage_index].parameters(),
            retain_graph=True,
        )
    fusion_gradient_l2 = gradient_l2(
        spar_loss, fusion_parameters, retain_graph=True
    )
    encoder_gradient_l2 = gradient_l2(
        spar_loss, model.encoder.parameters(), retain_graph=True
    )
    if fusion_gradient_l2 <= 0:
        raise RuntimeError(
            f"SPAR missed its private fusion head: fusion={fusion_gradient_l2}"
        )
    if gradient_scale == 0.0 and any(value != 0.0 for value in stage_gradient_l2.values()):
        raise RuntimeError(
            "SPAR warmup leaked into the backbone: "
            f"scale={gradient_scale}, stages={stage_gradient_l2}"
        )
    if gradient_scale > 0.0 and any(value <= 0 for value in stage_gradient_l2.values()):
        raise RuntimeError(
            "SPAR missed its intended feature path: "
            f"scale={gradient_scale}, stages={stage_gradient_l2}"
        )
    if encoder_gradient_l2 != 0:
        raise RuntimeError(f"SPAR leaked beyond the backbone: {encoder_gradient_l2}")

    shared_parameters = list(model.backbone.stages[1:].parameters())
    spar_shared_gradients = gradient_list(
        spar_loss, shared_parameters, retain_graph=True
    )
    detection_loss = sum(
        value for name, value in losses.items() if name != "loss_spar"
    )
    detection_shared_gradients = gradient_list(
        detection_loss, shared_parameters, retain_graph=False
    )
    spar_shared_gradient_l2 = gradients_l2(spar_shared_gradients)
    detection_shared_gradient_l2 = gradients_l2(detection_shared_gradients)
    shared_gradient_ratio = (
        spar_shared_gradient_l2 / detection_shared_gradient_l2
        if detection_shared_gradient_l2 > 0
        else float("inf")
    )
    shared_gradient_cosine = (
        gradient_cosine(spar_shared_gradients, detection_shared_gradients)
        if spar_shared_gradient_l2 > 0
        else float("nan")
    )
    if (
        args.max_shared_gradient_ratio > 0
        and shared_gradient_ratio > args.max_shared_gradient_ratio
    ):
        raise RuntimeError(
            "SPAR shared-gradient ratio exceeds its ceiling: "
            f"{shared_gradient_ratio} > {args.max_shared_gradient_ratio}"
        )
    if (
        spar_shared_gradient_l2 > 0
        and shared_gradient_cosine < args.min_shared_gradient_cosine
    ):
        raise RuntimeError(
            "SPAR is too strongly opposed to detection: "
            f"cosine={shared_gradient_cosine}"
        )

    report = {
        "status": "PASS",
        "config": str(args.config),
        "tuning_checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "baseline_matched_tensors": baseline_matched_tensors,
        "model_matched_tensors": model_matched_tensors,
        "gradient_batch": int(args.gradient_batch),
        "tested_epoch": int(args.epoch),
        "configured_train_batch": int(samples.shape[0]),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "spar_fusion_parameters": fusion_parameter_count,
        "identity_box_error": identity_box_error,
        "identity_logit_error": identity_logit_error,
        "shared_parameter_max_error_vs_a00_same_seed": shared_parameter_max_error,
        "spar_aux_weight": float(criterion.spar_aux_weight),
        "spar_loss_resolution": criterion.spar_loss_resolution,
        "spar_schedule": spar_schedule,
        "spar_backbone_gradient_scale": gradient_scale,
        "raw_spar_loss": float(raw_spar_loss.detach()),
        "weighted_spar_loss": float(spar_loss.detach()),
        "shifted_mask_spar_loss": float(shifted_loss.detach()),
        "absolute_aligned_shifted_difference": float(
            (raw_spar_loss - shifted_loss).abs().detach()
        ),
        "fusion_gradient_l2": fusion_gradient_l2,
        "stage_gradient_l2": stage_gradient_l2,
        "encoder_gradient_l2": encoder_gradient_l2,
        "spar_shared_gradient_l2": spar_shared_gradient_l2,
        "detection_shared_gradient_l2": detection_shared_gradient_l2,
        "spar_detection_shared_gradient_ratio": shared_gradient_ratio,
        "spar_detection_shared_gradient_cosine": (
            shared_gradient_cosine if math.isfinite(shared_gradient_cosine) else None
        ),
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "official_source_commit": "8ebd68c",
        "official_source_return_value": 0,
        "reproduction_return_value": "paper L1 + Dice loss",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
