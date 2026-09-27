#!/usr/bin/env python3
"""Strict one-step preflight for frozen-detector S-QMI2 S4 mask warmup."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_s/s_sqmi2_s4_mask_warmup_b16_6e_testdev_local.yml",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=ROOT / "experiments/phase_m/c_only_gq1_b8a4_20e_testdev_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=WORKSPACE / "outputs/C_ONLY_GQ1_B8A4_20E_TESTDEV/seed0/best_stg1.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "reports/146_sqmi2_s4/preflight.json",
    )
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weights(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"]
    return checkpoint.get("model", checkpoint)


def load_matching(model, state):
    own = model.state_dict()
    matched = {key: value for key, value in state.items() if key in own and value.shape == own[key].shape}
    model.load_state_dict(matched, strict=False)
    return sorted(set(own).difference(matched))


def move_targets(targets, device):
    return [
        {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()}
        for target in targets
    ]


def clone_fixed_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("decoder.sqmi.")
    }


def max_shared_error(reference, candidate):
    candidate_state = candidate.state_dict()
    errors = []
    missing = []
    for key, value in reference.state_dict().items():
        if key not in candidate_state:
            missing.append(key)
            continue
        other = candidate_state[key]
        errors.append(0.0 if torch.equal(value, other) else float((value - other).abs().max()))
    return max(errors, default=0.0), missing


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    state = weights(args.checkpoint)

    seed_all(args.seed)
    baseline_cfg = YAMLConfig(str(args.baseline_config))
    baseline = baseline_cfg.model
    seed_all(args.seed)
    cfg = YAMLConfig(str(args.config))
    model = cfg.model
    baseline_missing = load_matching(baseline, state)
    model_missing = load_matching(model, state)
    allowed_missing = sorted(key for key in model.state_dict() if key.startswith("decoder.sqmi."))
    if baseline_missing or model_missing != allowed_missing:
        raise RuntimeError(
            f"Checkpoint mismatch: baseline={baseline_missing}, model={model_missing}, allowed={allowed_missing}"
        )
    shared_error, shared_missing = max_shared_error(baseline, model)
    if shared_error != 0.0 or shared_missing:
        raise RuntimeError(
            f"S-QMI2 changed shared C weights: error={shared_error}, missing={shared_missing}"
        )

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(not name.startswith("decoder.sqmi.") for name in trainable):
        raise RuntimeError(f"Non-SQMI parameter is trainable: {trainable}")
    if any(name.startswith("decoder.sqmi.gate.") for name in trainable):
        raise RuntimeError(f"Stage-A gate must remain frozen: {trainable}")

    samples, targets = next(iter(cfg.train_dataloader))
    if samples.shape[0] != 16:
        raise RuntimeError(f"Expected batch16, got {samples.shape[0]}")
    valid_masks = sum(int(bool(target["masks"].any())) for target in targets)

    model = model.cuda().eval()
    sqmi_module = model.decoder.sqmi
    with torch.no_grad():
        sqmi_output = model(samples[:1].cuda())
        model.decoder.sqmi = None
        model.decoder.sqmi_enabled = False
        base_output = model(samples[:1].cuda())
        model.decoder.sqmi = sqmi_module
        model.decoder.sqmi_enabled = True
    box_error = float((base_output["pred_boxes"] - sqmi_output["pred_boxes"]).abs().max())
    logit_error = float((base_output["pred_logits"] - sqmi_output["pred_logits"]).abs().max())
    if box_error != 0.0 or logit_error != 0.0:
        raise RuntimeError(f"Frozen warmup is not detector-identical: {box_error}, {logit_error}")
    del baseline, base_output, sqmi_output
    torch.cuda.empty_cache()

    fixed_before = clone_fixed_state(model)
    model.train()
    criterion = cfg.criterion.cuda().train()
    optimizer = cfg.optimizer
    samples = samples.cuda()
    targets = move_targets(targets, device)
    torch.cuda.reset_peak_memory_stats(device)
    output = model(samples, targets=targets)
    if output["sqmi_pixel_features"].shape != (16, 32, 128, 160):
        raise RuntimeError(f"Unexpected S4 feature shape: {tuple(output['sqmi_pixel_features'].shape)}")
    losses = criterion(output, targets, epoch=0, step=0, global_step=0, epoch_step=1)
    loss = sum(losses.values())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    non_sqmi_gradients = [
        name for name, parameter in model.named_parameters()
        if not name.startswith("decoder.sqmi.") and parameter.grad is not None
    ]
    if non_sqmi_gradients:
        raise RuntimeError(f"Detector received gradients: {non_sqmi_gradients}")
    optimizer.step()
    fixed_after = clone_fixed_state(model)
    changed_fixed = [key for key in fixed_before if not torch.equal(fixed_before[key], fixed_after[key])]
    if changed_fixed:
        raise RuntimeError(f"Frozen detector state changed: {changed_fixed}")

    result = {
        "status": "pass",
        "formal_batch_size": 16,
        "valid_sam_samples": valid_masks,
        "trainable_parameter_names": trainable,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "pixel_feature_shape": list(output["sqmi_pixel_features"].shape),
        "loss_sqmi": float(losses["loss_sqmi"].detach()),
        "detector_box_identity_error": box_error,
        "detector_logit_identity_error": logit_error,
        "shared_checkpoint_weight_error": shared_error,
        "non_sqmi_gradient_count": len(non_sqmi_gradients),
        "changed_frozen_state_count": len(changed_fixed),
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "allowed_checkpoint_missing": allowed_missing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
