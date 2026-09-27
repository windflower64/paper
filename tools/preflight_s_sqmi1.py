#!/usr/bin/env python3
"""No-training preflight for the paired S-QMI1 AUX/INIT experiment."""

from __future__ import annotations

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


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=ROOT / "experiments/phase_m/c_only_gq1_b8a4_20e_testdev_local.yml",
    )
    parser.add_argument(
        "--aux-config",
        type=Path,
        default=ROOT / "experiments/phase_s/s_sqmi1_aux_c_gq1_b16a2_12e_testdev_local.yml",
    )
    parser.add_argument(
        "--init-config",
        type=Path,
        default=ROOT / "experiments/phase_s/s_sqmi1_init_c_gq1_b16a2_12e_testdev_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=WORKSPACE
        / "outputs/C_ONLY_GQ1_B8A4_20E_TESTDEV/seed0/best_stg1.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "reports/144_sqmi1_preflight/preflight.json",
    )
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--gradient-batch", type=int, default=2)
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_weights(checkpoint):
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"], "ema.module"
    model = checkpoint.get("model")
    if isinstance(model, dict):
        return model, "model"
    return checkpoint, "root"


def load_matching(model, weights):
    state = model.state_dict()
    matched = {
        key: value
        for key, value in weights.items()
        if key in state and state[key].shape == value.shape
    }
    model.load_state_dict(matched, strict=False)
    return sorted(set(state).difference(matched))


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


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("S-QMI1 preflight requires CUDA")
    device = torch.device("cuda")

    seed_all(args.seed)
    baseline_cfg = YAMLConfig(str(args.baseline_config))
    baseline = baseline_cfg.model
    seed_all(args.seed)
    aux_cfg = YAMLConfig(str(args.aux_config))
    aux = aux_cfg.model
    seed_all(args.seed)
    init_cfg = YAMLConfig(str(args.init_config))
    init = init_cfg.model

    shared_error, shared_missing = max_shared_error(baseline, aux)
    pair_error, pair_missing = max_shared_error(aux, init)
    if shared_error != 0.0 or shared_missing:
        raise RuntimeError(
            f"SQMI changed shared initialization: error={shared_error}, missing={shared_missing}"
        )
    if pair_error != 0.0 or pair_missing or aux.state_dict().keys() != init.state_dict().keys():
        raise RuntimeError(
            f"AUX/INIT initialization mismatch: error={pair_error}, missing={pair_missing}"
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights, weight_source = checkpoint_weights(checkpoint)
    baseline_missing = load_matching(baseline, weights)
    aux_missing = load_matching(aux, weights)
    init_missing = load_matching(init, weights)
    allowed_missing = sorted(
        key for key in aux.state_dict() if key.startswith("decoder.sqmi.")
    )
    if baseline_missing:
        raise RuntimeError(f"Baseline checkpoint mismatch: {baseline_missing}")
    if aux_missing != allowed_missing or init_missing != allowed_missing:
        raise RuntimeError(
            "Unexpected checkpoint misses: "
            f"aux={aux_missing}, init={init_missing}, allowed={allowed_missing}"
        )

    samples, targets = next(iter(aux_cfg.train_dataloader))
    if samples.shape[0] != 16:
        raise RuntimeError(f"Formal S-QMI1 config must load batch16, got {samples.shape[0]}")
    valid_mask_samples = sum(
        int("masks" in target and bool(target["masks"].any())) for target in targets
    )
    if valid_mask_samples == 0:
        raise RuntimeError("The first formal batch contains no accepted SAM mask")

    probe = samples[:1].to(device)
    baseline = baseline.to(device).eval()
    aux = aux.to(device).eval()
    init = init.to(device).eval()
    with torch.no_grad():
        baseline_output = baseline(probe)
        aux_output = aux(probe)
        init_output = init(probe)
    aux_box_error = float((baseline_output["pred_boxes"] - aux_output["pred_boxes"]).abs().max())
    aux_logit_error = float((baseline_output["pred_logits"] - aux_output["pred_logits"]).abs().max())
    if aux_box_error != 0.0 or aux_logit_error != 0.0:
        raise RuntimeError(
            f"AUX is not exact C identity: boxes={aux_box_error}, logits={aux_logit_error}"
        )
    init_box_delta = float((init_output["pred_boxes"] - baseline_output["pred_boxes"]).abs().max())

    del baseline, aux, baseline_output, aux_output, init_output
    torch.cuda.empty_cache()

    gradient_batch = min(args.gradient_batch, samples.shape[0])
    train_samples = samples[:gradient_batch].to(device)
    train_targets = move_targets(targets[:gradient_batch], device)
    init.train()
    criterion = init_cfg.criterion.to(device).train()
    outputs = init(train_samples, targets=train_targets)
    losses = criterion(outputs, train_targets, epoch=0, step=0, global_step=0, epoch_step=1)
    if "loss_sqmi" not in losses or not torch.isfinite(losses["loss_sqmi"]):
        raise RuntimeError("S-QMI1 mask loss is missing or non-finite")
    total_loss = sum(losses.values())
    gate_parameters = list(init.decoder.sqmi.gate.parameters())
    gate_gradients = torch.autograd.grad(
        total_loss, gate_parameters, allow_unused=True, retain_graph=False
    )
    gate_grad_norm = float(
        torch.stack([gradient.float().square().sum() for gradient in gate_gradients if gradient is not None])
        .sum()
        .sqrt()
    )
    if not gate_grad_norm > 0.0:
        raise RuntimeError("Detection losses do not reach the S-QMI1 reliability gate")

    diagnostics = init.decoder.last_sqmi_diagnostics
    result = {
        "status": "pass",
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_weight_source": weight_source,
        "formal_batch_size": int(samples.shape[0]),
        "gradient_probe_batch": gradient_batch,
        "valid_sam_mask_samples": valid_mask_samples,
        "shared_initialization_max_error": shared_error,
        "aux_init_pair_max_error": pair_error,
        "aux_identity_box_error": aux_box_error,
        "aux_identity_logit_error": aux_logit_error,
        "untrained_init_max_box_delta": init_box_delta,
        "loss_sqmi": float(losses["loss_sqmi"].detach()),
        "gate_gradient_norm": gate_grad_norm,
        "valid_query_mask_ratio": float(diagnostics["valid_mask"].float().mean()),
        "mean_initial_mix": float(diagnostics["mix"].detach().mean()),
        "max_initial_mix": float(diagnostics["mix"].detach().max()),
        "allowed_checkpoint_missing": allowed_missing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
