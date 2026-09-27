#!/usr/bin/env python3
"""Compare the P1 synthetic update with the configured AdamW first step."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from tools.preflight_s_bpc2_qrl_r1 import move_targets, selected_loss


def rms(tensor):
    return float(tensor.detach().float().square().mean().sqrt())


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("diagnostic requires CUDA")
    torch.manual_seed(20260815)
    device = torch.device("cuda")
    config = ROOT / "experiments/phase_s/s_bpc2_qrl_r1_detail_only_s8_s16_local.yml"
    cfg = YAMLConfig(str(config))
    samples, targets = next(iter(cfg.train_dataloader))
    model = cfg.model.to(device).train()
    criterion = cfg.criterion.to(device).train()
    configured_optimizer = cfg.optimizer
    qrl = model.decoder.qrl
    projection_parameters = list(qrl.side_projections.parameters())

    samples = samples[:2].to(device)
    targets = move_targets(targets[:2], device)
    outputs = model(samples, targets=targets)
    losses = criterion(outputs, targets, epoch=0)
    localization = selected_loss(
        losses, ("loss_bbox", "loss_giou", "loss_fgl", "loss_ddf")
    )
    gradients = torch.autograd.grad(localization, projection_parameters)
    side_tokens = qrl.last_side_tokens.detach()
    normalized_tokens = qrl.detail_norm(side_tokens)

    optimizer_group = next(
        group
        for group in configured_optimizer.param_groups
        if any(id(parameter) == id(projection_parameters[0]) for parameter in group["params"])
    )
    optimizer_hyperparameters = {
        "lr": float(optimizer_group["lr"]),
        "betas": list(optimizer_group["betas"]),
        "eps": float(optimizer_group["eps"]),
        "weight_decay": float(optimizer_group["weight_decay"]),
    }

    saved_weights = [parameter.detach().clone() for parameter in projection_parameters]
    with torch.no_grad():
        for parameter, gradient in zip(projection_parameters, gradients):
            parameter.copy_(saved_weights.pop(0) - optimizer_group["lr"] * gradient)
        sgd_delta_rms = rms(qrl.detail_only_delta_from_tokens(side_tokens))

    with torch.no_grad():
        for parameter in projection_parameters:
            parameter.zero_()
    adam = torch.optim.AdamW(
        projection_parameters,
        lr=optimizer_group["lr"],
        betas=optimizer_group["betas"],
        eps=optimizer_group["eps"],
        weight_decay=optimizer_group["weight_decay"],
    )
    for parameter, gradient in zip(projection_parameters, gradients):
        parameter.grad = gradient.detach().clone()
    adam.step()
    adam_delta_rms = rms(qrl.detail_only_delta_from_tokens(side_tokens))

    report = {
        "config": str(config),
        "localization_loss": float(localization.detach()),
        "gradient_l2": float(
            torch.stack([gradient.float().square().sum() for gradient in gradients])
            .sum()
            .sqrt()
        ),
        "gradient_rms": float(
            torch.cat([gradient.detach().float().flatten() for gradient in gradients])
            .square()
            .mean()
            .sqrt()
        ),
        "side_token_rms": rms(side_tokens),
        "normalized_side_token_rms": rms(normalized_tokens),
        "configured_optimizer_group": optimizer_hyperparameters,
        "synthetic_sgd_delta_rms": sgd_delta_rms,
        "configured_adamw_delta_rms": adam_delta_rms,
        "threshold": 1e-5,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
