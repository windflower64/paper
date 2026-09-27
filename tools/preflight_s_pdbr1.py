#!/usr/bin/env python3
"""CPU-only shape, gradient, supervision and cost checks for S-PDBR1."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from src.nn.backbone.hgnetv2 import HGNetv2
from src.zoo.dfine.dfine_criterion import DFINECriterion


def parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def conv_macs(module: nn.Module, args) -> int:
    total = 0
    hooks = []

    def count(layer, _inputs, output):
        nonlocal total
        _, cout, hout, wout = output.shape
        kh, kw = layer.kernel_size
        total += cout * hout * wout * (layer.in_channels // layer.groups) * kh * kw

    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            hooks.append(child.register_forward_hook(count))
    with torch.no_grad():
        module(*args)
    for hook in hooks:
        hook.remove()
    return int(total)


def main():
    torch.manual_seed(20260814)
    common = dict(name="B0", pretrained=False, freeze_at=-1, freeze_norm=False)
    baseline = HGNetv2(**common).train()
    pdbr_net = HGNetv2(**common, pdbr_stage=2).train()
    sample = torch.randn(2, 3, 128, 160)
    base_outputs = baseline(sample.clone())
    pdbr_outputs = pdbr_net(sample.clone())
    if [x.shape for x in base_outputs] != [x.shape for x in pdbr_outputs]:
        raise RuntimeError("PDBR changed detector feature shapes")
    if not all(torch.isfinite(x).all() for x in pdbr_outputs):
        raise RuntimeError("PDBR produced non-finite features")

    module = pdbr_net.stages[2].pdbr
    loss = sum(x.float().square().mean() for x in pdbr_outputs)
    loss.backward()
    gradients = {
        name: float(param.grad.abs().mean())
        for name, param in module.named_parameters()
        if param.grad is not None
    }
    if not gradients or max(gradients.values()) <= 0:
        raise RuntimeError("No detector gradient reached PDBR")

    logits_x, logits_y = module.last_boundary_logits
    criterion = DFINECriterion(
        matcher=None,
        weight_dict={},
        losses=[],
        pdbr_boundary_aux_weight=0.1,
        pdbr_boundary_sigma=0.75,
    )
    targets = [
        {"boxes": torch.tensor([[0.50, 0.50, 0.20, 0.10]])},
        {"boxes": torch.tensor([[0.25, 0.30, 0.08, 0.06]])},
    ]
    target_x, target_y = criterion._pdbr_boundary_targets(logits_x, logits_y, targets)
    if target_x.max() <= 0 or target_y.max() <= 0:
        raise RuntimeError("Boundary supervision maps are empty")
    boundary_loss = criterion._pdbr_boundary_loss(logits_x, target_x)
    boundary_loss = boundary_loss + criterion._pdbr_boundary_loss(logits_y, target_y)
    if not torch.isfinite(boundary_loss):
        raise RuntimeError("Boundary auxiliary loss is non-finite")

    stage_input = torch.randn(1, 256, 64, 80)
    stage_standard = baseline.stages[2].downsample.eval()
    stage_pdbr = pdbr_net.stages[2].pdbr.eval()
    with torch.no_grad():
        standard = stage_standard(stage_input)
    standard_macs = conv_macs(stage_standard, (stage_input,))
    pdbr_macs = conv_macs(stage_pdbr, (stage_input, standard))

    cfg = YAMLConfig(str(ROOT / "experiments/phase_s/s_pdbr1_posdir_boundary_s8_s16.yml"))
    configured = cfg.model.backbone.stages[2].pdbr
    if configured is None:
        raise RuntimeError("PDBR config did not construct the stage-2 branch")

    report = {
        "status": "pass",
        "experiment": "S_PDBR1_POSDIR_BOUNDARY_S8_S16",
        "scope": "HGNetv2 B0 S8 -> S16 only",
        "output_shapes": [list(x.shape) for x in pdbr_outputs],
        "parameter_delta": parameters(pdbr_net) - parameters(baseline),
        "pdbr_parameters": parameters(module),
        "stage_standard_downsample_macs": standard_macs,
        "pdbr_residual_macs": pdbr_macs,
        "pdbr_to_standard_downsample_macs_ratio": pdbr_macs / standard_macs,
        "initial_scale": float(module.last_scale),
        "initial_residual_rms_ratio": float(module.last_residual_rms_ratio),
        "boundary_shapes": [list(logits_x.shape), list(logits_y.shape)],
        "boundary_target_mass": [float(target_x.sum()), float(target_y.sum())],
        "gradient_mean_abs": gradients,
        "config_output_dir": cfg.output_dir,
    }
    destination = Path(
        "/root/autodl-tmp/rgbt_experiments/reports/20_spatial_importance/"
        "S5_PDBR1_PREFLIGHT/preflight.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
