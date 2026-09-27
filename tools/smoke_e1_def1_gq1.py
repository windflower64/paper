#!/usr/bin/env python3
"""Preflight E1 teacher architecture, tuning compatibility and gradients."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tuning", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.repo))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(a.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model, criterion = cfg.model.cuda().train(), cfg.criterion.cuda().train()
    checkpoint = torch.load(a.tuning, map_location="cpu", weights_only=False)
    weights = checkpoint.get("ema", {}).get("module") or checkpoint.get("model") or checkpoint
    own = model.state_dict()
    compatible = {k: v for k, v in weights.items() if k in own and own[k].shape == v.shape}
    load = model.load_state_dict(compatible, strict=False)

    backbone = model.backbone
    assert backbone.srtod_stage == 2 and not backbone.srtod_student_only
    assert backbone.srtod_rh is not None and backbone.srtod_dgfe is not None
    assert backbone.srtod_deficiency_head is None
    assert backbone.pat_sf_stage == 3 and backbone.pat_sf_variant == "global_query"
    assert criterion.srtod_reconstruction_weight == 1.0

    generator = torch.Generator(device="cuda").manual_seed(20260812)
    image = torch.randn(2, 3, 512, 640, generator=generator, device="cuda")
    targets = [
        {
            "boxes": torch.tensor([[0.48, 0.43, 0.05, 0.04]], device="cuda"),
            "labels": torch.zeros(1, dtype=torch.long, device="cuda"),
        }
        for _ in range(2)
    ]
    output = model(image, targets=targets)
    losses = criterion(output, targets, epoch=0, step=0, global_step=0, epoch_step=1)
    total = sum(losses.values())
    total.backward()

    groups = {
        "rh": "srtod_rh",
        "dgfe": "srtod_dgfe",
        "gq1": "pat_sf",
    }
    grad = {}
    for group, token in groups.items():
        values = [
            parameter.grad.detach().float().norm().square()
            for name, parameter in model.named_parameters()
            if token in name and parameter.requires_grad and parameter.grad is not None
        ]
        grad[group] = {
            "tensors": len(values),
            "l2": float(torch.stack(values).sum().sqrt().cpu()) if values else 0.0,
        }
        if not values or grad[group]["l2"] <= 0:
            raise RuntimeError(f"no effective gradient for {group}: {grad[group]}")

    result = {
        "loaded_tensors": len(compatible),
        "missing_keys": list(load.missing_keys),
        "unexpected_keys": list(load.unexpected_keys),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "output_shapes": {
            key: list(value.shape) for key, value in output.items() if torch.is_tensor(value)
        },
        "losses": {key: float(value.detach().cpu()) for key, value in losses.items()},
        "gradient_groups": grad,
        "reconstruction_loss_present": "loss_srtod_reconstruction" in losses,
        "cuda_peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
