#!/usr/bin/env python3
"""Small CPU detector forward/loss/backward integration test for S-PDBR1."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def main():
    torch.manual_seed(20260814)
    cfg = YAMLConfig(str(ROOT / "experiments/phase_s/s_pdbr1_posdir_boundary_s8_s16.yml"))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model.train()
    criterion = cfg.criterion.train()
    # D-FINE selects 300 encoder tokens, so the smoke spatial size must expose
    # at least that many multi-scale cells.
    samples = torch.randn(1, 3, 256, 320)
    targets = [{
        "labels": torch.tensor([0], dtype=torch.long),
        "boxes": torch.tensor([[0.50, 0.50, 0.16, 0.10]], dtype=torch.float32),
        "orig_size": torch.tensor([256, 320]),
        "size": torch.tensor([256, 320]),
        "image_id": torch.tensor(0),
    }]
    outputs = model(samples, targets=targets)
    losses = criterion(outputs, targets, epoch=0, step=0, global_step=0, epoch_step=1)
    total = sum(losses.values())
    total.backward()
    module = model.backbone.stages[2].pdbr
    grad = max(
        float(p.grad.abs().mean())
        for p in module.parameters()
        if p.grad is not None
    )
    result = {
        "status": "pass",
        "pred_logits": list(outputs["pred_logits"].shape),
        "pred_boxes": list(outputs["pred_boxes"].shape),
        "boundary_logits": [list(x.shape) for x in outputs["pdbr_boundary_logits"]],
        "loss_pdbr_boundary_aux": float(losses["loss_pdbr_boundary_aux"].detach()),
        "loss_total": float(total.detach()),
        "max_pdbr_gradient_mean_abs": grad,
    }
    if not torch.isfinite(total) or grad <= 0:
        raise RuntimeError(result)
    print(result)


if __name__ == "__main__":
    main()
