#!/usr/bin/env python3
"""One-batch forward/loss/backward validation for S-AUX."""

import argparse
import sys
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--baseline-config", type=Path)
    a = p.parse_args()
    sys.path.insert(0, str(a.repo))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(a.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model, criterion = cfg.model.cuda().train(), cfg.criterion.cuda().train()
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    weights = state["ema"]["module"] if "ema" in state else state["model"]
    own = model.state_dict()
    compatible = {k: v for k, v in weights.items() if k in own and own[k].shape == v.shape}
    info = model.load_state_dict(compatible, strict=False)
    samples, targets = next(iter(cfg.train_dataloader))
    samples = samples.cuda()
    targets = [{k: v.cuda() if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(samples, targets=targets)
    with torch.autocast("cuda", enabled=False):
        losses = criterion(outputs, targets, epoch=0, step=0, global_step=0, epoch_step=1)
        total = sum(losses.values())
    total.backward()
    head_grad = model.backbone.spatial_aux_head.weight.grad
    print({
        "batch": tuple(samples.shape),
        "importance": tuple(outputs["spatial_importance_logits"].shape),
        "loss_spatial_aux": float(losses["loss_spatial_aux"].detach()),
        "loss_total": float(total.detach()),
        "head_grad_finite": bool(torch.isfinite(head_grad).all()),
        "head_grad_mean_abs": float(head_grad.abs().mean()),
        "missing_keys": info.missing_keys,
        "unexpected_keys": info.unexpected_keys,
    })

    if a.baseline_config is not None:
        base_cfg = YAMLConfig(str(a.baseline_config))
        base_cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        base_cfg.yaml_cfg["HGNetv2"]["spatial_aux_stage"] = -1
        baseline = base_cfg.model.cuda().eval()
        baseline.load_state_dict(weights, strict=True)
        model.load_state_dict(compatible, strict=False)
        model.eval()
        probe = samples[:1]
        with torch.inference_mode():
            base_out, aux_out = baseline(probe), model(probe)
        print({
            "baseline_pred_logits_max_abs_diff": float(
                (base_out["pred_logits"] - aux_out["pred_logits"]).abs().max()
            ),
            "baseline_pred_boxes_max_abs_diff": float(
                (base_out["pred_boxes"] - aux_out["pred_boxes"]).abs().max()
            ),
        })


if __name__ == "__main__":
    main()
