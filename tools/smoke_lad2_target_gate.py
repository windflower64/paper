#!/usr/bin/env python3
"""One-batch strict-load/forward/loss/backward smoke test for S-LAD2-TG."""

import argparse
import json
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = 2
    cfg.yaml_cfg["train_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["base_size_repeat"] = None
    model, criterion = cfg.model.cuda().train(), cfg.criterion.cuda().train()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["ema"]["module"], strict=True)
    samples, targets = next(iter(cfg.train_dataloader))
    samples = samples[:2].cuda()
    targets = [
        {k: v.cuda() if torch.is_tensor(v) else v for k, v in t.items()}
        for t in targets[:2]
    ]
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(samples, targets=targets)
    with torch.autocast("cuda", enabled=False):
        losses = criterion(outputs, targets, epoch=0, step=0, global_step=0, epoch_step=1)
        total = sum(losses.values())
    total.backward()
    lad = model.backbone.stages[model.backbone.lad_stage].downsample
    grad = lad.target_projection.weight.grad
    result = {
        "batch": list(samples.shape),
        "importance_shape": list(outputs["spatial_importance_logits"].shape),
        "loss_total": float(total.detach()),
        "loss_spatial_aux": float(losses["loss_spatial_aux"].detach()),
        "target_gate_mean": float(lad.last_target_gate.mean()),
        "target_gate_min": float(lad.last_target_gate.min()),
        "target_gate_max": float(lad.last_target_gate.max()),
        "target_projection_grad_l2": float(grad.float().norm()),
        "target_projection_grad_finite": bool(torch.isfinite(grad).all()),
        "phase_entropy": float(lad.last_phase_entropy),
        "candidate_phase_variance": float(lad.last_candidate_variance),
        "outputs_finite": bool(
            torch.isfinite(outputs["pred_logits"]).all()
            and torch.isfinite(outputs["pred_boxes"]).all()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
