#!/usr/bin/env python3
"""Audit whether S-AUX/LAD gradients actually reach the intended branches."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch


def flatten_gradients(gradients):
    values = [g.detach().float().flatten() for g in gradients if g is not None]
    return torch.cat(values) if values else torch.zeros(1, device="cuda")


def grad_stats(loss, parameters, retain_graph=False):
    parameters = list(parameters)
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    flat = flatten_gradients(gradients)
    parameter_flat = torch.cat([p.detach().float().flatten() for p in parameters])
    return {
        "vector": flat,
        "l2": float(flat.norm()),
        "rms": float(flat.square().mean().sqrt()),
        "grad_to_parameter_l2": float(flat.norm() / parameter_flat.norm().clamp_min(1e-12)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batches", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = args.batch_size
    cfg.yaml_cfg["train_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["base_size_repeat"] = None
    model, criterion = cfg.model.cuda().train(), cfg.criterion.cuda().train()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = state["ema"]["module"] if "ema" in state else state["model"]
    model.load_state_dict(weights, strict=True)

    lad_modules = [m for m in model.modules() if hasattr(m, "phase_weight_mode")]
    has_aux = getattr(model.backbone, "spatial_aux_head", None) is not None
    rows = []
    loader = iter(cfg.train_dataloader)
    for batch_index in range(args.batches):
        samples, targets = next(loader)
        samples = samples[: args.batch_size].cuda()
        targets = [
            {k: v.cuda() if torch.is_tensor(v) else v for k, v in t.items()}
            for t in targets[: args.batch_size]
        ]
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(samples, targets=targets)
        with torch.autocast("cuda", enabled=False):
            losses = criterion(
                outputs, targets, epoch=0, step=0, global_step=0, epoch_step=args.batches
            )
        row = {
            "batch": batch_index,
            "loss_total": float(sum(losses.values()).detach()),
            "losses": {k: float(v.detach()) for k, v in losses.items()},
        }

        if has_aux and "loss_spatial_aux" in losses:
            shared_parameters = [
                p for p in model.backbone.stages[model.backbone.spatial_aux_stage].parameters()
                if p.requires_grad
            ]
            detection_loss = sum(v for k, v in losses.items() if k != "loss_spatial_aux")
            aux_loss = losses["loss_spatial_aux"]
            det = grad_stats(detection_loss, shared_parameters, retain_graph=True)
            aux = grad_stats(aux_loss, shared_parameters, retain_graph=False)
            denominator = det["l2"] * aux["l2"]
            cosine = float(torch.dot(det["vector"], aux["vector"]) / denominator) if denominator else None
            row["s_aux"] = {
                "detection_gradient_l2": det["l2"],
                "weighted_aux_gradient_l2": aux["l2"],
                "weighted_aux_to_detection_ratio": aux["l2"] / max(det["l2"], 1e-12),
                "gradient_cosine": cosine,
                "detection_grad_to_parameter_l2": det["grad_to_parameter_l2"],
                "aux_grad_to_parameter_l2": aux["grad_to_parameter_l2"],
            }
        elif lad_modules:
            lad = lad_modules[0]
            total = sum(losses.values())
            weight = grad_stats(total, lad.weight_projection.parameters(), retain_graph=True)
            local = grad_stats(total, lad.local_projection.parameters(), retain_graph=False)
            row["lad"] = {
                "weight_branch_gradient_l2": weight["l2"],
                "local_branch_gradient_l2": local["l2"],
                "weight_to_local_gradient_ratio": weight["l2"] / max(local["l2"], 1e-12),
                "weight_grad_to_parameter_l2": weight["grad_to_parameter_l2"],
                "local_grad_to_parameter_l2": local["grad_to_parameter_l2"],
                "normalized_phase_entropy": float(lad.last_phase_entropy),
                "candidate_phase_variance": float(lad.last_candidate_variance),
                "mean_abs_weight_deviation_from_uniform": float(
                    lad.last_weight_deviation_from_uniform
                ),
            }
        rows.append(row)

    summary = {"config": str(args.config), "checkpoint": str(args.checkpoint), "rows": rows}
    for section in ("s_aux", "lad"):
        present = [row[section] for row in rows if section in row]
        if present:
            summary[f"{section}_mean"] = {
                key: (
                    sum(item[key] for item in present if item[key] is not None)
                    / sum(item[key] is not None for item in present)
                    if any(item[key] is not None for item in present)
                    else None
                )
                for key in present[0]
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
