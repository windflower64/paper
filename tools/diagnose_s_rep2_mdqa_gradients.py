#!/usr/bin/env python3
"""Compare MDQA and detection gradients at trained REP2 checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--best", type=Path, required=True)
    parser.add_argument("--last", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def weights(checkpoint):
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"], "ema.module"
    return checkpoint.get("model", checkpoint), "model"


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def gradients(loss, parameters, retain_graph):
    return [
        grad.detach().float() if grad is not None else None
        for grad in torch.autograd.grad(
            loss,
            list(parameters),
            retain_graph=retain_graph,
            allow_unused=True,
        )
    ]


def norm(values):
    parts = [value.square().sum() for value in values if value is not None]
    return float(torch.stack(parts).sum().sqrt()) if parts else 0.0


def cosine(left, right):
    pairs = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
    if not pairs:
        return float("nan")
    dot = torch.stack([(a * b).sum() for a, b in pairs]).sum()
    left_norm = torch.stack([a.square().sum() for a, _ in pairs]).sum().sqrt()
    right_norm = torch.stack([b.square().sum() for _, b in pairs]).sum().sqrt()
    return float(dot / (left_norm * right_norm).clamp_min(1e-12))


def inspect(config, checkpoint_path, seed, device):
    seed_everything(seed)
    cfg = YAMLConfig(str(config))
    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state, source = weights(checkpoint)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")

    samples, targets = next(iter(cfg.train_dataloader))
    samples = samples.to(device)
    targets = move_targets(targets, device)
    model.train()
    criterion.train()
    outputs = model(samples, targets=targets)
    losses = criterion(outputs, targets, epoch=int(checkpoint.get("last_epoch", 0)))
    mdqa_loss = losses["loss_mdqa"]
    detection_loss = sum(value for name, value in losses.items() if name != "loss_mdqa")

    backbone_parameters = list(model.backbone.stages[1].parameters())
    decoder_parameters = list(model.decoder.decoder.layers[-1].parameters())
    parameters = backbone_parameters + decoder_parameters
    split = len(backbone_parameters)
    mdqa_grad = gradients(mdqa_loss, parameters, retain_graph=True)
    detection_grad = gradients(detection_loss, parameters, retain_graph=False)

    def group(left, right):
        left_norm = norm(left)
        right_norm = norm(right)
        return {
            "mdqa_gradient_l2": left_norm,
            "detection_gradient_l2": right_norm,
            "mdqa_to_detection_norm_ratio": left_norm / max(right_norm, 1e-12),
            "cosine": cosine(left, right),
        }

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("last_epoch", -1)),
        "weight_source": source,
        "batch": int(samples.shape[0]),
        "weighted_mdqa_loss": float(mdqa_loss.detach()),
        "detection_loss": float(detection_loss.detach()),
        "shared": group(mdqa_grad, detection_grad),
        "s8_backbone": group(mdqa_grad[:split], detection_grad[:split]),
        "final_decoder": group(mdqa_grad[split:], detection_grad[split:]),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("REP2 gradient diagnosis requires CUDA")
    device = torch.device("cuda")
    report = {
        "status": "PASS",
        "config": str(args.config),
        "best": inspect(args.config, args.best, args.seed, device),
        "last": inspect(args.config, args.last, args.seed, device),
    }
    for section in (report["best"], report["last"]):
        for group_name in ("shared", "s8_backbone", "final_decoder"):
            if not math.isfinite(section[group_name]["cosine"]):
                raise RuntimeError(f"non-finite gradient cosine in {group_name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
