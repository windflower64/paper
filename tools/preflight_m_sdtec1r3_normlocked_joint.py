"""One-step CUDA preflight for M-SDTEC1-R3 norm-locked joint tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"]
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={incompatible.missing_keys[:10]} "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )

    norm_modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }
    if not norm_modules:
        raise RuntimeError("no BatchNorm modules found for the norm-lock preflight")
    norm_before = {
        name: {
            "running_mean": module.running_mean.detach().clone(),
            "running_var": module.running_var.detach().clone(),
            "num_batches_tracked": module.num_batches_tracked.detach().clone(),
        }
        for name, module in norm_modules.items()
    }

    device = torch.device("cuda")
    model.to(device)
    criterion = cfg.criterion.to(device)
    optimizer = cfg.optimizer
    loader = cfg.train_dataloader
    model.set_training_epoch(0)
    model.train()

    norms_still_training = [
        name for name, module in norm_modules.items() if module.training
    ]
    if norms_still_training:
        raise RuntimeError(
            f"norm lock did not switch all BatchNorm modules to eval: "
            f"{norms_still_training[:10]}"
        )
    if not model.decoder.training:
        raise RuntimeError("norm lock incorrectly switched the decoder to eval mode")

    samples, targets = next(iter(loader))
    samples = samples.to(device)
    targets = [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(samples, targets=targets)
    with torch.autocast(device_type="cuda", enabled=False):
        losses = criterion(
            outputs,
            targets,
            epoch=0,
            step=0,
            global_step=0,
            epoch_step=len(loader),
        )
        total_loss = sum(losses.values())
    if not torch.isfinite(total_loss):
        raise RuntimeError(f"non-finite total loss: {total_loss}")
    total_loss.backward()

    group_gradients = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = group["params"]
        group_gradients.append(
            {
                "group": index,
                "lr": float(group["lr"]),
                "parameter_tensors": len(parameters),
                "gradient_tensors": sum(p.grad is not None for p in parameters),
            }
        )
    empty_gradient_groups = [
        item["group"]
        for item in group_gradients
        if item["parameter_tensors"] > 0 and item["gradient_tensors"] == 0
    ]
    if empty_gradient_groups:
        raise RuntimeError(f"optimizer groups without gradients: {empty_gradient_groups}")
    optimizer.step()

    changed_norm_buffers = []
    for name, module in norm_modules.items():
        current = {
            "running_mean": module.running_mean.detach().cpu(),
            "running_var": module.running_var.detach().cpu(),
            "num_batches_tracked": module.num_batches_tracked.detach().cpu(),
        }
        for key, value in current.items():
            if not torch.equal(value, norm_before[name][key]):
                changed_norm_buffers.append(f"{name}.{key}")
    if changed_norm_buffers:
        raise RuntimeError(
            f"normalization buffers changed despite lock: {changed_norm_buffers[:10]}"
        )

    result = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "batch_size": int(loader.batch_size),
        "batch_norm_modules": len(norm_modules),
        "changed_norm_buffers": changed_norm_buffers,
        "loss": float(total_loss.detach().cpu()),
        "optimizer_groups": group_gradients,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
