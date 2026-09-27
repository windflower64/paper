"""Two-step CUDA preflight for the strict M-SDTEC1-R2 reader protocol."""

from __future__ import annotations

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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    state = checkpoint["ema"]["module"]
    incompatible = model.load_state_dict(state, strict=False)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("decoder.sdtec_")
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"hybrid load mismatch: missing={invalid_missing[:10]} "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )

    non_sdtec_before = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("decoder.sdtec_")
    }
    device = torch.device("cuda")
    model.to(device)
    criterion = cfg.criterion.to(device)
    optimizer = cfg.optimizer
    data_loader = cfg.train_dataloader
    iterator = iter(data_loader)

    records = []
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in (0, 1):
        model.set_training_epoch(epoch)
        model.train()
        samples, targets = next(iterator)
        samples = samples.to(device)
        targets = [
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()
            }
            for target in targets
        ]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(samples, targets=targets)
        with torch.autocast(device_type="cuda", enabled=False):
            losses = criterion(
                outputs,
                targets,
                epoch=epoch,
                step=0,
                global_step=epoch,
                epoch_step=len(data_loader),
            )
            total_loss = sum(losses.values())
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite loss at epoch {epoch}: {total_loss}")
        total_loss.backward()

        residual_gradients = []
        if hasattr(model.decoder.sdtec_couplers, "layer_residual_scales"):
            shared_coupler = True
            scale_gradient = (
                model.decoder.sdtec_couplers.layer_residual_scales.grad
            )
            residual_gradients = (
                [None] * model.decoder.num_layers
                if scale_gradient is None
                else [float(value) for value in scale_gradient.detach().cpu()]
            )
        else:
            shared_coupler = False
            couplers = (
                list(model.decoder.sdtec_couplers)
                if model.decoder.sdtec_couplers is not None
                else [model.decoder.sdtec_spatial_coupler]
            )
            for coupler in couplers:
                gradient = coupler.residual_scale.grad
                residual_gradients.append(
                    None if gradient is None else float(gradient.detach().cpu())
                )
        trainable_with_gradient = sum(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        trainable_without_gradient = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        optimizer.step()
        if shared_coupler:
            effective_scales = [
                float(value)
                for value in model.decoder.sdtec_couplers.effective_scales()
                .detach()
                .cpu()
            ]
        else:
            effective_scales = [
                float(
                    (
                        coupler.max_residual_scale
                        * coupler.fusion_progress
                        * coupler.residual_scale.detach().tanh()
                    ).cpu()
                )
                for coupler in couplers
            ]
        records.append(
            {
                "epoch": epoch,
                "fusion_progress": float(model.decoder.sdtec_fusion_progress),
                "loss": float(total_loss.detach().cpu()),
                "residual_scale_gradients": residual_gradients,
                "trainable_tensors_with_gradient": trainable_with_gradient,
                "trainable_tensors_without_gradient": trainable_without_gradient,
                "effective_scales_after_step": effective_scales,
                "dropout_samples": int(
                    outputs["sdtec_dropout_mask"].sum().detach().cpu()
                ),
            }
        )

    # Frozen parameters and running-stat buffers must remain bit-exact.
    model.cpu()
    frozen_changes = []
    for key, value in model.state_dict().items():
        if key in non_sdtec_before and not torch.equal(
            value.detach().cpu(), non_sdtec_before[key]
        ):
            frozen_changes.append(key)

    if frozen_changes:
        raise RuntimeError(f"strict freeze changed tensors: {frozen_changes[:10]}")
    if records[0]["fusion_progress"] != 0.0:
        raise RuntimeError("epoch 0 did not preserve the exact RGB path")
    if records[1]["fusion_progress"] != 0.2:
        raise RuntimeError("epoch 1 did not open the expected 0.2 ramp")
    if not all(value is not None for value in records[1]["residual_scale_gradients"]):
        raise RuntimeError("epoch 1 did not provide detection gradients to all scales")

    result = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": args.seed,
        "batch_size": int(data_loader.batch_size),
        "steps": records,
        "frozen_tensor_changes": frozen_changes,
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
