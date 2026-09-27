"""Strict CUDA preflight for protected M2 thermal logit calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


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
    cfg = YAMLConfig(str(args.config.resolve()))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    state = checkpoint["ema"]["module"]
    model.load_state_dict(state, strict=True)

    decoder = model.decoder
    if decoder.sdtec_coupler_mode != "logit_calibration":
        raise RuntimeError("configured model is not M2 logit calibration")
    if decoder.sdtec_couplers is not None:
        raise RuntimeError("M2 unexpectedly retained query residual couplers")
    if decoder.sdtec_logit_calibrator is None:
        raise RuntimeError("M2 logit calibrator was not constructed")
    if decoder.sdtec_tokenizer.num_tokens != 8:
        raise RuntimeError("M2 requires eight unordered thermal evidence tokens")
    if decoder.sdtec_logit_calibrator.protected_tail_queries != 50:
        raise RuntimeError("M2 did not protect all 50 HRQS queries")

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    invalid_trainable = [
        name for name in trainable_names if not name.startswith("decoder.sdtec_")
    ]
    if invalid_trainable:
        raise RuntimeError(f"non-M2 trainable parameters: {invalid_trainable[:10]}")

    frozen_before = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("decoder.sdtec_")
    }
    device = torch.device("cuda")
    model.to(device)
    criterion = cfg.criterion.to(device)
    optimizer = cfg.optimizer
    loader = cfg.train_dataloader
    samples, targets = next(iter(loader))
    samples = samples.to(device)
    targets = move_targets(targets, device)
    if samples.shape[0] != 32 or samples.shape[1] != 6:
        raise RuntimeError(f"unexpected M2 preflight input shape {tuple(samples.shape)}")

    # Constructor identity and explicit zero-thermal fallback must be exact.
    model.eval()
    decoder.set_training_epoch(20)
    model.rgbt_thermal_intervention = "normal"
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        initial_normal = model(samples)
    model.rgbt_thermal_intervention = "zero"
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        initial_zero = model(samples)
    if not torch.equal(initial_normal["pred_logits"], initial_zero["pred_logits"]):
        raise RuntimeError("zero-initialized M2 changed logits between normal and zero IR")
    if not torch.equal(initial_normal["pred_boxes"], initial_zero["pred_boxes"]):
        raise RuntimeError("zero-initialized M2 changed boxes between normal and zero IR")

    records = []
    torch.cuda.reset_peak_memory_stats(device)
    model.rgbt_thermal_intervention = "normal"
    for step in range(2):
        decoder.set_training_epoch(1)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(samples, targets=targets)
        with torch.autocast(device_type="cuda", enabled=False):
            losses = criterion(
                outputs,
                targets,
                epoch=1,
                step=step,
                global_step=step,
                epoch_step=len(loader),
            )
            total_loss = sum(losses.values())
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite M2 loss at step {step}: {total_loss}")
        total_loss.backward()

        delta = outputs["sdtec_logit_delta"].detach()
        protected = delta[:, -50:]
        if torch.count_nonzero(protected):
            raise RuntimeError("M2 changed a protected HRQS classification logit")
        if float(delta.abs().max().cpu()) > decoder.sdtec_max_logit_delta + 1e-6:
            raise RuntimeError("M2 exceeded its hard logit-delta bound")
        if "sdtec_mismatch_logits" not in outputs:
            raise RuntimeError("M2 training omitted the paired mismatch branch")

        with_gradient = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        nonzero_gradient = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad.detach()))
        ]
        optimizer.step()
        records.append(
            {
                "step": step,
                "fusion_progress": float(decoder.sdtec_fusion_progress),
                "loss": float(total_loss.detach().cpu()),
                "loss_sdtec_pair_rank": float(
                    losses["loss_sdtec_pair_rank"].detach().cpu()
                ),
                "loss_sdtec_mismatch": float(
                    losses["loss_sdtec_mismatch"].detach().cpu()
                ),
                "max_abs_logit_delta_before_step": float(delta.abs().max().cpu()),
                "ordinary_mean_abs_logit_delta_before_step": float(
                    delta[:, :-50].abs().mean().cpu()
                ),
                "protected_hrqs_nonzero": int(torch.count_nonzero(protected).cpu()),
                "trainable_with_gradient": len(with_gradient),
                "trainable_with_nonzero_gradient": len(nonzero_gradient),
            }
        )

    # After M2 opens, logits may change but boxes must remain bit-exact between
    # correct and zero thermal.  Zero thermal must return the base C+D logits.
    model.eval()
    decoder.set_training_epoch(20)
    model.rgbt_thermal_intervention = "normal"
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        opened_normal = model(samples)
    model.rgbt_thermal_intervention = "zero"
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        opened_zero = model(samples)
    if not torch.equal(opened_normal["pred_boxes"], opened_zero["pred_boxes"]):
        raise RuntimeError("opened M2 changed box geometry")
    if torch.equal(opened_normal["pred_logits"], opened_zero["pred_logits"]):
        raise RuntimeError("opened M2 did not causally use thermal evidence")

    model.cpu()
    frozen_changes = []
    for key, value in model.state_dict().items():
        if key in frozen_before and not torch.equal(
            value.detach().cpu(), frozen_before[key]
        ):
            frozen_changes.append(key)
    if frozen_changes:
        raise RuntimeError(f"M2 training changed frozen tensors: {frozen_changes[:10]}")
    if records[1]["trainable_with_nonzero_gradient"] <= records[0][
        "trainable_with_nonzero_gradient"
    ]:
        raise RuntimeError(
            "M2 deeper tokenizer/calibrator parameters did not receive gradients "
            "after the zero output projection opened"
        )

    result = {
        "status": "PASS",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": args.seed,
        "batch_size": int(loader.batch_size),
        "input_shape": list(samples.shape),
        "trainable_tensor_count": len(trainable_names),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "m2_num_tokens": decoder.sdtec_tokenizer.num_tokens,
        "max_logit_delta": decoder.sdtec_max_logit_delta,
        "protected_hrqs_queries": 50,
        "initial_normal_equals_zero_logits": True,
        "initial_normal_equals_zero_boxes": True,
        "opened_normal_equals_zero_boxes": True,
        "opened_normal_differs_from_zero_logits": True,
        "steps": records,
        "frozen_tensor_changes": frozen_changes,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
