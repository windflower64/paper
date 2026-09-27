#!/usr/bin/env python3
"""Engineering preflight for the SET-HBS training-only D-FINE transfer."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    base_config = YAMLConfig(str(ROOT / "experiments/phase_s/visible_60e_base.yml"))
    base_config.yaml_cfg["HGNetv2"]["pretrained"] = False
    hbs_config = YAMLConfig(
        str(ROOT / "experiments/phase_s/s_hbs1_bg_smooth_aux_p16_p32.yml")
    )
    hbs_config.yaml_cfg["HGNetv2"]["pretrained"] = False

    torch.manual_seed(0)
    base_model = base_config.model.to(device).train()
    torch.manual_seed(0)
    hbs_model = hbs_config.model.to(device).train()
    criterion = hbs_config.criterion.to(device).train()
    optimizer = hbs_config.optimizer
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    missing_optimizer_parameters = [
        name
        for name, parameter in hbs_model.hbs.named_parameters()
        if id(parameter) not in optimizer_parameter_ids
    ]
    if missing_optimizer_parameters:
        raise RuntimeError(
            f"HBS parameters are absent from the optimizer: {missing_optimizer_parameters}"
        )

    base_state = base_model.state_dict()
    compatible = {
        key: value
        for key, value in base_state.items()
        if key in hbs_model.state_dict()
        and hbs_model.state_dict()[key].shape == value.shape
    }
    load_result = hbs_model.load_state_dict(compatible, strict=False)
    unexpected_missing = [
        key for key in load_result.missing_keys if not key.startswith("hbs.")
    ]
    if unexpected_missing or load_result.unexpected_keys:
        raise RuntimeError(
            f"unexpected transfer mismatch: missing={unexpected_missing}, "
            f"unexpected={load_result.unexpected_keys}"
        )

    samples = torch.randn(1, 3, 512, 640, device=device)
    targets = [
        {
            "labels": torch.tensor([0], dtype=torch.long, device=device),
            "boxes": torch.tensor(
                [[0.52, 0.47, 0.055, 0.040]], dtype=torch.float32, device=device
            ),
            "orig_size": torch.tensor([512, 640], device=device),
            "size": torch.tensor([512, 640], device=device),
            "image_id": torch.tensor(0, device=device),
        }
    ]

    torch.manual_seed(20260814)
    base_outputs = base_model(samples, targets=targets)
    torch.manual_seed(20260814)
    hbs_outputs = hbs_model(samples, targets=targets)
    main_differences = {
        key: float((hbs_outputs[key] - base_outputs[key]).abs().max())
        for key in ("pred_logits", "pred_boxes")
    }
    if max(main_differences.values()) != 0.0:
        raise RuntimeError(f"HBS changed the main training path: {main_differences}")
    if "hbs_aux_outputs" not in hbs_outputs:
        raise RuntimeError("HBS training path did not return auxiliary detections")

    masks = hbs_model.hbs.last_masks
    residuals = hbs_model.hbs.last_residuals
    mask_cells = [int(mask.sum()) for mask in masks]
    if any(value < 1 for value in mask_cells):
        raise RuntimeError(f"tiny target vanished from an HBS feature mask: {mask_cells}")
    region_residual = []
    for mask, residual in zip(masks, residuals):
        foreground = mask.expand_as(residual).bool()
        background = ~foreground
        region_residual.append(
            {
                "foreground_mean_abs": float(residual[foreground].abs().mean()),
                "foreground_max_abs": float(residual[foreground].abs().max()),
                "background_mean_abs": float(residual[background].abs().mean()),
            }
        )
    if any(item["foreground_max_abs"] != 0.0 for item in region_residual):
        raise RuntimeError(
            f"strict HBS allowed background spill into foreground: {region_residual}"
        )
    if not all(item["background_mean_abs"] > 0.0 for item in region_residual):
        raise RuntimeError("HBS did not modify background features")

    losses = criterion(hbs_outputs, targets, epoch=0, step=0, global_step=0, epoch_step=1)
    total_loss = sum(losses.values())
    if not torch.isfinite(total_loss):
        raise RuntimeError("HBS loss is non-finite")
    total_loss.backward()
    gradients = {
        name: float(parameter.grad.detach().abs().mean())
        for name, parameter in hbs_model.hbs.named_parameters()
        if parameter.grad is not None
    }
    if not gradients or max(gradients.values()) <= 0:
        raise RuntimeError("No non-zero gradient reached HBS")

    hbs_model.eval()
    with torch.inference_mode():
        eval_outputs = hbs_model(samples)
    if "hbs_aux_outputs" in eval_outputs:
        raise RuntimeError("HBS auxiliary path remained active during evaluation")

    base_deploy = copy.deepcopy(base_model).cpu().deploy()
    hbs_deploy = copy.deepcopy(hbs_model).cpu().deploy()
    deploy_parameters = {
        "base": parameter_count(base_deploy),
        "hbs": parameter_count(hbs_deploy),
    }
    if deploy_parameters["base"] != deploy_parameters["hbs"]:
        raise RuntimeError(f"deploy parameter mismatch: {deploy_parameters}")

    report = {
        "status": "pass",
        "device": str(device),
        "experiment": "S_HBS1_BG_SMOOTH_AUX_P16_P32",
        "main_path_max_abs_difference": main_differences,
        "feature_mask_cells": mask_cells,
        "feature_kernel_sizes": hbs_model.hbs.kernel_sizes,
        "region_residual": region_residual,
        "loss": float(total_loss.detach()),
        "loss_terms": len(losses),
        "hbs_gradient_mean_absolute": gradients,
        "training_parameters": {
            "base": parameter_count(base_model),
            "hbs": parameter_count(hbs_model),
            "delta": parameter_count(hbs_model) - parameter_count(base_model),
        },
        "deploy_parameters": deploy_parameters,
        "peak_memory_mib": (
            float(torch.cuda.max_memory_allocated() / 1024**2)
            if device.type == "cuda"
            else None
        ),
        "checkpoint_transfer": {
            "loaded_tensors": len(compatible),
            "hbs_missing_tensors": len(load_result.missing_keys),
            "unexpected_tensors": len(load_result.unexpected_keys),
        },
        "optimizer": {
            "hbs_parameters_included": True,
            "parameter_groups": len(optimizer.param_groups),
        },
    }
    destination = Path(
        "/root/autodl-tmp/rgbt_experiments/reports/20_spatial_importance/"
        "S4_HBS_PREFLIGHT/preflight.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
