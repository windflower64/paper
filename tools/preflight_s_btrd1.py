#!/usr/bin/env python3
"""CPU-only engineering preflight for S_BTRD1_ACT_TRANS_S8_S16."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch
import torch.nn as nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.nn.backbone.hgnetv2 import HGNetv2
from src.core import YAMLConfig


def trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def convolution_macs(module: nn.Module, sample: torch.Tensor) -> int:
    total = 0
    hooks = []

    def count(layer: nn.Conv2d, _inputs, output):
        nonlocal total
        batch, out_channels, out_height, out_width = output.shape
        kernel_height, kernel_width = layer.kernel_size
        operations = (
            batch
            * out_channels
            * out_height
            * out_width
            * (layer.in_channels // layer.groups)
            * kernel_height
            * kernel_width
        )
        total += int(operations)

    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            hooks.append(child.register_forward_hook(count))
    with torch.no_grad():
        module(sample)
    for hook in hooks:
        hook.remove()
    return total


def main() -> None:
    torch.manual_seed(0)
    common = dict(
        name="B0",
        pretrained=False,
        freeze_at=-1,
        freeze_norm=False,
    )
    standard = HGNetv2(**common).train()
    btrd = HGNetv2(**common, btrd_stage=2).train()

    sample = torch.randn(2, 3, 128, 160)
    standard_outputs = standard(sample.clone())
    btrd_outputs = btrd(sample.clone())
    standard_shapes = [list(output.shape) for output in standard_outputs]
    btrd_shapes = [list(output.shape) for output in btrd_outputs]
    if standard_shapes != btrd_shapes:
        raise RuntimeError(
            f"BTRD output shapes differ: standard={standard_shapes}, btrd={btrd_shapes}"
        )
    if not all(torch.isfinite(output).all() for output in btrd_outputs):
        raise RuntimeError("BTRD produced a non-finite feature")

    loss = sum(output.float().square().mean() for output in btrd_outputs)
    loss.backward()
    btrd_module = btrd.stages[2].downsample
    gradients = {
        name: float(parameter.grad.detach().abs().mean())
        for name, parameter in btrd_module.named_parameters()
        if parameter.grad is not None
    }
    if not gradients or max(gradients.values()) <= 0.0:
        raise RuntimeError("No non-zero gradient reached the BTRD parameters")

    stage_sample = torch.randn(1, 256, 64, 80)
    standard_downsample = standard.stages[2].downsample.eval()
    btrd_downsample = btrd_module.eval()
    standard_macs = convolution_macs(standard_downsample, stage_sample.clone())
    btrd_macs = convolution_macs(btrd_downsample, stage_sample.clone())
    transition = btrd_downsample.last_transition_map
    if transition is None:
        raise RuntimeError("BTRD transition map was not retained for audit")

    experiment_config = YAMLConfig(
        str(REPOSITORY_ROOT / "experiments/phase_s/s_btrd1_act_trans_s8_s16.yml")
    )
    configured_model = experiment_config.model
    configured_downsample = configured_model.backbone.stages[2].downsample
    if configured_downsample.__class__.__name__ != "BoundaryTransitionRegionDownsample":
        raise RuntimeError(
            "Experiment config did not construct BTRD at HGNetv2 stage index 2"
        )
    configured_parameters = sum(parameter.numel() for parameter in configured_model.parameters())
    parameter_delta = trainable_parameters(btrd) - trainable_parameters(standard)

    report = {
        "status": "pass",
        "experiment": "S_BTRD1_ACT_TRANS_S8_S16",
        "scope": "HGNetv2 B0 Stage3 entrance, S8 -> S16 only",
        "output_shapes": btrd_shapes,
        "model_trainable_parameters": {
            "standard": trainable_parameters(standard),
            "btrd": trainable_parameters(btrd),
            "delta": trainable_parameters(btrd) - trainable_parameters(standard),
        },
        "configured_detector_parameters": {
            "btrd": configured_parameters,
            "baseline_inferred": configured_parameters - parameter_delta,
            "delta": parameter_delta,
            "relative_increase_percent": 100.0
            * parameter_delta
            / (configured_parameters - parameter_delta),
        },
        "configuration": {
            "path": "experiments/phase_s/s_btrd1_act_trans_s8_s16.yml",
            "output_dir": experiment_config.output_dir,
            "pretrained_loading": "pass",
            "constructed_module": configured_downsample.__class__.__name__,
        },
        "stage_downsample_trainable_parameters": {
            "standard": trainable_parameters(standard_downsample),
            "btrd": trainable_parameters(btrd_downsample),
            "delta": trainable_parameters(btrd_downsample)
            - trainable_parameters(standard_downsample),
        },
        "stage_downsample_conv_macs_at_s8_64x80": {
            "standard": standard_macs,
            "btrd": btrd_macs,
            "ratio": btrd_macs / standard_macs,
        },
        "transition_map": {
            "shape": list(transition.shape),
            "minimum": float(transition.min()),
            "mean": float(transition.mean()),
            "maximum": float(transition.max()),
            "standard_deviation": float(transition.std(unbiased=False)),
        },
        "gradient_mean_absolute": gradients,
    }
    destination = Path(
        "/root/autodl-tmp/rgbt_experiments/reports/20_spatial_importance/"
        "S3_BTRD_PREFLIGHT/preflight.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
