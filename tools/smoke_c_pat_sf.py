"""CPU-only structural smoke test for the C-PAT-SF1 transfer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def parameter_count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def gradient_l1(module) -> float:
    return float(
        sum(
            parameter.grad.abs().sum()
            for parameter in module.parameters()
            if parameter.grad is not None
        )
    )


def tensors_are_finite(value) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(tensors_are_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(tensors_are_finite(item) for item in value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/phase_c/c_pat_sf_s32_r4.yml",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    torch.set_num_threads(1)

    from src.core import YAMLConfig
    from src.nn.backbone.hgnetv2 import HGNetv2
    from src.nn.backbone.partialnet_pat_sf import PartialSelfAttentionConv

    baseline = HGNetv2(
        name="B0",
        pretrained=False,
        freeze_at=-1,
        freeze_norm=False,
    )
    transferred = HGNetv2(
        name="B0",
        pretrained=False,
        freeze_at=-1,
        freeze_norm=False,
        pat_sf_stage=3,
        pat_sf_n_div=4,
    )
    pat_modules = [
        module
        for module in transferred.modules()
        if isinstance(module, PartialSelfAttentionConv)
    ]
    assert len(pat_modules) == 3, len(pat_modules)
    assert all(module.conv_channels == 32 for module in pat_modules)
    assert all(module.attention_channels == 96 for module in pat_modules)

    transferred.train()
    small = torch.randn(2, 3, 128, 160, requires_grad=True)
    small_outputs = transferred(small)
    small_loss = sum(output.square().mean() for output in small_outputs)
    small_loss.backward()
    conv_grad = sum(
        gradient_l1(module.partial_conv3) for module in pat_modules
    )
    attention_grad = sum(gradient_l1(module.attn) for module in pat_modules)
    assert conv_grad > 0.0
    assert attention_grad > 0.0

    transferred.eval()
    with torch.no_grad():
        target_outputs = transferred(torch.randn(1, 3, 512, 640))
    expected_shapes = [
        (1, 256, 64, 80),
        (1, 512, 32, 40),
        (1, 1024, 16, 20),
    ]
    actual_shapes = [tuple(output.shape) for output in target_outputs]
    assert actual_shapes == expected_shapes, actual_shapes

    config = YAMLConfig(str(repo / args.config))
    configured_modules = [
        module
        for module in config.model.modules()
        if isinstance(module, PartialSelfAttentionConv)
    ]
    assert len(configured_modules) == 3, len(configured_modules)
    configured_model = config.model.eval()
    with torch.no_grad():
        full_prediction = configured_model(torch.randn(1, 3, 512, 640))
    assert tensors_are_finite(full_prediction)

    baseline_params = parameter_count(baseline)
    transferred_params = parameter_count(transferred)
    print(f"device=cpu")
    print(f"pat_sf_modules={len(pat_modules)}")
    print("per_module_split=32_conv+96_rpe_attention")
    print(f"small_output_shapes={[tuple(x.shape) for x in small_outputs]}")
    print(f"target_output_shapes={actual_shapes}")
    print(f"conv_grad_l1={conv_grad:.8f}")
    print(f"attention_grad_l1={attention_grad:.8f}")
    print(f"baseline_backbone_params={baseline_params}")
    print(f"pat_sf_backbone_params={transferred_params}")
    print(f"parameter_delta={transferred_params - baseline_params}")
    print(f"config_module_count={len(configured_modules)}")
    print(
        "full_model_output_keys="
        f"{sorted(full_prediction) if isinstance(full_prediction, dict) else type(full_prediction).__name__}"
    )


if __name__ == "__main__":
    main()
