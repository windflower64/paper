"""CPU/static preflight audit for C-PAT-SF1; this script never trains."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/phase_c/c_pat_sf_s32_r4.yml",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))

    from src.core import YAMLConfig
    from src.nn.backbone.partialnet_pat_sf import PartialSelfAttentionConv

    config = YAMLConfig(str(repo / args.config))
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = config.model
    optimizer = config.optimizer

    trainable = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assigned = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assignment_counts = Counter(assigned)
    missing = [name for identity, name in trainable.items() if identity not in assignment_counts]
    duplicated = [
        trainable[identity]
        for identity, count in assignment_counts.items()
        if identity in trainable and count != 1
    ]
    foreign = [identity for identity in assignment_counts if identity not in trainable]
    if missing or duplicated or foreign:
        raise RuntimeError(
            f"optimizer mismatch: missing={missing}, duplicated={duplicated}, "
            f"foreign_count={len(foreign)}"
        )

    parameter_group = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter in group["params"]:
            parameter_group[id(parameter)] = (
                group_index,
                group["lr"],
                group.get("weight_decay", optimizer.defaults.get("weight_decay")),
            )
    pat_parameters = {
        name: parameter_group[id(parameter)]
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".pat_sf." in name
    }
    if not pat_parameters:
        raise RuntimeError("No trainable PAT_sf parameters were found")

    modules = [
        module
        for module in model.modules()
        if isinstance(module, PartialSelfAttentionConv)
    ]
    if len(modules) != 3:
        raise RuntimeError(f"Expected 3 PAT_sf modules, found {len(modules)}")

    # Exact operation count for the operator replacement at 512x640 input.
    # HGNetv2 Stage4 is S32, so HxW=16x20.  The common static profiler
    # counts Conv/Linear but not the two attention matrix multiplications.
    height, width = 16, 20
    tokens = height * width
    channels = 128
    conv_channels = channels // 4
    attention_channels = channels - conv_channels
    baseline_dw5_macs = tokens * channels * 5 * 5
    partial_conv_macs = tokens * conv_channels * conv_channels * 3 * 3
    qkv_macs = tokens * attention_channels * attention_channels * 3
    projection_macs = tokens * attention_channels * attention_channels
    attention_matmul_macs = 2 * tokens * tokens * attention_channels
    pat_sf_macs = (
        partial_conv_macs
        + qkv_macs
        + projection_macs
        + attention_matmul_macs
    )
    delta_per_module = pat_sf_macs - baseline_dw5_macs
    total_delta = len(modules) * delta_per_module
    attention_score_elements_per_sample = 4 * tokens * tokens

    print(f"optimizer_trainable_tensors={len(trainable)}")
    print(f"optimizer_assigned_once=True")
    print(f"pat_sf_trainable_tensors={len(pat_parameters)}")
    print(f"pat_sf_optimizer_groups={sorted(set(pat_parameters.values()))}")
    print(f"modules={len(modules)} stage_resolution={height}x{width}")
    print(f"baseline_dw5_macs_per_module={baseline_dw5_macs}")
    print(f"pat_sf_macs_per_module={pat_sf_macs}")
    print(f"attention_matmul_macs_per_module={attention_matmul_macs}")
    print(f"replacement_macs_delta_total={total_delta}")
    print(
        "attention_score_elements_per_module_per_sample="
        f"{attention_score_elements_per_sample}"
    )


if __name__ == "__main__":
    main()
