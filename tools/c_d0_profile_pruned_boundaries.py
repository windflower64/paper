#!/usr/bin/env python3
"""Profile legal S16/S32 output-boundary pruning dry-runs at several ratios."""

from __future__ import annotations

import argparse
import copy
import sys
import warnings
from pathlib import Path

import torch
import torch_pruning as tp
from calflops import calculate_flops


def load_model(repo: Path, config: Path):
    sys.path.insert(0, str(repo))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    return cfg.model.eval()


def prune_boundary(model, image, stage: str, ratio: float):
    graph = tp.DependencyGraph().build_dependency(
        model.backbone, example_inputs=image
    )
    if stage == "s16":
        root = model.backbone.stages[2].blocks[0].aggregation[1].conv
        projection = model.encoder.input_proj[0].conv
    else:
        root = model.backbone.stages[3].blocks[0].aggregation[1].conv
        projection = model.encoder.input_proj[1].conv
    original = root.out_channels
    remove = int(round(original * ratio))
    indices = list(range(original - remove, original))
    group = graph.get_pruning_group(
        root, tp.prune_conv_out_channels, idxs=indices
    )
    if not graph.check_pruning_group(group):
        raise RuntimeError(f"invalid group: {stage} {ratio}")
    group.prune()
    tp.prune_conv_in_channels(projection, indices)
    return original, root.out_channels, projection.in_channels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()
    image = torch.randn(1, 3, args.height, args.width)
    warnings.filterwarnings("ignore", message="Unwrapped parameters detected")

    for stage in ("s16", "s32"):
        for ratio in (0.125, 0.25, 0.375):
            model = load_model(args.repo, args.config)
            original, remaining, projection_in = prune_boundary(
                model, image, stage, ratio
            )
            with torch.no_grad():
                model(image)
            deployed = copy.deepcopy(model).deploy().eval()
            flops, macs, _ = calculate_flops(
                model=deployed,
                input_shape=(1, 3, args.height, args.width),
                output_as_string=True,
                output_precision=6,
                print_detailed=False,
                print_results=False,
            )
            params = sum(parameter.numel() for parameter in deployed.parameters())
            print(
                f"STAGE={stage.upper()} RATIO={ratio:.3f} "
                f"CHANNELS={original}->{remaining} PROJECTION_IN={projection_in} "
                f"PARAMS={params} MACS={macs} FLOPS={flops} FORWARD_OK"
            )


if __name__ == "__main__":
    main()
