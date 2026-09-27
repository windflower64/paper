#!/usr/bin/env python3
"""Dry-run two cross-component D-FINE channel-pruning templates.

All mutations are made in memory on fresh random-weight models.  Nothing is
saved.  The goal is to verify which dimensions have to change together.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch_pruning as tp


def load_model(repo: Path, config: Path):
    sys.path.insert(0, str(repo))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    return cfg.model.eval()


def tensor_summary(value):
    if torch.is_tensor(value):
        return tuple(value.shape)
    if isinstance(value, dict):
        return {key: tensor_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_summary(item) for item in value]
    return type(value).__name__


def run_case(repo: Path, config: Path, stage: str, remove: int, image):
    model = load_model(repo, config)
    before_params = sum(parameter.numel() for parameter in model.parameters())
    graph = tp.DependencyGraph().build_dependency(
        model.backbone, example_inputs=image
    )

    if stage == "s16":
        root = model.backbone.stages[2].blocks[0].aggregation[1].conv
        external_projection = model.encoder.input_proj[0].conv
    elif stage == "s32":
        root = model.backbone.stages[3].blocks[0].aggregation[1].conv
        external_projection = model.encoder.input_proj[1].conv
    else:
        raise ValueError(stage)

    indices = list(range(root.out_channels - remove, root.out_channels))
    group = graph.get_pruning_group(
        root, tp.prune_conv_out_channels, idxs=indices
    )
    if not graph.check_pruning_group(group):
        raise RuntimeError(f"invalid dependency group for {stage}")
    group.prune()

    # The backbone was traced independently, so its returned feature tensor is
    # an output boundary.  Couple that boundary explicitly to the corresponding
    # HybridEncoder projection.
    tp.prune_conv_in_channels(external_projection, indices)

    with torch.no_grad():
        backbone_output = model.backbone(image)
        encoder_output = model.encoder(backbone_output)
        full_output = model.decoder(encoder_output)

    after_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"CASE={stage.upper()} REMOVE={remove}")
    print(
        f"ROOT_OUT={root.out_channels} PROJECTION_IN={external_projection.in_channels}"
    )
    print(f"PARAMS_BEFORE={before_params} PARAMS_AFTER={after_params}")
    print(f"BACKBONE_OUTPUT={tensor_summary(backbone_output)}")
    print(f"ENCODER_OUTPUT={tensor_summary(encoder_output)}")
    print(f"DECODER_OUTPUT={tensor_summary(full_output)}")
    print("FORWARD_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--remove", type=int, default=8)
    args = parser.parse_args()
    image = torch.randn(1, 3, args.height, args.width)
    for stage in ("s16", "s32"):
        run_case(args.repo, args.config, stage, args.remove, image)


if __name__ == "__main__":
    main()
