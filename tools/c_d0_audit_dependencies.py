#!/usr/bin/env python3
"""Audit D-FINE channel dependency groups without pruning the model."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--scope",
        choices=("full", "backbone", "encoder", "decoder"),
        default="full",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model.eval()
    image = torch.randn(1, 3, args.height, args.width)
    if args.scope == "full":
        subject = model
        example_inputs = image
    elif args.scope == "backbone":
        subject = model.backbone
        example_inputs = image
    elif args.scope == "encoder":
        with torch.no_grad():
            backbone_features = model.backbone(image)
        subject = model.encoder
        example_inputs = (backbone_features,)
    else:
        with torch.no_grad():
            encoder_features = model.encoder(model.backbone(image))
        subject = model.decoder
        example_inputs = (encoder_features,)

    module_names = {module: name for name, module in subject.named_modules()}

    print("=== PRUNABLE MODULE INVENTORY ===")
    for module, name in module_names.items():
        if isinstance(module, nn.Conv2d):
            print(
                f"CONV\t{name}\tin={module.in_channels}\tout={module.out_channels}"
                f"\tk={module.kernel_size}\ts={module.stride}\tg={module.groups}"
            )
        elif isinstance(module, nn.Linear):
            print(f"LINEAR\t{name}\tin={module.in_features}\tout={module.out_features}")

    print("=== DEPENDENCY GRAPH BUILD ===")
    try:
        graph = tp.DependencyGraph().build_dependency(
            subject, example_inputs=example_inputs
        )
    except Exception:
        traceback.print_exc()
        raise

    print("DEPENDENCY_GRAPH_OK")
    groups = list(
        graph.get_all_groups(
            ignored_layers=[],
            root_module_types=[nn.Conv2d, nn.Linear],
        )
    )
    print(f"GROUP_COUNT={len(groups)}")
    for index, group in enumerate(groups):
        print(f"=== GROUP {index:04d} ===")
        print(group.details())


if __name__ == "__main__":
    main()
