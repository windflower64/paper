#!/usr/bin/env python3
"""Generate a detailed, inference-graph FLOP/MAC/parameter report for D-FINE.

This is a diagnostic helper for C-D0.  It does not modify model weights.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    from calflops import calculate_flops
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = copy.deepcopy(cfg.model).deploy().eval()

    flops, macs, params = calculate_flops(
        model=model,
        input_shape=(1, 3, args.height, args.width),
        output_as_string=True,
        output_precision=6,
        print_detailed=True,
        print_results=True,
    )
    print(f"SUMMARY FLOPs={flops} MACs={macs} Params={params}")


if __name__ == "__main__":
    main()
