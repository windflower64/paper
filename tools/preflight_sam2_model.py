#!/usr/bin/env python3
"""Load the exact SAM2.1 checkpoint on CPU without running expensive inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.sam2_repo))
    from sam2.build_sam import build_sam2

    model = build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml", str(args.checkpoint), device="cpu")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "parameters": parameters,
        "finite_first_parameter": bool(torch.isfinite(next(model.parameters())).all()),
        "cuda_available": torch.cuda.is_available(),
    }, indent=2))


if __name__ == "__main__":
    main()
