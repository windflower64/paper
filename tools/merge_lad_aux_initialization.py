#!/usr/bin/env python3
"""Merge S-LAD1 and S-AUX EMA weights for the S-LAD2-TG compatibility test."""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lad", type=Path, required=True)
    parser.add_argument("--aux", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lad_state = torch.load(args.lad, map_location="cpu", weights_only=False)
    aux_state = torch.load(args.aux, map_location="cpu", weights_only=False)
    source_weight = "backbone.spatial_aux_head.weight"
    source_bias = "backbone.spatial_aux_head.bias"
    target_weight = "backbone.stages.2.downsample.target_projection.weight"
    target_bias = "backbone.stages.2.downsample.target_projection.bias"

    for state_key in ("model",):
        if state_key in lad_state and state_key in aux_state:
            lad_state[state_key][target_weight] = aux_state[state_key][source_weight].clone()
            lad_state[state_key][target_bias] = aux_state[state_key][source_bias].clone()
    if "ema" in lad_state and "ema" in aux_state:
        lad_state["ema"]["module"][target_weight] = aux_state["ema"]["module"][source_weight].clone()
        lad_state["ema"]["module"][target_bias] = aux_state["ema"]["module"][source_bias].clone()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lad_state, args.output)
    print({"output": str(args.output), "copied": [target_weight, target_bias]})


if __name__ == "__main__":
    main()
