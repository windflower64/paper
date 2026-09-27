#!/usr/bin/env python3
"""Build and forward every stage-1 model before launching long training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()

    repo = args.repo.resolve(strict=True)
    sys.path.insert(0, str(repo))
    from src.core import YAMLConfig  # noqa: PLC0415

    checkpoint = torch.load(args.checkpoint, map_location="cpu")["model"]
    device = torch.device("cuda:0")
    print("device", torch.cuda.get_device_name(device), flush=True)

    for relative_config in args.configs:
        path = repo / relative_config
        cfg = YAMLConfig(str(path))
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        model = cfg.model
        state = model.state_dict()
        compatible = {
            key: value
            for key, value in checkpoint.items()
            if key in state and state[key].shape == value.shape
        }
        result = model.load_state_dict(compatible, strict=False)
        params = sum(parameter.numel() for parameter in model.parameters())

        model = model.to(device).eval()
        sample = torch.rand(1, 3, 512, 640, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            output = model(sample)
        torch.cuda.synchronize(device)

        print(
            path.name,
            "params=", params,
            "loaded=", len(compatible),
            "missing=", len(result.missing_keys),
            "peak_mib=", round(torch.cuda.max_memory_allocated(device) / 1024**2, 2),
            "pred_logits=", tuple(output["pred_logits"].shape),
            "pred_boxes=", tuple(output["pred_boxes"].shape),
            flush=True,
        )
        del model, sample, output
        torch.cuda.empty_cache()

    print("STAGE1_MODEL_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()

