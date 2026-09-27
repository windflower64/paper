"""Build an SDTEC tuning checkpoint from the frozen GQ1 checkpoint.

RGB weights remain unchanged.  The independent thermal backbone and encoder
receive copies of the corresponding trained RGB tensors only as initialization;
their parameters are independent during training.  New SDTEC parameters are
intentionally absent and keep their constructor initialization.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def duplicate_modal_weights(state_dict):
    result = OrderedDict(state_dict)
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            result["thermal_backbone." + key[len("backbone.") :]] = value.clone()
        elif key.startswith("encoder."):
            result["thermal_encoder." + key[len("encoder.") :]] = value.clone()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError("Source checkpoint has no model state dict")
    checkpoint["model"] = duplicate_modal_weights(checkpoint["model"])
    if isinstance(checkpoint.get("ema"), dict) and isinstance(
        checkpoint["ema"].get("module"), dict
    ):
        checkpoint["ema"]["module"] = duplicate_modal_weights(
            checkpoint["ema"]["module"]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    source_keys = checkpoint["model"]
    print(f"output={args.output}")
    print(f"model_keys={len(source_keys)}")
    print(
        "thermal_backbone_keys="
        f"{sum(key.startswith('thermal_backbone.') for key in source_keys)}"
    )
    print(
        "thermal_encoder_keys="
        f"{sum(key.startswith('thermal_encoder.') for key in source_keys)}"
    )


if __name__ == "__main__":
    main()

