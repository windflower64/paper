"""Build the dual-pretrained initialization used by M-SDTEC1.

The visible checkpoint supplies every ordinary detector tensor.  Only the
thermal checkpoint's backbone and encoder are remapped into the independent
``thermal_*`` branches.  SDTEC reader parameters remain absent on purpose and
therefore keep their constructor initialization when this file is used with
``train.py --tuning``.

This is a tuning checkpoint, not a resumable optimizer checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import OrderedDict
from pathlib import Path

import torch


def checkpoint_state(checkpoint, source_name):
    if isinstance(checkpoint.get("ema"), dict) and isinstance(
        checkpoint["ema"].get("module"), dict
    ):
        return checkpoint["ema"]["module"], "ema.module"
    if isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"], "model"
    raise RuntimeError(f"{source_name} has neither ema.module nor model state")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible", type=Path, required=True)
    parser.add_argument("--thermal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    visible_checkpoint = torch.load(
        args.visible, map_location="cpu", weights_only=False
    )
    thermal_checkpoint = torch.load(
        args.thermal, map_location="cpu", weights_only=False
    )
    visible_state, visible_field = checkpoint_state(
        visible_checkpoint, "visible checkpoint"
    )
    thermal_state, thermal_field = checkpoint_state(
        thermal_checkpoint, "thermal checkpoint"
    )

    hybrid = OrderedDict((key, value.clone()) for key, value in visible_state.items())
    mapped = []
    for key, value in thermal_state.items():
        if key.startswith("backbone."):
            target_key = "thermal_backbone." + key[len("backbone.") :]
        elif key.startswith("encoder."):
            target_key = "thermal_encoder." + key[len("encoder.") :]
        else:
            continue
        visible_peer = key
        if visible_peer not in visible_state:
            raise RuntimeError(
                f"thermal source key has no visible architecture peer: {key}"
            )
        if visible_state[visible_peer].shape != value.shape:
            raise RuntimeError(
                f"shape mismatch for {key}: visible "
                f"{tuple(visible_state[visible_peer].shape)} vs thermal "
                f"{tuple(value.shape)}"
            )
        hybrid[target_key] = value.clone()
        mapped.append(target_key)

    expected_count = sum(
        key.startswith(("backbone.", "encoder.")) for key in thermal_state
    )
    if len(mapped) != expected_count or not mapped:
        raise RuntimeError(
            f"incomplete thermal mapping: mapped={len(mapped)} "
            f"expected={expected_count}"
        )
    if any(key.startswith("decoder.sdtec_") for key in hybrid):
        raise RuntimeError("hybrid tuning state must not contain trained SDTEC tensors")

    output_checkpoint = {
        "model": hybrid,
        "ema": {"module": OrderedDict(hybrid), "updates": 0},
        "hybrid_init": {
            "visible_path": str(args.visible.resolve()),
            "visible_field": visible_field,
            "visible_sha256": sha256(args.visible),
            "thermal_path": str(args.thermal.resolve()),
            "thermal_field": thermal_field,
            "thermal_sha256": sha256(args.thermal),
            "thermal_mapped_keys": len(mapped),
            "sdtec_parameters": "constructor initialization",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)

    print(f"output={args.output}")
    print(f"visible_field={visible_field} visible_keys={len(visible_state)}")
    print(f"thermal_field={thermal_field} thermal_keys={len(thermal_state)}")
    print(f"thermal_mapped_keys={len(mapped)}")
    print(f"hybrid_keys={len(hybrid)}")
    print(f"output_sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
