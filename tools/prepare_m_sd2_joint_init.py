"""Build the task-start initialization for joint C + D + M-SD2 training."""

from __future__ import annotations

import argparse
import hashlib
import sys
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
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--visible", type=Path, required=True)
    parser.add_argument("--thermal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    visible_checkpoint = torch.load(
        args.visible, map_location="cpu", weights_only=False
    )
    thermal_checkpoint = torch.load(
        args.thermal, map_location="cpu", weights_only=False
    )
    visible_state, visible_field = checkpoint_state(
        visible_checkpoint, "visible COCO checkpoint"
    )
    thermal_state, thermal_field = checkpoint_state(
        thermal_checkpoint, "thermal task checkpoint"
    )

    tuning_state = OrderedDict(
        (key, value.clone()) for key, value in visible_state.items()
    )
    mapped_thermal = []
    for key, value in thermal_state.items():
        if key.startswith("backbone."):
            target_key = "thermal_backbone." + key[len("backbone.") :]
        elif key.startswith("encoder."):
            target_key = "thermal_encoder." + key[len("encoder.") :]
        else:
            continue
        tuning_state[target_key] = value.clone()
        mapped_thermal.append(target_key)
    if not mapped_thermal:
        raise RuntimeError("thermal checkpoint supplied no backbone/encoder tensors")

    cfg = YAMLConfig(str(args.config.resolve()))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    expected = cfg.model.state_dict()
    compatible_state = OrderedDict(
        (key, value)
        for key, value in tuning_state.items()
        if key in expected and value.shape == expected[key].shape
    )
    skipped_visible = sorted(
        key
        for key in visible_state
        if key not in compatible_state
    )
    loaded_thermal = sum(key in compatible_state for key in mapped_thermal)
    if loaded_thermal != len(mapped_thermal):
        missing_thermal = sorted(
            key for key in mapped_thermal if key not in compatible_state
        )
        raise RuntimeError(
            f"incomplete thermal mapping: {loaded_thermal}/{len(mapped_thermal)} "
            f"missing={missing_thermal[:10]}"
        )

    output_checkpoint = {
        "model": compatible_state,
        "ema": {"module": OrderedDict(compatible_state), "updates": 0},
        "msd2_joint_init": {
            "visible_path": str(args.visible.resolve()),
            "visible_field": visible_field,
            "visible_sha256": sha256(args.visible),
            "thermal_path": str(args.thermal.resolve()),
            "thermal_field": thermal_field,
            "thermal_sha256": sha256(args.thermal),
            "thermal_mapped_keys": len(mapped_thermal),
            "visible_loaded_keys": sum(
                key in compatible_state for key in visible_state
            ),
            "visible_skipped_keys": skipped_visible,
            "sd2_parameters": "constructor initialization",
            "sd2_residual_scale": 0.0,
            "training_scope": "joint C+D+thermal+M-SD2 from task epoch 0",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)

    print(f"output={args.output}")
    print(f"visible_field={visible_field} visible_keys={len(visible_state)}")
    print(f"thermal_field={thermal_field} thermal_keys={len(thermal_state)}")
    print(f"thermal_mapped_keys={len(mapped_thermal)}")
    print(f"tuning_keys={len(compatible_state)}")
    print(f"visible_skipped_keys={len(skipped_visible)}")
    print(f"output_sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
