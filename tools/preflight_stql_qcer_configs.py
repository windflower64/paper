"""Static/configuration preflight for STQL/QCER B0--B5."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


ARMS = {
    "B0": "b0_rgb.yml",
    "B1": "b1_stql_sam.yml",
    "B2": "b2_qcer.yml",
    "B3": "b3_box_qcer.yml",
    "B4": "b4_sam_qcer.yml",
    "B5": "b5_uniform_qcer.yml",
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    common = checkpoint["ema"]["module"]
    reports = {}
    base_snapshots = {}
    expected = {
        "B0": (False, False, "sam", False),
        "B1": (True, False, "sam", False),
        "B2": (False, True, "sam", False),
        "B3": (True, True, "box", False),
        "B4": (True, True, "sam", False),
        "B5": (False, True, "sam", True),
    }

    for arm, filename in ARMS.items():
        seed_all(args.seed)
        config_path = args.repo / "experiments/phase_stql_qcer" / filename
        cfg = YAMLConfig(str(config_path.resolve()))
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        model = cfg.model
        criterion = cfg.criterion
        state = model.state_dict()
        compatible = {
            key: value for key, value in common.items()
            if key in state and state[key].shape == value.shape
        }
        incompatible = model.load_state_dict(compatible, strict=False)
        missing = sorted(incompatible.missing_keys)
        unexpected = sorted(incompatible.unexpected_keys)
        allowed_missing = sorted(
            key for key in state
            if key.startswith(("stql_", "qcer."))
        )
        if missing != allowed_missing or unexpected:
            raise RuntimeError(
                f"{arm} checkpoint whitelist failure: missing={missing}, "
                f"allowed={allowed_missing}, unexpected={unexpected}"
            )

        optimizer = cfg.optimizer
        parameter_names = {id(value): name for name, value in model.named_parameters()}
        groups = []
        for group in optimizer.param_groups:
            names = [parameter_names[id(value)] for value in group["params"]]
            groups.append({
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameter_tensors": len(names),
                "parameters": int(sum(value.numel() for value in group["params"])),
                "examples": names[:8],
            })
        thermal_trainable = [
            name for name, value in model.named_parameters()
            if name.startswith("thermal_") and value.requires_grad
        ]
        if thermal_trainable:
            raise RuntimeError(f"{arm} has trainable thermal source: {thermal_trainable[:10]}")
        if not bool(model.rgbt_lock_norm_stats):
            raise RuntimeError(f"{arm} does not lock normalization statistics")

        actual = (
            bool(model.stql_enabled),
            bool(model.qcer_enabled),
            str(criterion.stql_supervision),
            bool(model.qcer.uniform_attention) if model.qcer is not None else False,
        )
        if actual != expected[arm]:
            raise RuntimeError(f"{arm} switches are {actual}, expected {expected[arm]}")

        # Store the complete shared RGB/C tensors after loading.  Every arm must
        # start from exactly the same values regardless of optional modules.
        base_snapshots[arm] = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if key.startswith(("backbone.", "encoder.", "decoder."))
        }
        reports[arm] = {
            "config": str(config_path.resolve()),
            "rgbt": bool(model.rgbt_enabled),
            "stql": bool(model.stql_enabled),
            "qcer": bool(model.qcer_enabled),
            "stql_supervision": str(criterion.stql_supervision),
            "uniform_attention": actual[3],
            "model_parameters": int(sum(value.numel() for value in model.parameters())),
            "trainable_parameters": int(sum(
                value.numel() for value in model.parameters() if value.requires_grad
            )),
            "new_parameters": int(sum(
                value.numel() for name, value in model.named_parameters()
                if name.startswith(("stql_", "qcer."))
            )),
            "checkpoint_loaded_keys": len(compatible),
            "checkpoint_missing_whitelist": missing,
            "optimizer_groups": groups,
            "physical_batch": int(cfg.yaml_cfg["train_dataloader"]["total_batch_size"]),
            "gradient_accumulation_steps": int(
                cfg.yaml_cfg["gradient_accumulation_steps"]
            ),
            "effective_batch": int(
                cfg.yaml_cfg["train_dataloader"]["total_batch_size"]
                * cfg.yaml_cfg["gradient_accumulation_steps"]
            ),
            "use_amp": bool(cfg.yaml_cfg.get("use_amp", False)),
            "use_ema": bool(cfg.yaml_cfg.get("use_ema", False)),
            "normalization_running_stats_locked": bool(model.rgbt_lock_norm_stats),
            "thermal_trainable_tensors": len(thermal_trainable),
        }
        del optimizer, criterion, model, cfg
        gc.collect()

    reference = base_snapshots["B0"]
    for arm, snapshot in base_snapshots.items():
        if snapshot.keys() != reference.keys():
            raise RuntimeError(f"{arm} RGB/C state keys differ from B0")
        differences = [
            key for key in reference
            if not torch.equal(reference[key], snapshot[key])
        ]
        if differences:
            raise RuntimeError(f"{arm} RGB/C values differ from B0: {differences[:10]}")

    output = {
        "schema": "stql_qcer_config_preflight_v1",
        "status": "PASS",
        "common_rgb_state_bitwise_equal": True,
        "arms": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
