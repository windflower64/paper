"""Strict structural audit for an M-SDTEC1 hybrid tuning checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    state = checkpoint["ema"]["module"]
    incompatible = model.load_state_dict(state, strict=False)

    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [
        key for key in missing if not key.startswith("decoder.sdtec_")
    ]
    thermal_model_keys = [
        key
        for key in model.state_dict()
        if key.startswith(("thermal_backbone.", "thermal_encoder."))
    ]
    thermal_loaded = [key for key in thermal_model_keys if key in state]

    print(f"model_keys={len(model.state_dict())}")
    print(f"checkpoint_keys={len(state)}")
    print(f"missing_keys={len(missing)}")
    print(f"unexpected_keys={len(unexpected)}")
    print(f"thermal_model_keys={len(thermal_model_keys)}")
    print(f"thermal_loaded_keys={len(thermal_loaded)}")
    print(f"sdtec_constructor_keys={len(missing) - len(invalid_missing)}")

    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:10]}")
    if invalid_missing:
        raise RuntimeError(f"non-SDTEC keys missing: {invalid_missing[:10]}")
    if len(thermal_loaded) != len(thermal_model_keys):
        raise RuntimeError("thermal branch mapping is incomplete")
    if not missing:
        raise RuntimeError("SDTEC keys should be absent for constructor initialization")
    print("hybrid_init_audit=PASS")


if __name__ == "__main__":
    main()
