"""Build and audit the common initialization used by STQL/QCER B0--B5.

The RGB/C part comes only from the historical common task-start checkpoint.
The frozen thermal backbone/encoder comes only from the verified, corrected-IR
checkpoint.  Parameters that did not exist in the historical common checkpoint
(C-GQ1 additions and one-class score heads) are initialized once from seed 0
and stored in the common file so every arm receives byte-identical RGB/C state.
STQL/QCER parameters are deliberately left to constructor initialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch


BASE_PREFIXES = ("backbone.", "encoder.", "decoder.")
THERMAL_PREFIXES = ("thermal_backbone.", "thermal_encoder.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_state(checkpoint, name: str):
    ema = checkpoint.get("ema") if isinstance(checkpoint, dict) else None
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"], "ema.module"
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"], "model"
    raise RuntimeError(f"{name} has neither ema.module nor model state")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--visible", type=Path, required=True)
    parser.add_argument("--thermal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for path in (args.repo, args.config, args.visible, args.thermal):
        if not path.exists():
            raise FileNotFoundError(path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config.resolve()))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    expected = cfg.model.state_dict()

    visible_checkpoint = torch.load(
        args.visible, map_location="cpu", weights_only=False
    )
    thermal_checkpoint = torch.load(
        args.thermal, map_location="cpu", weights_only=False
    )
    visible_state, visible_field = checkpoint_state(
        visible_checkpoint, "historical common initialization"
    )
    thermal_state, thermal_field = checkpoint_state(
        thermal_checkpoint, "verified thermal checkpoint"
    )

    output_state = OrderedDict()
    rgb_from_visible = []
    rgb_from_constructor = []
    thermal_mapped = []

    # Materialize the complete RGB/C state.  Constructor fallbacks are explicit
    # and audited rather than silently depending on per-arm module creation order.
    for key, expected_value in expected.items():
        if not key.startswith(BASE_PREFIXES):
            continue
        source_value = visible_state.get(key)
        if source_value is not None and source_value.shape == expected_value.shape:
            output_state[key] = source_value.detach().clone()
            rgb_from_visible.append(key)
        else:
            output_state[key] = expected_value.detach().clone()
            rgb_from_constructor.append(key)

    for source_key, source_value in thermal_state.items():
        if source_key.startswith("backbone."):
            target_key = "thermal_backbone." + source_key[len("backbone.") :]
        elif source_key.startswith("encoder."):
            target_key = "thermal_encoder." + source_key[len("encoder.") :]
        else:
            continue
        expected_value = expected.get(target_key)
        if expected_value is None or expected_value.shape != source_value.shape:
            raise RuntimeError(
                f"thermal mapping mismatch: {source_key} -> {target_key}; "
                f"source={tuple(source_value.shape)}, "
                f"expected={None if expected_value is None else tuple(expected_value.shape)}"
            )
        output_state[target_key] = source_value.detach().clone()
        thermal_mapped.append(target_key)

    expected_thermal = sorted(
        key for key in expected if key.startswith(THERMAL_PREFIXES)
    )
    missing_thermal = sorted(set(expected_thermal).difference(thermal_mapped))
    if missing_thermal:
        raise RuntimeError(f"incomplete thermal mapping: {missing_thermal[:20]}")

    forbidden = [
        key
        for key in output_state
        if key.startswith(("qcer.", "stql_", "sd2_", "sgc_", "hrqs_"))
    ]
    if forbidden:
        raise RuntimeError(f"forbidden new/legacy module keys: {forbidden[:20]}")

    metadata = {
        "schema": "stql_qcer_common_init_v1",
        "seed": args.seed,
        "rgb_source": str(args.visible.resolve()),
        "rgb_source_field": visible_field,
        "rgb_source_sha256": sha256(args.visible),
        "thermal_source": str(args.thermal.resolve()),
        "thermal_source_field": thermal_field,
        "thermal_source_sha256": sha256(args.thermal),
        "thermal_source_last_epoch": thermal_checkpoint.get("last_epoch"),
        "rgb_from_historical_keys": len(rgb_from_visible),
        "rgb_constructor_fallback_keys": rgb_from_constructor,
        "thermal_mapped_keys": len(thermal_mapped),
        "stored_keys": len(output_state),
        "excluded_prefixes": ["qcer.", "stql_*", "all old thermal from RGB source"],
        "prefix_counts": dict(Counter(key.split(".")[0] for key in output_state)),
    }
    payload = {
        "model": output_state,
        "ema": {"module": OrderedDict(output_state), "updates": 0},
        "stql_qcer_common_init": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    metadata["output"] = str(args.output.resolve())
    metadata["output_sha256"] = sha256(args.output)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
