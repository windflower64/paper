"""Build one audited RGB/C + corrected-IR initializer for M-OTE2 arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict(payload, label):
    if isinstance(payload.get("ema"), dict) and isinstance(payload["ema"].get("module"), dict):
        return payload["ema"]["module"], "ema.module"
    if isinstance(payload.get("model"), dict):
        return payload["model"], "model"
    raise RuntimeError(f"No model or EMA state in {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--visible", type=Path, required=True)
    parser.add_argument("--thermal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    if not args.repo.is_dir():
        raise FileNotFoundError(args.repo)
    for path in (args.config, args.visible, args.thermal):
        if not path.is_file():
            raise FileNotFoundError(path)

    import sys

    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config.resolve()))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    expected = cfg.model.state_dict()
    visible_payload = torch.load(args.visible, map_location="cpu", weights_only=False)
    thermal_payload = torch.load(args.thermal, map_location="cpu", weights_only=False)
    visible, visible_field = state_dict(visible_payload, "visible source")
    thermal, thermal_field = state_dict(thermal_payload, "verified IR source")

    output = OrderedDict()
    visible_loaded, thermal_loaded, missing_visible, missing_thermal = [], [], [], []
    for key, template in expected.items():
        if key.startswith(("backbone.", "encoder.", "decoder.")):
            source = visible.get(key)
            if source is not None and source.shape == template.shape:
                output[key] = source.detach().clone()
                visible_loaded.append(key)
            else:
                missing_visible.append(key)
        elif key.startswith("thermal_backbone."):
            source_key = "backbone." + key[len("thermal_backbone.") :]
            source = thermal.get(source_key)
            if source is not None and source.shape == template.shape:
                output[key] = source.detach().clone()
                thermal_loaded.append(key)
            else:
                missing_thermal.append(key)
        elif key.startswith("thermal_encoder."):
            source_key = "encoder." + key[len("thermal_encoder.") :]
            source = thermal.get(source_key)
            if source is not None and source.shape == template.shape:
                output[key] = source.detach().clone()
                thermal_loaded.append(key)
            else:
                missing_thermal.append(key)

    if missing_visible or missing_thermal:
        raise RuntimeError(
            "Initializer incomplete: "
            f"missing_visible={missing_visible[:12]}, "
            f"missing_thermal={missing_thermal[:12]}"
        )
    if not visible_loaded or not thermal_loaded:
        raise RuntimeError("No visible or corrected thermal tensors were loaded")
    if any(key.startswith(("sd2_conditioner.", "mote_fusion.")) for key in output):
        raise RuntimeError("M module parameters must be freshly initialized per arm")

    metadata = {
        "schema": "mote2_common_init_v1",
        "visible_source": str(args.visible.resolve()),
        "visible_source_sha256": sha256(args.visible),
        "visible_source_field": visible_field,
        "thermal_source": str(args.thermal.resolve()),
        "thermal_source_sha256": sha256(args.thermal),
        "thermal_source_field": thermal_field,
        "visible_key_count": len(visible_loaded),
        "thermal_key_count": len(thermal_loaded),
        "loaded_prefix_counts": {
            "backbone": sum(k.startswith("backbone.") for k in output),
            "encoder": sum(k.startswith("encoder.") for k in output),
            "decoder": sum(k.startswith("decoder.") for k in output),
            "thermal_backbone": sum(k.startswith("thermal_backbone.") for k in output),
            "thermal_encoder": sum(k.startswith("thermal_encoder.") for k in output),
        },
    }
    payload = {
        "model": output,
        "ema": {"module": OrderedDict(output), "updates": 0},
        "mote2_common_init": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    metadata["output"] = str(args.output.resolve())
    metadata["output_sha256"] = sha256(args.output)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
