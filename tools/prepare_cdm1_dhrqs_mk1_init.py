"""Merge the first genuine C + D + M RGB-T initialization.

The D-HRQS1 checkpoint supplies the complete visible detector, including the
C-GQ1 backbone and HRQS adapter.  A trained M1-K1 checkpoint supplies the
thermal backbone/encoder and the coordinate-free thermal reader.  Only the
three bounded fusion residuals are reset to zero, so the merged model starts
as the exact D-HRQS1 visible detector while retaining mature thermal features
and reader weights for subsequent frozen-reader adaptation.
"""

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
    parser.add_argument("--visible-d", type=Path, required=True)
    parser.add_argument("--multimodal-m", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    visible_checkpoint = torch.load(
        args.visible_d, map_location="cpu", weights_only=False
    )
    multimodal_checkpoint = torch.load(
        args.multimodal_m, map_location="cpu", weights_only=False
    )
    visible_state, visible_field = checkpoint_state(
        visible_checkpoint, "D-HRQS1 checkpoint"
    )
    multimodal_state, multimodal_field = checkpoint_state(
        multimodal_checkpoint, "M1-K1 checkpoint"
    )

    hybrid = OrderedDict((key, value.clone()) for key, value in visible_state.items())
    copied_thermal = []
    copied_sdtec = []
    for key, value in multimodal_state.items():
        if key.startswith(("thermal_backbone.", "thermal_encoder.")):
            hybrid[key] = value.clone()
            copied_thermal.append(key)
        elif key.startswith("decoder.sdtec_"):
            hybrid[key] = value.clone()
            copied_sdtec.append(key)

    if not copied_thermal:
        raise RuntimeError("M1-K1 source contains no thermal stream tensors")
    if not copied_sdtec:
        raise RuntimeError("M1-K1 source contains no SDTEC reader tensors")

    reset_scales = []
    for key in list(hybrid):
        if key.startswith("decoder.sdtec_couplers.") and key.endswith(
            ".residual_scale"
        ):
            hybrid[key] = torch.zeros_like(hybrid[key])
            reset_scales.append(key)
    if len(reset_scales) != 3:
        raise RuntimeError(
            f"expected three independent SDTEC residual scales, got {reset_scales}"
        )

    cfg = YAMLConfig(str(args.config.resolve()))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    expected_state = model.state_dict()
    missing = sorted(set(expected_state) - set(hybrid))
    unexpected = sorted(set(hybrid) - set(expected_state))
    shape_mismatches = [
        key
        for key in expected_state.keys() & hybrid.keys()
        if expected_state[key].shape != hybrid[key].shape
    ]
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "CDM1 merged state does not strictly match the configured model: "
            f"missing={missing[:10]} unexpected={unexpected[:10]} "
            f"shape_mismatches={shape_mismatches[:10]}"
        )
    model.load_state_dict(hybrid, strict=True)

    output_checkpoint = {
        "model": hybrid,
        "ema": {"module": OrderedDict(hybrid), "updates": 0},
        "cdm1_init": {
            "visible_d_path": str(args.visible_d.resolve()),
            "visible_d_field": visible_field,
            "visible_d_sha256": sha256(args.visible_d),
            "multimodal_m_path": str(args.multimodal_m.resolve()),
            "multimodal_m_field": multimodal_field,
            "multimodal_m_sha256": sha256(args.multimodal_m),
            "thermal_keys_copied": len(copied_thermal),
            "sdtec_keys_copied": len(copied_sdtec),
            "sdtec_residual_scales_reset": reset_scales,
            "strict_model_key_count": len(expected_state),
            "training_scope": "decoder.sdtec_* only",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)

    print(f"output={args.output}")
    print(f"visible_d_field={visible_field} visible_d_keys={len(visible_state)}")
    print(
        f"multimodal_m_field={multimodal_field} "
        f"multimodal_m_keys={len(multimodal_state)}"
    )
    print(f"thermal_keys_copied={len(copied_thermal)}")
    print(f"sdtec_keys_copied={len(copied_sdtec)}")
    print(f"reset_scales={reset_scales}")
    print(f"strict_model_key_count={len(expected_state)}")
    print(f"output_sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
