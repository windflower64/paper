"""Build the strict C+D starting checkpoint for M2 logit calibration.

D-HRQS1 supplies the complete visible detector.  The successful M1-K1 reader
supplies the frozen thermal backbone/encoder and every tokenizer tensor whose
shape is compatible with the new K=8 tokenizer.  The new multi-token slots and
the classification-only calibrator keep their deterministic fresh
initialization; its final projection is exactly zero, so the merged checkpoint
is functionally identical to D-HRQS1 before M2 training starts.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    cfg = YAMLConfig(str(args.config.resolve()))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    expected_state = model.state_dict()
    hybrid = OrderedDict(
        (key, value.detach().clone()) for key, value in expected_state.items()
    )

    copied_visible = []
    for key, value in visible_state.items():
        if key.startswith(("thermal_backbone.", "thermal_encoder.")):
            continue
        if key.startswith("decoder.sdtec_"):
            continue
        if key in hybrid and hybrid[key].shape == value.shape:
            hybrid[key] = value.detach().clone()
            copied_visible.append(key)

    copied_thermal = []
    copied_tokenizer = []
    skipped_tokenizer = []
    for key, value in multimodal_state.items():
        if key.startswith(("thermal_backbone.", "thermal_encoder.")):
            if key not in hybrid or hybrid[key].shape != value.shape:
                raise RuntimeError(f"thermal tensor is incompatible with M2: {key}")
            hybrid[key] = value.detach().clone()
            copied_thermal.append(key)
        elif key.startswith("decoder.sdtec_tokenizer."):
            if key in hybrid and hybrid[key].shape == value.shape:
                hybrid[key] = value.detach().clone()
                copied_tokenizer.append(key)
            else:
                skipped_tokenizer.append(key)

    if not copied_visible:
        raise RuntimeError("D-HRQS1 source copied no visible detector tensors")
    if not copied_thermal:
        raise RuntimeError("M1-K1 source copied no thermal stream tensors")
    if not copied_tokenizer:
        raise RuntimeError("M1-K1 source copied no compatible tokenizer tensors")

    calibrator_keys = [
        key for key in hybrid if key.startswith("decoder.sdtec_logit_calibrator.")
    ]
    if not calibrator_keys:
        raise RuntimeError("configured model contains no M2 logit calibrator")
    final_weight = hybrid[
        "decoder.sdtec_logit_calibrator.delta_head.2.weight"
    ]
    final_bias = hybrid["decoder.sdtec_logit_calibrator.delta_head.2.bias"]
    if torch.count_nonzero(final_weight) or torch.count_nonzero(final_bias):
        raise RuntimeError("M2 final logit projection is not exact zero at initialization")

    model.load_state_dict(hybrid, strict=True)
    output_checkpoint = {
        "model": hybrid,
        "ema": {"module": OrderedDict(hybrid), "updates": 0},
        "cdm2_m2_init": {
            "seed": args.seed,
            "visible_d_path": str(args.visible_d.resolve()),
            "visible_d_field": visible_field,
            "visible_d_sha256": sha256(args.visible_d),
            "multimodal_m_path": str(args.multimodal_m.resolve()),
            "multimodal_m_field": multimodal_field,
            "multimodal_m_sha256": sha256(args.multimodal_m),
            "visible_keys_copied": len(copied_visible),
            "thermal_keys_copied": len(copied_thermal),
            "compatible_k1_tokenizer_keys_copied": copied_tokenizer,
            "incompatible_k1_tokenizer_keys_skipped": skipped_tokenizer,
            "fresh_m2_calibrator_keys": len(calibrator_keys),
            "strict_model_key_count": len(expected_state),
            "training_scope": "decoder.sdtec_* only",
            "functional_start": "exact D-HRQS1 because M2 logit delta is zero",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)

    print(f"output={args.output}")
    print(f"visible_keys_copied={len(copied_visible)}")
    print(f"thermal_keys_copied={len(copied_thermal)}")
    print(f"compatible_tokenizer_keys_copied={len(copied_tokenizer)}")
    print(f"skipped_tokenizer_keys={skipped_tokenizer}")
    print(f"fresh_m2_calibrator_keys={len(calibrator_keys)}")
    print(f"strict_model_key_count={len(expected_state)}")
    print(f"output_sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
