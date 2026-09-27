"""构造C+D严格起点，并注入红外单模态候选头供M3使用。"""

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


def copy_checked(target, target_key, source, source_key, copied):
    if source_key not in source:
        raise KeyError(f"source tensor is missing: {source_key}")
    if target_key not in target:
        raise KeyError(f"target tensor is missing: {target_key}")
    if target[target_key].shape != source[source_key].shape:
        raise RuntimeError(
            f"shape mismatch {source_key} -> {target_key}: "
            f"{tuple(source[source_key].shape)} vs {tuple(target[target_key].shape)}"
        )
    target[target_key] = source[source_key].detach().clone()
    copied.append((source_key, target_key))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--visible-d", type=Path, required=True)
    parser.add_argument("--thermal-detector", type=Path, required=True)
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
    thermal_checkpoint = torch.load(
        args.thermal_detector, map_location="cpu", weights_only=False
    )
    visible_state, visible_field = checkpoint_state(
        visible_checkpoint, "D-HRQS1 checkpoint"
    )
    thermal_state, thermal_field = checkpoint_state(
        thermal_checkpoint, "thermal detector checkpoint"
    )

    cfg = YAMLConfig(str(args.config.resolve()))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    expected = model.state_dict()
    hybrid = OrderedDict(
        (key, value.detach().clone()) for key, value in expected.items()
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
    for source_prefix, target_prefix in (
        ("backbone.", "thermal_backbone."),
        ("encoder.", "thermal_encoder."),
    ):
        for source_key, value in thermal_state.items():
            if not source_key.startswith(source_prefix):
                continue
            target_key = target_prefix + source_key[len(source_prefix) :]
            if target_key not in hybrid or hybrid[target_key].shape != value.shape:
                raise RuntimeError(f"thermal stream tensor is incompatible: {source_key}")
            hybrid[target_key] = value.detach().clone()
            copied_thermal.append((source_key, target_key))

    candidate_mapping = {
        "decoder.enc_output.proj.weight": (
            "decoder.sdtec_candidate_tokenizer.target_proj.weight"
        ),
        "decoder.enc_output.proj.bias": (
            "decoder.sdtec_candidate_tokenizer.target_proj.bias"
        ),
        "decoder.enc_output.norm.weight": (
            "decoder.sdtec_candidate_tokenizer.target_norm.weight"
        ),
        "decoder.enc_output.norm.bias": (
            "decoder.sdtec_candidate_tokenizer.target_norm.bias"
        ),
        "decoder.enc_score_head.weight": (
            "decoder.sdtec_candidate_tokenizer.target_score_head.weight"
        ),
        "decoder.enc_score_head.bias": (
            "decoder.sdtec_candidate_tokenizer.target_score_head.bias"
        ),
    }
    copied_candidate_head = []
    for source_key, target_key in candidate_mapping.items():
        copy_checked(
            hybrid,
            target_key,
            thermal_state,
            source_key,
            copied_candidate_head,
        )

    if not copied_visible or not copied_thermal:
        raise RuntimeError("M3 initialization copied an empty mature stream")
    final_weight = hybrid[
        "decoder.sdtec_logit_calibrator.delta_head.2.weight"
    ]
    final_bias = hybrid["decoder.sdtec_logit_calibrator.delta_head.2.bias"]
    if torch.count_nonzero(final_weight) or torch.count_nonzero(final_bias):
        raise RuntimeError("M3 final calibration projection is not exact zero")

    model.load_state_dict(hybrid, strict=True)
    output_checkpoint = {
        "model": hybrid,
        "ema": {"module": OrderedDict(hybrid), "updates": 0},
        "cdm3_m3_init": {
            "seed": args.seed,
            "visible_d_path": str(args.visible_d.resolve()),
            "visible_d_field": visible_field,
            "visible_d_sha256": sha256(args.visible_d),
            "thermal_detector_path": str(args.thermal_detector.resolve()),
            "thermal_detector_field": thermal_field,
            "thermal_detector_sha256": sha256(args.thermal_detector),
            "visible_keys_copied": len(copied_visible),
            "thermal_stream_keys_copied": len(copied_thermal),
            "candidate_head_mapping": copied_candidate_head,
            "strict_model_key_count": len(expected),
            "functional_start": "exact D-HRQS1 because M3 logit delta is zero",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    print(f"output={args.output}")
    print(f"visible_keys_copied={len(copied_visible)}")
    print(f"thermal_stream_keys_copied={len(copied_thermal)}")
    print(f"candidate_head_keys_copied={len(copied_candidate_head)}")
    print(f"strict_model_key_count={len(expected)}")
    print(f"output_sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
