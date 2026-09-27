#!/usr/bin/env python3
"""Verify that the training-only SPAR branch is absent during inference.

The check separates physical SPAR removal from D-FINE's general ``deploy()``
conversion so a repository-wide conversion issue cannot be attributed to SPAR.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch


def checkpoint_weights(path: Path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        ema = state.get("ema")
        if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
            return ema["module"], "ema.module"
        if isinstance(state.get("model"), dict):
            return state["model"], "model"
    return state, "root"


def tensor_stats(reference: torch.Tensor, deployed: torch.Tensor):
    difference = (reference.float() - deployed.float()).abs()
    return {
        "shape": list(reference.shape),
        "max_abs_difference": float(difference.max().item()),
        "mean_abs_difference": float(difference.mean().item()),
        "allclose_atol_1e-5_rtol_1e-4": bool(
            torch.allclose(reference, deployed, atol=1e-5, rtol=1e-4)
        ),
    }


def compare_outputs(reference_output, candidate_output):
    comparisons = {}
    for key in ("pred_logits", "pred_boxes"):
        if key not in reference_output or key not in candidate_output:
            raise RuntimeError(f"missing required detector output: {key}")
        comparisons[key] = tensor_stats(reference_output[key], candidate_output[key])
    return comparisons


def comparisons_pass(comparisons):
    return all(
        item["allclose_atol_1e-5_rtol_1e-4"] for item in comparisons.values()
    )


def remove_spar_only(model):
    model.eval()
    if hasattr(model.backbone, "spar_fusion"):
        model.backbone.spar_fusion = None
        model.backbone.spar_enabled = False
        model.backbone.spar_fused_features = None
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expect-spar", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    torch.manual_seed(20260824)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    reference = cfg.model.eval()
    weights, source = checkpoint_weights(args.checkpoint)
    incompatible = reference.load_state_dict(weights, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict load was not clean: {incompatible}")

    spar_removed = remove_spar_only(copy.deepcopy(reference)).eval()
    deployed = copy.deepcopy(reference).deploy().eval()
    device = torch.device(args.device)
    reference = reference.to(device)
    spar_removed = spar_removed.to(device)
    deployed = deployed.to(device)
    sample = torch.randn(1, 3, args.height, args.width, device=device)

    with torch.inference_mode():
        reference_output = reference(sample)
        reference_repeat_output = reference(sample)
        spar_removed_output = spar_removed(sample)
        deployed_output = deployed(sample)

    repeat_comparisons = compare_outputs(reference_output, reference_repeat_output)
    spar_removal_comparisons = compare_outputs(reference_output, spar_removed_output)
    full_deploy_comparisons = compare_outputs(reference_output, deployed_output)

    reference_parameters = sum(p.numel() for p in reference.parameters())
    spar_removed_parameters = sum(p.numel() for p in spar_removed.parameters())
    deployed_parameters = sum(p.numel() for p in deployed.parameters())
    reference_state_keys = list(reference.state_dict())
    deployed_state_keys = list(deployed.state_dict())
    reference_spar_keys = [key for key in reference_state_keys if "spar_fusion" in key]
    deployed_spar_keys = [key for key in deployed_state_keys if "spar_fusion" in key]

    spar_expectation_pass = (
        len(reference_spar_keys) > 0 if args.expect_spar else len(reference_spar_keys) == 0
    )
    spar_removal_pass = (
        comparisons_pass(repeat_comparisons)
        and comparisons_pass(spar_removal_comparisons)
        and spar_expectation_pass
        and len(deployed_spar_keys) == 0
    )
    output = {
        "status": "PASS" if spar_removal_pass else "FAIL",
        "scope": "SPAR inference removal; full deploy conversion reported separately",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "weight_source": source,
        "device": str(device),
        "input_shape": [1, 3, args.height, args.width],
        "reference_parameter_count": reference_parameters,
        "spar_removed_parameter_count": spar_removed_parameters,
        "deployed_parameter_count": deployed_parameters,
        "spar_removed_parameter_count_delta": (
            reference_parameters - spar_removed_parameters
        ),
        "full_deploy_parameter_count_delta": reference_parameters - deployed_parameters,
        "reference_spar_state_key_count": len(reference_spar_keys),
        "deployed_spar_state_key_count": len(deployed_spar_keys),
        "reference_repeat_comparisons": repeat_comparisons,
        "spar_removal_comparisons": spar_removal_comparisons,
        "full_deploy_conversion_status": (
            "PASS" if comparisons_pass(full_deploy_comparisons) else "FAIL"
        ),
        "full_deploy_comparisons": full_deploy_comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if output["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
