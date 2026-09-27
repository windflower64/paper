"""Summarize exact COCO metrics saved by M-SDTEC1 causal validations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pycocotools.cocoeval import COCOeval


METRIC_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "APS",
    "APM",
    "APL",
    "AR1",
    "AR10",
    "AR100",
    "ARS",
    "ARM",
    "ARL",
)


def load_stats(path):
    evaluation = torch.load(path, map_location="cpu", weights_only=False)
    evaluator = COCOeval()
    evaluator.eval = evaluation
    evaluator.params = evaluation["params"]
    evaluator.summarize()
    return {name: float(value) for name, value in zip(METRIC_NAMES, evaluator.stats)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = {}
    required_modes = ("normal", "zero")
    optional_modes = ("batch_shuffle", "feature_permute", "global_mismatch")
    for mode in required_modes:
        eval_path = args.root / mode / "eval.pth"
        if not eval_path.is_file():
            raise FileNotFoundError(eval_path)
        results[mode] = load_stats(eval_path)
    for mode in optional_modes:
        eval_path = args.root / mode / "eval.pth"
        if eval_path.is_file():
            results[mode] = load_stats(eval_path)

    normal = results["normal"]
    for mode, metrics in results.items():
        metrics["delta_AP_vs_normal"] = metrics["AP"] - normal["AP"]
        metrics["delta_AP75_vs_normal"] = metrics["AP75"] - normal["AP75"]

    summary = {
        "results": results,
        "normal_minus_zero_AP": normal["AP"] - results["zero"]["AP"],
        "thermal_input_effect_pass": normal["AP"] > results["zero"]["AP"],
    }
    if "feature_permute" in results:
        permutation_gap = abs(
            results["feature_permute"]["AP"] - results["normal"]["AP"]
        )
        summary["feature_permute_abs_AP_gap"] = permutation_gap
        summary["coordinate_free_AP_tolerance"] = 1e-6
        summary["coordinate_free_metric_pass"] = permutation_gap <= 1e-6
    if "global_mismatch" in results:
        paired_gap = normal["AP"] - results["global_mismatch"]["AP"]
        summary["normal_minus_global_mismatch_AP"] = paired_gap
        summary["paired_thermal_content_pass"] = paired_gap > 0.0
    output = args.output or (args.root / "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()
