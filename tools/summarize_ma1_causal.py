"""汇总M-A1学习后权重的精确COCO指标与因果裁决。"""

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
    return {
        name: float(value)
        for name, value in zip(METRIC_NAMES, evaluator.stats)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict-reference-ap", type=float, required=True)
    parser.add_argument("--minimum-gain", type=float, default=0.001)
    args = parser.parse_args()

    modes = (
        "normal",
        "zero_content_valid",
        "batch_shuffle",
        "feature_permute",
        "global_mismatch",
    )
    results = {}
    for mode in modes:
        eval_path = args.root / mode / "eval.pth"
        if not eval_path.is_file():
            raise FileNotFoundError(eval_path)
        results[mode] = load_stats(eval_path)

    normal = results["normal"]
    for mode, metrics in results.items():
        metrics["delta_AP_vs_normal"] = metrics["AP"] - normal["AP"]
        metrics["delta_AP75_vs_normal"] = (
            metrics["AP75"] - normal["AP75"]
        )

    zero_gap = normal["AP"] - results["zero_content_valid"]["AP"]
    batch_gap = normal["AP"] - results["batch_shuffle"]["AP"]
    global_gap = normal["AP"] - results["global_mismatch"]["AP"]
    spatial_gap = normal["AP"] - results["feature_permute"]["AP"]
    gain = normal["AP"] - args.strict_reference_ap
    summary = {
        "status": "PASS" if (
            gain >= args.minimum_gain and zero_gap > 0.0
        ) else "FAIL",
        "checkpoint_role": "highest-AP saved checkpoint after fusion opened",
        "strict_reference_ap": args.strict_reference_ap,
        "minimum_required_gain": args.minimum_gain,
        "normal_gain_over_strict_reference": gain,
        "normal_minus_zero_content_AP": zero_gap,
        "normal_minus_batch_shuffle_AP": batch_gap,
        "normal_minus_global_mismatch_AP": global_gap,
        "normal_minus_feature_permute_AP": spatial_gap,
        "thermal_content_helped": zero_gap > 0.0,
        "correct_pairing_better_than_batch_shuffle": batch_gap > 0.0,
        "correct_pairing_better_than_global_mismatch": global_gap > 0.0,
        "spatial_arrangement_used": abs(spatial_gap) > 1e-6,
        "results": results,
    }
    output = args.output or (args.root / "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()
