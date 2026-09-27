"""Summarize original-test development metrics for M-SDTEC1-R2 checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import torch
from pycocotools.cocoeval import COCOeval


METRIC_NAMES = (
    "AP", "AP50", "AP75", "APS", "APM", "APL",
    "AR1", "AR10", "AR100", "ARS", "ARM", "ARL",
)

CANDIDATES = (
    ("rgb_gq1_baseline", "RGB GQ1 baseline"),
    ("reader_best_e2", "Reader best validation AP, epoch 2"),
    ("reader_e4", "Reader balanced candidate, epoch 4"),
    ("reader_e9", "Reader checkpoint, epoch 9"),
    ("reader_e14", "Reader checkpoint, epoch 14"),
    ("reader_e19", "Reader checkpoint, epoch 19"),
    ("reader_e24", "Reader checkpoint, epoch 24"),
    ("reader_e29", "Reader checkpoint, epoch 29"),
)


def load_stats(path: Path) -> dict[str, float]:
    evaluation = torch.load(path, map_location="cpu", weights_only=False)
    evaluator = COCOeval()
    evaluator.eval = evaluation
    evaluator.params = evaluation["params"]
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.summarize()
    return {
        name: float(value)
        for name, value in zip(METRIC_NAMES, evaluator.stats)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = {}
    for key, description in CANDIDATES:
        eval_path = args.root / key / "eval.pth"
        if not eval_path.is_file():
            raise FileNotFoundError(eval_path)
        results[key] = {
            "description": description,
            **load_stats(eval_path),
        }

    baseline = results["rgb_gq1_baseline"]
    for metrics in results.values():
        metrics["delta_AP_vs_rgb"] = metrics["AP"] - baseline["AP"]
        metrics["delta_AP75_vs_rgb"] = metrics["AP75"] - baseline["AP75"]
        metrics["delta_APS_vs_rgb"] = metrics["APS"] - baseline["APS"]

    fusion_keys = [key for key, _ in CANDIDATES if key != "rgb_gq1_baseline"]
    best_ap_key = max(fusion_keys, key=lambda key: results[key]["AP"])
    # Balanced selection requires simultaneous gains in AP, AP75 and APS;
    # among eligible checkpoints, primary AP remains the ranking metric.
    balanced_keys = [
        key for key in fusion_keys
        if results[key]["AP"] > baseline["AP"]
        and results[key]["AP75"] > baseline["AP75"]
        and results[key]["APS"] > baseline["APS"]
    ]
    best_balanced_key = (
        max(balanced_keys, key=lambda key: results[key]["AP"])
        if balanced_keys else None
    )
    summary = {
        "protocol": {
            "split_role": "original_test_used_as_development_validation",
            "selection_metric": "AP",
            "balanced_requirements": "AP, AP75 and APS all above RGB baseline",
        },
        "results": results,
        "best_testdev_AP_candidate": best_ap_key,
        "best_testdev_balanced_candidate": best_balanced_key,
    }
    output = args.output or (args.root / "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()
