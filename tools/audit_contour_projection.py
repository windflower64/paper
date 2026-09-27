#!/usr/bin/env python3
"""Check whether pseudo-contours remain distinguishable after projection to S8."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def statistics(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--causality-script-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.causality_script_dir))
    from diagnose_true_contour_causality import mask_modes

    records = [item for item in json.loads(args.records.read_text(encoding="utf-8")) if item["accepted"]]
    metrics = {"contour_box_cosine": [], "contour_interior_cosine": [], "contour_shifted_cosine": [], "contour_mass": []}
    with torch.inference_mode():
        for record in records:
            masks = mask_modes(record, 512, 640, 64, 80, "cpu")
            contour = masks["true_contour"].flatten()
            metrics["contour_box_cosine"].append(float(F.cosine_similarity(contour, masks["box_edge"].flatten(), dim=0, eps=1e-12)))
            metrics["contour_interior_cosine"].append(float(F.cosine_similarity(contour, masks["target_interior"].flatten(), dim=0, eps=1e-12)))
            metrics["contour_shifted_cosine"].append(float(F.cosine_similarity(contour, masks["shifted_contour"].flatten(), dim=0, eps=1e-12)))
            metrics["contour_mass"].append(float(contour.sum()))
    summary = {name: statistics(values) for name, values in metrics.items()}
    summary["decision_gate"] = {
        "pass": summary["contour_box_cosine"]["median"] < 0.95 and summary["contour_shifted_cosine"]["median"] < 0.80,
        "rule": "median contour-vs-box cosine < 0.95 and contour-vs-one-cell-shift cosine < 0.80",
        "interpretation": "Failure means S8 spatial resolution cannot distinguish the proposed supports; detector causal evaluation should not be run.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
