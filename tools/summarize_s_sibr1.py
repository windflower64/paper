#!/usr/bin/env python3
"""Summarize SIBR1 AP and boundary-retention results against exact controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--sibr-log", type=Path, required=True)
    parser.add_argument("--start-retention", type=Path, required=True)
    parser.add_argument("--control-retention", type=Path, required=True)
    parser.add_argument("--best-retention", type=Path, required=True)
    parser.add_argument("--last-retention", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        metrics = row.get("test_coco_eval_bbox")
        if isinstance(metrics, list) and len(metrics) >= 6:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "ap": float(metrics[0]),
                    "ap50": float(metrics[1]),
                    "ap75": float(metrics[2]),
                    "ap_small": float(metrics[3]),
                    "ap_medium": float(metrics[4]),
                    "ap_large": float(metrics[5]),
                }
            )
    if not rows:
        raise RuntimeError(f"no validation rows in {path}")
    return rows


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    control_rows = read_rows(args.control_log)
    sibr_rows = read_rows(args.sibr_log)
    control_best = max(control_rows, key=lambda row: (row["ap"], row["ap75"]))
    sibr_best = max(sibr_rows, key=lambda row: (row["ap"], row["ap75"]))
    start = load_json(args.start_retention)
    control_retention = load_json(args.control_retention)
    best_retention = load_json(args.best_retention)
    last_retention = load_json(args.last_retention)
    key = "aligned_retention_error_mean"
    report = {
        "control_best": control_best,
        "sibr_best": sibr_best,
        "delta_best": {
            metric: sibr_best[metric] - control_best[metric]
            for metric in ("ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large")
        },
        "control_last": control_rows[-1],
        "sibr_last": sibr_rows[-1],
        "retention": {
            "a00_start": start[key],
            "control_best": control_retention[key],
            "sibr_best": best_retention[key],
            "sibr_last": last_retention[key],
            "sibr_best_minus_control_best": best_retention[key]
            - control_retention[key],
            "sibr_best_improvement_from_start": start[key] - best_retention[key],
        },
        "gates": {
            "ap_beats_control": sibr_best["ap"] > control_best["ap"],
            "ap75_beats_control": sibr_best["ap75"] > control_best["ap75"],
            "boundary_retention_beats_control": best_retention[key]
            < control_retention[key],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
