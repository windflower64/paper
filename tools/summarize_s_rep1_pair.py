#!/usr/bin/env python3
"""Summarize the paired A00 fine-tuning and REP1-SPAR experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--spar-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_log(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        metrics = row.get("test_coco_eval_bbox")
        if isinstance(metrics, list) and len(metrics) >= 3:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "ap": float(metrics[0]),
                    "ap50": float(metrics[1]),
                    "ap75": float(metrics[2]),
                }
            )
    if not rows:
        raise RuntimeError(f"no validation rows in {path}")
    return rows


def best(rows):
    return max(rows, key=lambda row: (row["ap"], row["ap75"]))


def main():
    args = parse_args()
    control_rows = read_log(args.control_log)
    spar_rows = read_log(args.spar_log)
    control_best = best(control_rows)
    spar_best = best(spar_rows)
    report = {
        "control_best": control_best,
        "spar_best": spar_best,
        "delta_ap": spar_best["ap"] - control_best["ap"],
        "delta_ap75": spar_best["ap75"] - control_best["ap75"],
        "spar_beats_control_ap": spar_best["ap"] > control_best["ap"],
        "spar_beats_control_ap75": spar_best["ap75"] > control_best["ap75"],
        "control_last": control_rows[-1],
        "spar_last": spar_rows[-1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
