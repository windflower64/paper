#!/usr/bin/env python3
"""Summarize the uninterrupted 30-epoch REP1.1 restart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("ap", "ap50", "ap75", "aps", "apm", "apl", "ar100")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-log", type=Path, required=True)
    parser.add_argument("--restart-log", type=Path, required=True)
    parser.add_argument("--best-alignment", type=Path, required=True)
    parser.add_argument("--last-alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        values = row.get("test_coco_eval_bbox")
        if not isinstance(values, list) or len(values) < 9:
            continue
        rows.append(
            {
                "epoch": int(row["epoch"]),
                "ap": float(values[0]),
                "ap50": float(values[1]),
                "ap75": float(values[2]),
                "aps": float(values[3]),
                "apm": float(values[4]),
                "apl": float(values[5]),
                "ar100": float(values[8]),
                "spar_loss": float(row.get("train_loss_spar", 0.0)),
            }
        )
    if not rows:
        raise RuntimeError(f"no validation rows in {path}")
    return rows


def best(rows):
    return max(rows, key=lambda row: (row["ap"], row["ap75"]))


def delta(left, right):
    return {metric: left[metric] - right[metric] for metric in METRICS}


def main():
    args = parse_args()
    original = read_rows(args.original_log)
    restart = read_rows(args.restart_log)
    actual_epochs = [row["epoch"] for row in restart]
    if actual_epochs != list(range(30)):
        raise RuntimeError(f"expected uninterrupted epochs 0..29, got {actual_epochs}")
    original_best = best(original)
    restart_best = best(restart)
    windows = {
        "0_to_5": best(restart[0:6]),
        "6_to_11": best(restart[6:12]),
        "12_to_19": best(restart[12:20]),
        "20_to_29": best(restart[20:30]),
    }
    report = {
        "protocol": {
            "restart_from_a00": True,
            "resume_used": False,
            "epochs": 30,
            "continuous_single_process": True,
        },
        "original_rep1_1_best_0_to_5": original_best,
        "restart_best_0_to_29": restart_best,
        "restart_minus_original_best": delta(restart_best, original_best),
        "restart_last": restart[-1],
        "window_bests": windows,
        "best_alignment": json.loads(args.best_alignment.read_text(encoding="utf-8")),
        "last_alignment": json.loads(args.last_alignment.read_text(encoding="utf-8")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
