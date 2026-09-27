#!/usr/bin/env python3
"""Compare no-mask, aligned-mask and deliberately shifted-mask REP1.1 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--aligned-log", type=Path, required=True)
    parser.add_argument("--shifted-log", type=Path, required=True)
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
        if isinstance(metrics, list) and len(metrics) >= 9:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "ap": float(metrics[0]),
                    "ap50": float(metrics[1]),
                    "ap75": float(metrics[2]),
                    "aps": float(metrics[3]),
                    "apm": float(metrics[4]),
                    "apl": float(metrics[5]),
                    "ar100": float(metrics[8]),
                }
            )
    if not rows:
        raise RuntimeError(f"no validation rows in {path}")
    return rows


def best(rows):
    return max(rows, key=lambda row: (row["ap"], row["ap75"]))


def subtract(left, right):
    return {
        key: left[key] - right[key]
        for key in ("ap", "ap50", "ap75", "aps", "apm", "apl", "ar100")
    }


def main():
    args = parse_args()
    control_rows = read_rows(args.control_log)
    aligned_rows = read_rows(args.aligned_log)
    shifted_rows = read_rows(args.shifted_log)
    control_best = best(control_rows)
    aligned_best = best(aligned_rows)
    shifted_best = best(shifted_rows)
    report = {
        "control_best": control_best,
        "aligned_best": aligned_best,
        "shifted_best": shifted_best,
        "aligned_minus_control_best": subtract(aligned_best, control_best),
        "shifted_minus_control_best": subtract(shifted_best, control_best),
        "aligned_minus_shifted_best": subtract(aligned_best, shifted_best),
        "last_epoch": {
            "control": control_rows[-1],
            "aligned": aligned_rows[-1],
            "shifted": shifted_rows[-1],
            "aligned_minus_shifted": subtract(aligned_rows[-1], shifted_rows[-1]),
        },
        "causal_gate_pass": (
            aligned_best["ap"] > control_best["ap"]
            and aligned_best["ap"] > shifted_best["ap"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
