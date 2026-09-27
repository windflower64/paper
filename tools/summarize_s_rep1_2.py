#!/usr/bin/env python3
"""Compare REP1.2 mask-resolution loss with REP1.1 and its no-SAM control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = ("ap", "ap50", "ap75", "aps", "apm", "apl", "ar100")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--rep11-log", type=Path, required=True)
    parser.add_argument("--rep12-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
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
                "spar_loss": row.get("train_loss_spar"),
            }
        )
    if len(rows) < 6:
        raise RuntimeError(f"expected at least 6 validation rows in {path}")
    return rows[:6]


def best(rows):
    return max(rows, key=lambda row: (row["ap"], row["ap75"]))


def delta(left, right):
    return {field: left[field] - right[field] for field in FIELDS}


def main():
    args = parse_args()
    control = read_rows(args.control_log)
    rep11 = read_rows(args.rep11_log)
    rep12 = read_rows(args.rep12_log)
    control_best, rep11_best, rep12_best = map(best, (control, rep11, rep12))
    report = {
        "controlled_change": "SPAR loss resolution only: feature -> mask",
        "control_best": control_best,
        "rep1_1_best": rep11_best,
        "rep1_2_best": rep12_best,
        "rep1_2_minus_control_best": delta(rep12_best, control_best),
        "rep1_2_minus_rep1_1_best": delta(rep12_best, rep11_best),
        "per_epoch": [
            {
                "epoch": epoch,
                "control": control[epoch],
                "rep1_1": rep11[epoch],
                "rep1_2": rep12[epoch],
                "rep1_2_minus_rep1_1": delta(rep12[epoch], rep11[epoch]),
            }
            for epoch in range(6)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
