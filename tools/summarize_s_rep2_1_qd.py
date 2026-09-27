#!/usr/bin/env python3
"""Summarize REP2.1 query-stop-gradient against all paired predecessors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = ("ap", "ap50", "ap75", "aps", "apm", "apl", "ar100")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--rep11-log", type=Path, required=True)
    parser.add_argument("--rep2-log", type=Path, required=True)
    parser.add_argument("--rep21-log", type=Path, required=True)
    parser.add_argument("--best-alignment", type=Path, required=True)
    parser.add_argument("--last-alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path, loss_key=None):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        values = row.get("test_coco_eval_bbox")
        if not isinstance(values, list) or len(values) < 9:
            continue
        parsed = {
            "epoch": int(row["epoch"]),
            "ap": float(values[0]),
            "ap50": float(values[1]),
            "ap75": float(values[2]),
            "aps": float(values[3]),
            "apm": float(values[4]),
            "apl": float(values[5]),
            "ar100": float(values[8]),
        }
        if loss_key:
            parsed["auxiliary_loss"] = row.get(loss_key)
        rows.append(parsed)
    if len(rows) < 6 or [row["epoch"] for row in rows[:6]] != list(range(6)):
        raise RuntimeError(f"Expected complete epochs 0..5 in {path}")
    return rows[:6]


def best(rows):
    return max(rows, key=lambda row: (row["ap"], row["ap75"]))


def delta(left, right):
    return {field: left[field] - right[field] for field in FIELDS}


def main():
    args = parse_args()
    control = read_rows(args.control_log)
    rep11 = read_rows(args.rep11_log, "train_loss_spar")
    rep2 = read_rows(args.rep2_log, "train_loss_mdqa")
    rep21 = read_rows(args.rep21_log, "train_loss_mdqa")
    control_best, rep11_best, rep2_best, rep21_best = map(
        best, (control, rep11, rep2, rep21)
    )
    report = {
        "method": "REP2.1-MDQA-QD",
        "controlled_change": "detach matched detector query before MDQA projection",
        "control_best": control_best,
        "rep1_1_best": rep11_best,
        "rep2_best": rep2_best,
        "rep2_1_best": rep21_best,
        "rep2_1_minus_control_best": delta(rep21_best, control_best),
        "rep2_1_minus_rep1_1_best": delta(rep21_best, rep11_best),
        "rep2_1_minus_rep2_best": delta(rep21_best, rep2_best),
        "best_alignment": json.loads(args.best_alignment.read_text(encoding="utf-8")),
        "last_alignment": json.loads(args.last_alignment.read_text(encoding="utf-8")),
        "per_epoch": [
            {
                "epoch": epoch,
                "control": control[epoch],
                "rep1_1": rep11[epoch],
                "rep2": rep2[epoch],
                "rep2_1": rep21[epoch],
                "rep2_1_minus_control": delta(rep21[epoch], control[epoch]),
                "rep2_1_minus_rep1_1": delta(rep21[epoch], rep11[epoch]),
                "rep2_1_minus_rep2": delta(rep21[epoch], rep2[epoch]),
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
