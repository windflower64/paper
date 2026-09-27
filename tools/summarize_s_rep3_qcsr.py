#!/usr/bin/env python3
"""Summarize REP3-QCSR against paired S-series references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = ("ap", "ap50", "ap75", "aps", "apm", "apl", "ar100")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--rep11-log", type=Path, required=True)
    parser.add_argument("--rep21-log", type=Path, required=True)
    parser.add_argument("--rep3-log", type=Path, required=True)
    parser.add_argument("--best-mechanism", type=Path, required=True)
    parser.add_argument("--last-mechanism", type=Path, required=True)
    parser.add_argument("--best-scale", type=Path)
    parser.add_argument("--last-scale", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="REP3-QCSR")
    return parser.parse_args()


def read_rows(path, loss_keys=()):
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
        for key in loss_keys:
            parsed[key.removeprefix("train_loss_")] = row.get(key)
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
    rep11 = read_rows(args.rep11_log, ("train_loss_spar",))
    rep21 = read_rows(args.rep21_log, ("train_loss_mdqa",))
    rep3 = read_rows(
        args.rep3_log,
        ("train_loss_qcsr_teacher", "train_loss_qcsr_transfer"),
    )
    control_best, rep11_best, rep21_best, rep3_best = map(
        best, (control, rep11, rep21, rep3)
    )
    report = {
        "method": args.method,
        "controlled_change": (
            "private SAM-anchored detached S8/query teacher; student-only "
            "query-conditioned S8-to-S16 retention; transfer comparison "
            "defined by the selected QCSR criterion mode"
        ),
        "control_best": control_best,
        "rep1_1_best": rep11_best,
        "rep2_1_best": rep21_best,
        "rep3_best": rep3_best,
        "rep3_minus_control_best": delta(rep3_best, control_best),
        "rep3_minus_rep1_1_best": delta(rep3_best, rep11_best),
        "rep3_minus_rep2_1_best": delta(rep3_best, rep21_best),
        "best_mechanism": json.loads(args.best_mechanism.read_text(encoding="utf-8")),
        "last_mechanism": json.loads(args.last_mechanism.read_text(encoding="utf-8")),
        "best_scale_diagnosis": (
            json.loads(args.best_scale.read_text(encoding="utf-8"))
            if args.best_scale is not None
            else None
        ),
        "last_scale_diagnosis": (
            json.loads(args.last_scale.read_text(encoding="utf-8"))
            if args.last_scale is not None
            else None
        ),
        "per_epoch": [
            {
                "epoch": epoch,
                "control": control[epoch],
                "rep1_1": rep11[epoch],
                "rep2_1": rep21[epoch],
                "rep3": rep3[epoch],
                "rep3_minus_control": delta(rep3[epoch], control[epoch]),
                "rep3_minus_rep1_1": delta(rep3[epoch], rep11[epoch]),
                "rep3_minus_rep2_1": delta(rep3[epoch], rep21[epoch]),
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
