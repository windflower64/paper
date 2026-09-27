#!/usr/bin/env python3
"""Summarize the strictly paired REP1.1 JointInit control and progressive run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_NAMES = (
    "ap",
    "ap50",
    "ap75",
    "aps",
    "apm",
    "apl",
    "ar1",
    "ar10",
    "ar100",
    "ars",
    "arm",
    "arl",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--progressive-log", type=Path, required=True)
    parser.add_argument("--best-alignment", type=Path, required=True)
    parser.add_argument("--last-alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rep11-reference", type=float, default=0.661319)
    return parser.parse_args()


def read_log(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            source = json.loads(line)
        except json.JSONDecodeError:
            continue
        metrics = source.get("test_coco_eval_bbox")
        if not isinstance(metrics, list) or len(metrics) < 12:
            continue
        row = {"epoch": int(source["epoch"])}
        row.update(
            {name: float(value) for name, value in zip(METRIC_NAMES, metrics)}
        )
        for name in ("train_loss", "train_loss_spar"):
            if name in source:
                row[name.removeprefix("train_")] = float(source[name])
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no validation rows in {path}")
    return rows


def best(rows):
    return max(rows, key=lambda row: (row["ap"], row["ap75"]))


def mean_metrics(rows):
    return {
        name: sum(row[name] for row in rows) / len(rows)
        for name in ("ap", "ap75", "aps", "apm", "ar100")
    }


def delta(left, right):
    return {
        name: left[name] - right[name]
        for name in ("ap", "ap50", "ap75", "aps", "apm", "ar100")
    }


def main():
    args = parse_args()
    control = read_log(args.control_log)
    progressive = read_log(args.progressive_log)
    control_by_epoch = {row["epoch"]: row for row in control}
    progressive_epochs = [row["epoch"] for row in progressive]
    missing_control_epochs = [
        epoch for epoch in progressive_epochs if epoch not in control_by_epoch
    ]
    if missing_control_epochs:
        raise RuntimeError(
            f"control log misses progressive epochs: {missing_control_epochs}"
        )
    # A00 continued beyond 60 epochs.  For a 60-epoch JointInit run, compare
    # only the exact common 0-59 window rather than letting later A00 epochs
    # enter best/tail selection.
    control = [control_by_epoch[epoch] for epoch in progressive_epochs]

    control_best = best(control)
    progressive_best = best(progressive)
    best_delta = delta(progressive_best, control_best)
    same_epoch_control = control[progressive_best["epoch"]]
    same_epoch_delta = delta(progressive_best, same_epoch_control)
    control_tail = mean_metrics(control[-10:])
    progressive_tail = mean_metrics(progressive[-10:])
    tail_delta = {
        name: progressive_tail[name] - control_tail[name] for name in control_tail
    }

    gates = {
        "best_ap_gain_at_least_0_002": best_delta["ap"] >= 0.002,
        "best_aps_gain_at_least_0_002": best_delta["aps"] >= 0.002,
        "best_ap75_drop_no_more_than_0_001": best_delta["ap75"] >= -0.001,
        "tail_ap_not_below_control": tail_delta["ap"] >= 0.0,
        "reaches_rep11_reference": progressive_best["ap"] >= args.rep11_reference,
        "convincing_replacement_ap_0_6623": progressive_best["ap"] >= 0.6623,
    }
    report = {
        "protocol": "same initialization, seed, 60 epochs and batch32; only progressive SPAR differs",
        "control_epochs": len(control),
        "progressive_epochs": len(progressive),
        "control_best": control_best,
        "progressive_best": progressive_best,
        "best_to_best_delta": best_delta,
        "control_at_progressive_best_epoch": same_epoch_control,
        "same_epoch_delta": same_epoch_delta,
        "control_last": control[-1],
        "progressive_last": progressive[-1],
        "control_last10_mean": control_tail,
        "progressive_last10_mean": progressive_tail,
        "last10_mean_delta": tail_delta,
        "rep11_reference_ap": args.rep11_reference,
        "gates": gates,
        "all_selection_gates_pass": all(
            gates[name]
            for name in (
                "best_ap_gain_at_least_0_002",
                "best_aps_gain_at_least_0_002",
                "best_ap75_drop_no_more_than_0_001",
                "tail_ap_not_below_control",
                "reaches_rep11_reference",
            )
        ),
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
