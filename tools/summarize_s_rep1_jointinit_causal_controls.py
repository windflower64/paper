#!/usr/bin/env python3
"""Compare aligned-SAM JointInit with box and shifted-mask causal controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = {
    "ap": 0,
    "ap50": 1,
    "ap75": 2,
    "aps": 3,
    "apm": 4,
    "apl": 5,
    "ar100": 8,
    "ars": 9,
    "arm": 10,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a00-log", type=Path, required=True)
    parser.add_argument("--aligned-log", type=Path, required=True)
    parser.add_argument("--box-log", type=Path, required=True)
    parser.add_argument("--shift-log", type=Path, required=True)
    parser.add_argument("--aligned-validation", type=Path)
    parser.add_argument("--box-validation", type=Path)
    parser.add_argument("--shift-validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if "epoch" in row and "test_coco_eval_bbox" in row]
    by_epoch = {int(row["epoch"]): row for row in rows}
    if not by_epoch:
        raise RuntimeError(f"no validation rows in {path}")
    return by_epoch


def metric_row(row):
    coco = row["test_coco_eval_bbox"]
    result = {name: float(coco[index]) for name, index in METRICS.items()}
    result["epoch"] = int(row["epoch"])
    result["loss"] = float(row.get("train_loss", float("nan")))
    result["loss_spar"] = float(row.get("train_loss_spar", 0.0))
    return result


def best_ap(rows):
    return metric_row(max(rows.values(), key=lambda row: float(row["test_coco_eval_bbox"][0])))


def mean_metrics(rows, epochs):
    selected = [metric_row(rows[epoch]) for epoch in epochs]
    return {
        name: sum(row[name] for row in selected) / len(selected)
        for name in ("ap", "ap75", "aps", "apm", "ar100")
    }


def delta(left, right):
    return {name: left[name] - right[name] for name in ("ap", "ap75", "aps", "apm", "ar100")}


def read_optional(path):
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def main():
    args = parse_args()
    groups = {
        "a00": read_rows(args.a00_log),
        "aligned_sam": read_rows(args.aligned_log),
        "box_mask": read_rows(args.box_log),
        "shift_mask": read_rows(args.shift_log),
    }
    required_epochs = set(range(60))
    for name, rows in groups.items():
        missing = sorted(required_epochs - set(rows))
        if missing:
            raise RuntimeError(f"{name} is missing epochs: {missing}")

    best = {name: best_ap(rows) for name, rows in groups.items()}
    tail = {name: mean_metrics(rows, range(50, 60)) for name, rows in groups.items()}
    aligned_epoch = best["aligned_sam"]["epoch"]
    at_aligned_epoch = {
        name: metric_row(rows[aligned_epoch]) for name, rows in groups.items()
    }

    best_deltas = {
        "aligned_minus_a00": delta(best["aligned_sam"], best["a00"]),
        "aligned_minus_box": delta(best["aligned_sam"], best["box_mask"]),
        "aligned_minus_shift": delta(best["aligned_sam"], best["shift_mask"]),
    }
    same_epoch_deltas = {
        "aligned_minus_a00": delta(at_aligned_epoch["aligned_sam"], at_aligned_epoch["a00"]),
        "aligned_minus_box": delta(at_aligned_epoch["aligned_sam"], at_aligned_epoch["box_mask"]),
        "aligned_minus_shift": delta(at_aligned_epoch["aligned_sam"], at_aligned_epoch["shift_mask"]),
    }
    tail_deltas = {
        "aligned_minus_a00": delta(tail["aligned_sam"], tail["a00"]),
        "aligned_minus_box": delta(tail["aligned_sam"], tail["box_mask"]),
        "aligned_minus_shift": delta(tail["aligned_sam"], tail["shift_mask"]),
    }

    contour_gate = all(
        best_deltas[key][metric] > 0 and tail_deltas[key][metric] > 0
        for key in ("aligned_minus_box", "aligned_minus_shift")
        for metric in ("ap75", "aps")
    )
    report = {
        "protocol": "same COCO tuning checkpoint, seed0, batch32, 60 epochs and JointInit schedule; only SPAR target differs",
        "best": best,
        "aligned_best_epoch": aligned_epoch,
        "at_aligned_best_epoch": at_aligned_epoch,
        "last10_mean": tail,
        "best_to_best_delta": best_deltas,
        "same_epoch_delta": same_epoch_deltas,
        "last10_mean_delta": tail_deltas,
        "target_preference_validation": {
            "aligned_sam": read_optional(args.aligned_validation),
            "box_mask": read_optional(args.box_validation),
            "shift_mask": read_optional(args.shift_validation),
        },
        "gates": {
            "aligned_beats_box_and_shift_on_ap75_and_aps_best_and_tail": contour_gate,
            "enter_sabr": contour_gate,
        },
        "interpretation": (
            "PRECISE_SAM_CONTOUR_SUPPORTED"
            if contour_gate
            else "PRECISE_SAM_CONTOUR_NOT_YET_SUPPORTED"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

