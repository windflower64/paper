#!/usr/bin/env python3
"""Summarize paired REP1.1 histories before and after exact-state resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-base-log", type=Path, required=True)
    parser.add_argument("--control-resume-log", type=Path, required=True)
    parser.add_argument("--spar-base-log", type=Path, required=True)
    parser.add_argument("--spar-resume-log", type=Path, required=True)
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
                    "spar_loss": (
                        float(row["train_loss_spar"])
                        if "train_loss_spar" in row
                        else None
                    ),
                }
            )
    if not rows:
        raise RuntimeError(f"no validation rows in {path}")
    return rows


def merge(base_path, resume_path):
    by_epoch = {
        row["epoch"]: row
        for row in read_rows(base_path) + read_rows(resume_path)
    }
    rows = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    expected = list(range(12))
    actual = [row["epoch"] for row in rows]
    if actual != expected:
        raise RuntimeError(f"expected epochs {expected}, got {actual}")
    return rows


def best(rows):
    return max(rows, key=lambda row: (row["ap"], row["ap75"]))


def delta(left, right):
    return {
        key: left[key] - right[key]
        for key in ("ap", "ap50", "ap75", "aps", "apm", "apl", "ar100")
    }


def main():
    args = parse_args()
    control = merge(args.control_base_log, args.control_resume_log)
    spar = merge(args.spar_base_log, args.spar_resume_log)
    paired = [
        {
            "epoch": control[index]["epoch"],
            "control": control[index],
            "spar": spar[index],
            "spar_minus_control": delta(spar[index], control[index]),
        }
        for index in range(len(control))
    ]
    control_best = best(control)
    spar_best = best(spar)
    report = {
        "control_best_0_to_11": control_best,
        "spar_best_0_to_11": spar_best,
        "spar_minus_control_best": delta(spar_best, control_best),
        "extension_best_control_6_to_11": best(control[6:]),
        "extension_best_spar_6_to_11": best(spar[6:]),
        "last_epoch": paired[-1],
        "spar_ap_wins": sum(item["spar_minus_control"]["ap"] > 0 for item in paired),
        "spar_aps_wins": sum(item["spar_minus_control"]["aps"] > 0 for item in paired),
        "mean_spar_minus_control": {
            key: sum(item["spar_minus_control"][key] for item in paired) / len(paired)
            for key in ("ap", "ap75", "aps", "apm", "ar100")
        },
        "paired_epochs": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
