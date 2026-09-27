#!/usr/bin/env python3
"""Aggregate strictly paired REP1.1-JointInit results across three seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


CORE_METRICS = ("ap", "ap50", "ap75", "aps", "apm", "ar100")
TAIL_METRICS = ("ap", "ap75", "aps", "apm", "ar100")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-summary", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean_std(values):
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def main():
    args = parse_args()
    if len(args.seed_summary) != 3:
        raise RuntimeError(
            f"exactly three seed summaries are required, got {len(args.seed_summary)}"
        )

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.seed_summary]
    seeds = []
    for path, report in zip(args.seed_summary, reports):
        seed_text = path.stem
        seed = int("".join(ch for ch in seed_text if ch.isdigit()))
        seeds.append(
            {
                "seed": seed,
                "source": str(path.resolve()),
                "control_best": report["control_best"],
                "jointinit_best": report["progressive_best"],
                "best_delta": report["best_to_best_delta"],
                "tail_delta": report["last10_mean_delta"],
            }
        )
    seeds.sort(key=lambda item: item["seed"])

    absolute = {
        metric: mean_std([item["jointinit_best"][metric] for item in seeds])
        for metric in CORE_METRICS
    }
    best_delta = {
        metric: mean_std([item["best_delta"][metric] for item in seeds])
        for metric in CORE_METRICS
    }
    tail_delta = {
        metric: mean_std([item["tail_delta"][metric] for item in seeds])
        for metric in TAIL_METRICS
    }
    positive_seed_counts = {
        metric: sum(item["best_delta"][metric] > 0.0 for item in seeds)
        for metric in ("ap", "ap75", "aps")
    }

    # Narrative-specific gates frozen before seed1/seed2 results are observed.
    # AP remains a safety metric; the primary claims concern strict localization
    # and small targets, which are measured by AP75 and APS.
    gates = {
        "mean_best_ap_nonnegative": best_delta["ap"]["mean"] >= 0.0,
        "mean_best_ap75_gain_at_least_0_005": best_delta["ap75"]["mean"] >= 0.005,
        "mean_best_aps_gain_at_least_0_004": best_delta["aps"]["mean"] >= 0.004,
        "ap75_positive_in_at_least_2_of_3_seeds": positive_seed_counts["ap75"] >= 2,
        "aps_positive_in_at_least_2_of_3_seeds": positive_seed_counts["aps"] >= 2,
        "mean_tail_ap75_positive": tail_delta["ap75"]["mean"] > 0.0,
        "mean_tail_aps_positive": tail_delta["aps"]["mean"] > 0.0,
    }

    output = {
        "protocol": (
            "three paired seeds; each A00 and JointInit pair shares COCO tuning "
            "checkpoint, seed, batch32, 60 epochs, AMP init scale 1024 and data order"
        ),
        "selection_target": "strict localization and small-target accuracy",
        "seeds": seeds,
        "jointinit_absolute": absolute,
        "paired_best_delta": best_delta,
        "paired_last10_delta": tail_delta,
        "positive_seed_counts": positive_seed_counts,
        "gates": gates,
        "all_narrative_gates_pass": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
