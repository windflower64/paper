#!/usr/bin/env python3
"""汇总S-HRBR1 seed0/1/2的同进程配对结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("AP", "AP50", "AP75", "APS", "APM", "AR100")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for seed, path in enumerate(args.summaries):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload["best_validation"]
        rows.append(
            {
                "seed": seed,
                "best_epoch": payload["best_epoch"],
                "absolute": {key: validation[key] for key in METRICS},
                "paired_delta": {key: validation["delta"][key] for key in METRICS},
                "paired_baseline": {
                    key: validation["baseline"][key] for key in METRICS
                },
            }
        )

    aggregate = {}
    for family in ("absolute", "paired_delta"):
        aggregate[family] = {}
        for metric in METRICS:
            values = np.asarray([row[family][metric] for row in rows], dtype=np.float64)
            aggregate[family][metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
                "positive_seeds": int((values > 0).sum()) if family == "paired_delta" else None,
            }

    ap_delta = aggregate["paired_delta"]["AP"]
    ap75_delta = aggregate["paired_delta"]["AP75"]
    aps_delta = aggregate["paired_delta"]["APS"]
    gate = {
        "all_three_ap_positive": ap_delta["positive_seeds"] == 3,
        "mean_ap_gain_at_least_0_001": ap_delta["mean"] >= 0.001,
        "mean_ap75_positive": ap75_delta["mean"] > 0,
        "mean_aps_positive": aps_delta["mean"] > 0,
    }
    gate["pass"] = all(gate.values())
    report = {
        "experiment": "S-HRBR1-MULTISEED",
        "seeds": rows,
        "aggregate": aggregate,
        "gate": gate,
        "interpretation": (
            "通过：可把HRBR1作为稳定性能模块，仍不能把公开RefineBox适配单独包装成核心创新。"
            if gate["pass"]
            else "未通过：seed0正增益不能视为稳定，HRBR1降级为探索性结果。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

