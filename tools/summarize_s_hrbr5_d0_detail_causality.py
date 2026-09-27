#!/usr/bin/env python3
"""汇总 S-HRBR5-D0 三组因果对照，并按预注册门槛裁决。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODES = ("full", "lowpass", "shifted_detail")
METRICS = ("AP", "AP50", "AP75", "APS", "APM", "AR100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--lowpass", type=Path, required=True)
    parser.add_argument("--shifted-detail", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_arm(path: Path) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    history_path = Path(summary["history_path"])
    history = json.loads(history_path.read_text(encoding="utf-8"))
    best = max(history, key=lambda row: row["validation"]["AP"])
    last_four = history[-4:]
    return {
        "summary_path": str(path.resolve()),
        "best_epoch": int(best["epoch"]),
        "best": {metric: float(best["validation"][metric]) for metric in METRICS},
        "best_paired_delta": {
            metric: float(best["validation"]["delta"][metric]) for metric in METRICS
        },
        "last_four_epochs": [int(row["epoch"]) for row in last_four],
        "last_four_mean": {
            metric: sum(float(row["validation"][metric]) for row in last_four)
            / len(last_four)
            for metric in METRICS
        },
        "last_train_loss": float(history[-1]["train_loss"]),
    }


def subtract(left: dict, right: dict) -> dict:
    return {metric: left[metric] - right[metric] for metric in METRICS}


def main() -> None:
    args = parse_args()
    paths = {
        "full": args.full,
        "lowpass": args.lowpass,
        "shifted_detail": args.shifted_detail,
    }
    arms = {mode: load_arm(paths[mode]) for mode in MODES}
    comparisons = {}
    for control in ("lowpass", "shifted_detail"):
        comparisons[f"full_minus_{control}"] = {
            "best": subtract(arms["full"]["best"], arms[control]["best"]),
            "last_four_mean": subtract(
                arms["full"]["last_four_mean"], arms[control]["last_four_mean"]
            ),
        }

    full_low = comparisons["full_minus_lowpass"]
    full_shift = comparisons["full_minus_shifted_detail"]
    gate = {
        "full_best_ap_over_lowpass_at_least_0_001": full_low["best"]["AP"] >= 0.001,
        "full_best_ap_over_shifted_at_least_0_001": full_shift["best"]["AP"] >= 0.001,
        "full_best_ap75_over_both": min(
            full_low["best"]["AP75"], full_shift["best"]["AP75"]
        ) > 0,
        "full_best_aps_over_both": min(
            full_low["best"]["APS"], full_shift["best"]["APS"]
        ) > 0,
        "full_last_four_ap_over_both": min(
            full_low["last_four_mean"]["AP"],
            full_shift["last_four_mean"]["AP"],
        ) > 0,
    }
    gate["pass"] = all(gate.values())
    report = {
        "experiment": "S-HRBR5-D0-DETAIL-CAUSALITY",
        "arms": arms,
        "comparisons": comparisons,
        "preregistered_gate": gate,
        "interpretation": (
            "通过：HRBR1 的收益依赖位置正确的高分辨率细节，可进入 HRBR5-D1 专属模块设计。"
            if gate["pass"]
            else "未通过：现有证据不能把 HRBR1 增益归因于下采样丢失的细节，停止用该叙事包装并回到 HRBR1 稳定版本。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
