#!/usr/bin/env python3
"""汇总FSD0目标区域与等能量背景高频删除实验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    full_data = load(args.full)
    target_data = load(args.target)
    background_data = load(args.background)
    full = full_data["custom_size_metrics"]["all"]
    target = target_data["custom_size_metrics"]["all"]
    background = background_data["custom_size_metrics"]["all"]

    target_drop = {
        "dAP": full["AP50_95"] - target["AP50_95"],
        "dAP75": full["AP75"] - target["AP75"],
        "dAR100": full["AR100"] - target["AR100"],
    }
    background_drop = {
        "dAP": full["AP50_95"] - background["AP50_95"],
        "dAP75": full["AP75"] - background["AP75"],
        "dAR100": full["AR100"] - background["AR100"],
    }
    report = {
        "协议": "同一FSD0 epoch57 EMA；删除GT框加一圈S16单元内的高频，对照删除空间隔离且总能量匹配的最高能背景高频",
        "FULL": full_data,
        "目标邻域高频删除": target_data,
        "等能量背景高频删除": background_data,
        "目标删除相对FULL下降": target_drop,
        "背景删除相对FULL下降": background_drop,
        "目标特异额外下降": {
            key: target_drop[key] - background_drop[key]
            for key in target_drop
        },
        "能量匹配比例": background_data["fsd_local_intervention"][
            "removed_to_target_energy_ratio"
        ],
    }
    report["预注册判定"] = {
        "背景删除能量不少于目标": report["能量匹配比例"] >= 1.0,
        "目标特异AP伤害至少0.50点": report["目标特异额外下降"]["dAP"] >= 0.005,
        "目标特异AP75伤害至少1.00点": report["目标特异额外下降"]["dAP75"] >= 0.01,
    }
    report["允许进入定位专用联合支路"] = all(report["预注册判定"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
