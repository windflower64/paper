#!/usr/bin/env python3
"""汇总 FSD1 功能初始化实验，并与 FSD0、重初始化控制和 A00 比较。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from summarize_s_fsd0 import read_controls, summarize_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsd1-log", type=Path, required=True)
    parser.add_argument("--fsd0-log", type=Path, required=True)
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--a00-log", type=Path, required=True)
    parser.add_argument("--causal-dir", type=Path, required=True)
    parser.add_argument("--functional-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fsd1 = summarize_log(args.fsd1_log)
    fsd0 = summarize_log(args.fsd0_log)
    control = summarize_log(args.control_log)
    a00 = summarize_log(args.a00_log)
    causal = read_controls(args.causal_dir)
    functional = json.loads(args.functional_report.read_text(encoding="utf-8"))

    fsd1_ap = fsd1["最佳指标"]["AP"]
    report = {
        "协议": "FSD1仅增加固定3轮的下采样功能复现；随后与FSD0相同，seed0、batch16、60轮",
        "FSD1功能初始化": functional,
        "FSD1": fsd1,
        "FSD0": fsd0,
        "标准重初始化控制": control,
        "A00历史正式基线": a00,
        "FSD1相对FSD0最佳AP": fsd1_ap - fsd0["最佳指标"]["AP"],
        "FSD1相对标准重初始化控制最佳AP": fsd1_ap
        - control["最佳指标"]["AP"],
        "FSD1相对A00最佳AP": fsd1_ap - a00["最佳指标"]["AP"],
        "固定FSD1最佳权重因果结果": causal,
    }
    report["预注册判定"] = {
        "功能初始化带来至少0.20AP点": report["FSD1相对FSD0最佳AP"] >= 0.002,
        "FSD1不低于A00超过0.20AP点": report["FSD1相对A00最佳AP"] >= -0.002,
    }
    report["结论"] = (
        "保留FSD功能初始化路线"
        if all(report["预注册判定"].values())
        else "停止FSD直接替换路线，不再追加变体"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
