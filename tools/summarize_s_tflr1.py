#!/usr/bin/env python3
"""汇总TFLR1正式训练和同权重频率因果控制。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from summarize_s_fsd0 import summarize_log


def read_controls(directory):
    controls = {}
    for path in directory.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mode = data.get("tflr_mode")
        metrics = data.get("custom_size_metrics", {})
        if mode and "all" in metrics:
            controls[mode] = {
                "文件": str(path),
                "总体": metrics["all"],
                "8到16像素": metrics.get("8to16"),
                "16到32像素": metrics.get("16to32"),
            }
    required = {"full", "zero", "shifted", "swap_direction"}
    missing = sorted(required - set(controls))
    if missing:
        raise RuntimeError(f"TFLR固定best控制不完整：{missing}")
    return controls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--a00-log", type=Path, required=True)
    parser.add_argument("--causal-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--local-causality", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tflr = summarize_log(args.train_log)
    a00 = summarize_log(args.a00_log)
    controls = read_controls(args.causal_dir)
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    local_causality = json.loads(args.local_causality.read_text(encoding="utf-8"))
    full = controls["full"]["总体"]

    causal_delta = {}
    for mode in ("zero", "shifted", "swap_direction"):
        other = controls[mode]["总体"]
        causal_delta[mode] = {
            "dAP": full["AP50_95"] - other["AP50_95"],
            "dAP75": full["AP75"] - other["AP75"],
            "dAR100": full["AR100"] - other["AR100"],
        }

    tflr_ap = tflr["最佳指标"]["AP"]
    tflr_ap75 = tflr["最佳指标"]["AP75"]
    tflr_aps = tflr["最佳指标"]["APS"]
    report = {
        "协议": "标准A00主路不变；TFLR只读取S8方向高频并修正最终框；seed0、batch16、60轮",
        "TFLR1": tflr,
        "A00历史正式基线": a00,
        "TFLR1相对A00": {
            "dAP": tflr_ap - a00["最佳指标"]["AP"],
            "dAP75": tflr_ap75 - a00["最佳指标"]["AP75"],
            "dAPS": tflr_aps - a00["最佳指标"]["APS"],
            "dAR100": tflr["最佳指标"]["AR"] - a00["最佳指标"]["AR"],
        },
        "固定best因果控制": controls,
        "FULL相对控制": causal_delta,
        "工程预检": preflight,
        "FSD目标局部因果前提": local_causality,
    }
    absolute = report["TFLR1相对A00"]
    report["预注册判定"] = {
        "绝对AP至少提高0.20点": absolute["dAP"] >= 0.002,
        "绝对AP75至少提高0.20点": absolute["dAP75"] >= 0.002,
        "绝对APS至少提高0.20点": absolute["dAPS"] >= 0.002,
        "FULL相对ZERO_AP至少提高0.10点": causal_delta["zero"]["dAP"] >= 0.001,
        "FULL相对ZERO_AP75至少提高0.20点": causal_delta["zero"]["dAP75"] >= 0.002,
        "FULL_AP75高于错位": causal_delta["shifted"]["dAP75"] >= 0.001,
        "FULL_AP75高于方向交换": causal_delta["swap_direction"]["dAP75"] >= 0.001,
    }
    report["主线门槛全部通过"] = all(report["预注册判定"].values())
    report["结论"] = (
        "TFLR1可进入复核"
        if report["主线门槛全部通过"]
        else "TFLR1关闭，不追加宽度、幅度、采样点或训练轮数搜索"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
