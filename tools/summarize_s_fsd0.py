#!/usr/bin/env python3
"""汇总FSD0、重初始化控制和同权重频带干预。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("AP", "AP50", "AP75", "APS", "APM", "APL", "AR1", "AR10", "AR", "ARS", "ARM", "ARL")


def read_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        values = row.get("test_coco_eval_bbox")
        if isinstance(values, list) and len(values) >= 12:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"没有验证记录：{path}")
    return rows


def summarize_log(path):
    rows = read_rows(path)
    best = max(rows, key=lambda row: row["test_coco_eval_bbox"][0])
    values = best["test_coco_eval_bbox"]
    last = rows[-10:]
    return {
        "轮数": len(rows),
        "最佳epoch": int(best["epoch"]),
        "最佳指标": {name: float(values[index]) for index, name in enumerate(METRICS)},
        "最后10轮平均AP": sum(row["test_coco_eval_bbox"][0] for row in last) / len(last),
        "最后10轮平均AP75": sum(row["test_coco_eval_bbox"][2] for row in last) / len(last),
    }


def read_controls(directory):
    controls = {}
    for path in directory.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        mode = data.get("fsd_mode")
        metrics = data.get("custom_size_metrics", {})
        if mode and "all" in metrics:
            controls[mode] = {
                "文件": str(path),
                "总体": metrics["all"],
                "16到32像素": metrics.get("16to32"),
                "8到16像素": metrics.get("8to16"),
            }
    required = {"full", "ll_only", "shift_hf", "phase_permute", "spatial_only"}
    missing = sorted(required - set(controls))
    if missing:
        raise RuntimeError(f"固定best频带控制不完整：{missing}")
    return controls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsd-log", type=Path, required=True)
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--causal-dir", type=Path, required=True)
    parser.add_argument("--a00-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fsd = summarize_log(args.fsd_log)
    control = summarize_log(args.control_log)
    a00 = summarize_log(args.a00_log)
    causal = read_controls(args.causal_dir)
    full = causal["full"]["总体"]
    deltas = {}
    for mode in ("ll_only", "shift_hf", "phase_permute", "spatial_only"):
        other = causal[mode]["总体"]
        deltas[mode] = {
            "dAP": full["AP50_95"] - other["AP50_95"],
            "dAP75": full["AP75"] - other["AP75"],
            "dAR": full["AR100"] - other["AR100"],
        }

    fsd_ap = fsd["最佳指标"]["AP"]
    report = {
        "协议": "同一COCO检测器、seed0、batch16、60轮；FSD与标准重初始化控制仅替换S8到S16算子",
        "FSD0": fsd,
        "标准重初始化控制": control,
        "A00历史正式基线": a00,
        "FSD相对标准重初始化控制最佳AP": fsd_ap - control["最佳指标"]["AP"],
        "FSD相对A00最佳AP": fsd_ap - a00["最佳指标"]["AP"],
        "固定FSD最佳权重因果结果": causal,
        "FULL相对各干预": deltas,
        "预注册判定": {
            "结构优于重初始化控制至少0.20AP点": fsd_ap - control["最佳指标"]["AP"] >= 0.002,
            "绝对AP优于A00至少0.20AP点": fsd_ap - a00["最佳指标"]["AP"] >= 0.002,
            "FULL_AP75优于LL_ONLY至少0.20点": deltas["ll_only"]["dAP75"] >= 0.002,
            "FULL_AP75优于SHIFT_HF至少0.20点": deltas["shift_hf"]["dAP75"] >= 0.002,
            "FULL_AP75优于PHASE_PERMUTE至少0.10点": deltas["phase_permute"]["dAP75"] >= 0.001,
            "FULL_AP75优于SPATIAL_ONLY至少0.20点": deltas["spatial_only"]["dAP75"] >= 0.002,
        },
        "machine_decision": {
            "fsd_best_ap": fsd_ap,
            "control_best_ap": control["最佳指标"]["AP"],
            "a00_best_ap": a00["最佳指标"]["AP"],
            "fsd_minus_control_ap": fsd_ap - control["最佳指标"]["AP"],
            "fsd_minus_a00_ap": fsd_ap - a00["最佳指标"]["AP"],
        },
    }
    report["主线门槛全部通过"] = all(report["预注册判定"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
