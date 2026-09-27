#!/usr/bin/env python3
"""Create a compact decision table from the spatial causal diagnostics."""

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(result):
    all_metrics = result["custom_size_metrics"]["all"]
    small = result["custom_size_metrics"]["16to32"]
    return all_metrics["AP50_95"], all_metrics["AP75"], small["AP50_95"], small["AP75"]


def fmt(value):
    return "—" if value is None else f"{value:.5f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    causal = {}
    for mode in ("learned", "uniform", "shuffled"):
        path = args.run_dir / "eval" / f"custom_metrics_and_importance_ema_{mode}.json"
        causal[mode] = load(path)
    learned_values = metrics(causal["learned"])

    lines = [
        "# 空间重要性机制诊断（epoch 43权重）",
        "",
        "## 1. LAD因果权重干预",
        "",
        "| 模式 | AP | AP75 | 16–32 AP | 16–32 AP75 | 相对Learned AP | 相对Learned AP75 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("learned", "uniform", "shuffled"):
        values = metrics(causal[mode])
        lines.append(
            f"| {mode} | {fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])} | "
            f"{fmt(values[3])} | {values[0]-learned_values[0]:+.5f} | "
            f"{values[1]-learned_values[1]:+.5f} |"
        )
    internal = causal["learned"].get("importance", {}).get("lad_internal", {})
    lines += [
        "",
        "Learned内部统计：",
        "",
        f"- 归一化四相位熵：{fmt(internal.get('normalized_phase_entropy'))}",
        f"- 候选相位方差：{fmt(internal.get('candidate_phase_variance'))}",
        f"- 相对均匀权重平均绝对偏差：{fmt(internal.get('mean_abs_weight_deviation_from_uniform'))}",
        "",
        "## 2. S-BUDGET空间预算",
        "",
        "| 保留比例 | 模式 | AP | AP75 | 16–32 AP | 16–32 AP75 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    baseline_path = args.run_dir / "budget" / "custom_metrics_and_importance_ema_learned.json"
    baseline = load(baseline_path)
    base_values = metrics(baseline)
    lines.append(
        f"| 100% | baseline | {fmt(base_values[0])} | {fmt(base_values[1])} | "
        f"{fmt(base_values[2])} | {fmt(base_values[3])} |"
    )
    for pct in (5, 10, 20, 30):
        for mode in ("learned", "random"):
            path = args.run_dir / "budget" / (
                f"custom_metrics_and_importance_ema_learned_budget{pct}_{mode}.json"
            )
            values = metrics(load(path))
            lines.append(
                f"| {pct}% | {mode} | {fmt(values[0])} | {fmt(values[1])} | "
                f"{fmt(values[2])} | {fmt(values[3])} |"
            )

    lines += ["", "## 3. 梯度审计", ""]
    for label, filename, key in (
        ("S-LAD1", "lad_epoch43.json", "lad_mean"),
        ("S-AUX1", "s_aux_best.json", "s_aux_mean"),
    ):
        data = load(args.run_dir / "gradient" / filename)
        lines += [f"### {label}", "", "```json", json.dumps(data.get(key), ensure_ascii=False, indent=2), "```", ""]
    lines += [
        "## 4. 裁决提示",
        "",
        "- Learned优于Uniform：四相位自适应选择有用。",
        "- Learned优于Shuffled：正确空间对应关系有用。",
        "- Importance预算优于等面积随机预算：预测空间位置具有检测因果价值。",
        "- 若预算有效而LAD无效，应停止照搬LAD，改为直接由Importance控制细节保护路径。",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
