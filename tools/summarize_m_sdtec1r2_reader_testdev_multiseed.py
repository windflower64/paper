"""汇总M-SDTEC1-R2冻结读取器的严格三随机种子复核结果。

主指标使用每个训练日志所选best权重的独立normal复评结果；训练日志仅用于
核对权重选择轮次、完整训练曲线和最后10轮稳定性。这样可以避免控制台四舍
五入或checkpoint选择错误影响最终判断。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


METRIC_NAMES = ("AP", "AP50", "AP75", "APS", "APM", "APL")


def read_curve(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        metrics = record.get("test_coco_eval_bbox")
        if not isinstance(metrics, list) or len(metrics) < 9:
            raise ValueError(f"{path}: 第{line_number}行缺少完整COCO指标")
        rows.append(
            {
                "epoch": int(record["epoch"]),
                **{
                    name: float(value)
                    for name, value in zip(METRIC_NAMES, metrics[:6])
                },
                "AR100": float(metrics[8]),
                "train_loss": float(record["train_loss"]),
            }
        )
    if len(rows) != 30 or [row["epoch"] for row in rows] != list(range(30)):
        raise ValueError(f"{path}: 必须包含连续的epoch 0-29，共30行")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def stats(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--causal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgb-baseline-ap", type=float, default=0.5241381956306778)
    parser.add_argument("--mean-gain-threshold", type=float, default=0.0010)
    parser.add_argument("--permutation-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    seeds: list[dict] = []
    for seed in (0, 1, 2):
        run_dir = args.training_root / f"seed{seed}"
        curve = read_curve(run_dir / "log.txt")
        logged_best = max(curve, key=lambda row: row["AP"])
        checkpoint = run_dir / "best_stg1.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        causal_path = args.causal_root / f"seed{seed}" / "summary.json"
        causal = json.loads(causal_path.read_text(encoding="utf-8"))
        for mode in ("normal", "zero", "feature_permute"):
            if mode not in causal.get("results", {}):
                raise ValueError(f"{causal_path}: 缺少{mode}复评")
        normal = causal["results"]["normal"]
        zero = causal["results"]["zero"]
        permuted = causal["results"]["feature_permute"]
        last10 = curve[-10:]
        independent_ap = float(normal["AP"])
        normal_minus_zero = independent_ap - float(zero["AP"])
        permutation_gap = abs(independent_ap - float(permuted["AP"]))
        batch_shuffle = causal["results"].get("batch_shuffle")
        global_mismatch = causal["results"].get("global_mismatch")
        normal_minus_batch_shuffle = (
            independent_ap - float(batch_shuffle["AP"])
            if batch_shuffle is not None
            else None
        )
        normal_minus_global_mismatch = (
            independent_ap - float(global_mismatch["AP"])
            if global_mismatch is not None
            else None
        )
        seeds.append(
            {
                "seed": seed,
                "training_log": str((run_dir / "log.txt").resolve()),
                "best_checkpoint": str(checkpoint.resolve()),
                "best_checkpoint_sha256": sha256(checkpoint),
                "logged_best": logged_best,
                "independent_normal": {
                    name: float(normal[name]) for name in METRIC_NAMES
                },
                "independent_minus_logged_best_AP": independent_ap
                - logged_best["AP"],
                "delta_AP_vs_fixed_rgb_baseline": independent_ap
                - args.rgb_baseline_ap,
                "positive_vs_fixed_rgb_baseline": independent_ap
                > args.rgb_baseline_ap,
                "last10": {
                    "epochs": "20-29",
                    "mean_AP": statistics.fmean(row["AP"] for row in last10),
                    "population_std_AP": statistics.pstdev(
                        row["AP"] for row in last10
                    ),
                    "mean_AP75": statistics.fmean(
                        row["AP75"] for row in last10
                    ),
                    "mean_APS": statistics.fmean(row["APS"] for row in last10),
                    "mean_APM": statistics.fmean(row["APM"] for row in last10),
                },
                "causal": {
                    "normal_minus_zero_AP": normal_minus_zero,
                    "thermal_input_effect_pass": normal_minus_zero > 0.0,
                    "feature_permute_abs_AP_gap": permutation_gap,
                    "coordinate_free_metric_pass": permutation_gap
                    <= args.permutation_tolerance,
                    "normal_minus_batch_shuffle_AP": normal_minus_batch_shuffle,
                    "normal_minus_global_mismatch_AP": normal_minus_global_mismatch,
                    "paired_thermal_content_pass": (
                        normal_minus_global_mismatch > 0.0
                        if normal_minus_global_mismatch is not None
                        else None
                    ),
                },
            }
        )

    aps = [seed["independent_normal"]["AP"] for seed in seeds]
    gains = [seed["delta_AP_vs_fixed_rgb_baseline"] for seed in seeds]
    positive_count = sum(seed["positive_vs_fixed_rgb_baseline"] for seed in seeds)
    mean_gain = statistics.fmean(gains)
    performance_pass = (
        mean_gain >= args.mean_gain_threshold and positive_count >= 2
    )
    thermal_effect_pass = all(
        seed["causal"]["thermal_input_effect_pass"] for seed in seeds
    )
    coordinate_free_pass = all(
        seed["causal"]["coordinate_free_metric_pass"] for seed in seeds
    )
    paired_content_results = [
        seed["causal"]["normal_minus_global_mismatch_AP"] for seed in seeds
    ]
    paired_content_values = [
        value for value in paired_content_results if value is not None
    ]
    paired_content_all_pass = len(paired_content_values) == 3 and all(
        value > 0.0 for value in paired_content_values
    )
    overall_pass = performance_pass and thermal_effect_pass and coordinate_free_pass

    aggregate_metrics = {}
    for metric in ("AP", "AP75", "APS", "APM"):
        aggregate_metrics[metric] = stats(
            [seed["independent_normal"][metric] for seed in seeds]
        )

    result = {
        "protocol": "strict_three_seed_original_test_as_development_validation",
        "warning": (
            "原test已参与逐轮best选择；这些结果是开发集复核，不能再称为未调参最终测试。"
        ),
        "fixed_rgb_baseline_AP": args.rgb_baseline_ap,
        "thresholds": {
            "mean_AP_gain_at_least": args.mean_gain_threshold,
            "positive_seed_count_at_least": 2,
            "stable_positive_claim_requires": "3/3 seeds positive",
            "normal_minus_zero_AP_each_seed": "> 0",
            "feature_permute_abs_AP_gap_each_seed_at_most": args.permutation_tolerance,
        },
        "seeds": seeds,
        "aggregate": {
            "metrics": aggregate_metrics,
            "AP_gain_vs_fixed_rgb": stats(gains),
            "positive_seed_count": positive_count,
            "all_three_seeds_positive": positive_count == 3,
            "mean_gain_threshold_pass": mean_gain >= args.mean_gain_threshold,
            "at_least_two_positive_pass": positive_count >= 2,
            "performance_pass": performance_pass,
            "thermal_input_effect_all_seeds_pass": thermal_effect_pass,
            "coordinate_free_all_seeds_pass": coordinate_free_pass,
            "normal_minus_global_mismatch_AP": (
                stats(paired_content_values)
                if len(paired_content_values) == 3
                else None
            ),
            "paired_thermal_content_all_seeds_pass": paired_content_all_pass,
            "overall_pass": overall_pass,
        },
        "verdict": (
            "通过：M1冻结读取器满足预注册的三种子性能与因果门槛"
            if overall_pass
            else "未通过：停止把M1冻结读取器作为性能主模块继续调参"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
