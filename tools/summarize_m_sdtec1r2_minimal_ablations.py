"""汇总M1三个最小结构消融的严格三随机种子结果。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METRICS = ("AP", "AP50", "AP75", "APS", "APM", "APL")
VARIANTS = ("k1", "noreliability", "sameposition")
OUTPUT_NAMES = {
    "k1": "M_SDTEC1R2_ABLATION_K1_TESTDEV",
    "noreliability": "M_SDTEC1R2_ABLATION_NORELIABILITY_TESTDEV",
    "sameposition": "M_SDTEC1R2_ABLATION_SAMEPOSITION_TESTDEV",
}


def describe(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def read_curve(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        metrics = record.get("test_coco_eval_bbox")
        if not isinstance(metrics, list) or len(metrics) < 9:
            raise ValueError(f"{path}: 第{line_number}行COCO指标不完整")
        rows.append(
            {
                "epoch": int(record["epoch"]),
                **{
                    name: float(value)
                    for name, value in zip(METRICS, metrics[:6])
                },
                "AR100": float(metrics[8]),
            }
        )
    if len(rows) != 30 or [row["epoch"] for row in rows] != list(range(30)):
        raise ValueError(f"{path}: 训练日志必须连续覆盖epoch 0-29")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--causal-root", type=Path, required=True)
    parser.add_argument("--main-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--necessity-threshold", type=float, default=0.0010)
    parser.add_argument("--rgb-baseline-ap", type=float, default=0.5241381956306778)
    args = parser.parse_args()

    main_summary = json.loads(args.main_summary.read_text(encoding="utf-8"))
    main_by_seed = {
        int(row["seed"]): float(row["independent_normal"]["AP"])
        for row in main_summary["seeds"]
    }
    main_zero_minus_rgb = [
        float(row["independent_normal"]["AP"])
        - float(row["causal"]["normal_minus_zero_AP"])
        - args.rgb_baseline_ap
        for row in main_summary["seeds"]
    ]
    main_normal_minus_zero = [
        float(row["causal"]["normal_minus_zero_AP"])
        for row in main_summary["seeds"]
    ]
    main_normal_minus_mismatch = [
        float(row["causal"]["normal_minus_global_mismatch_AP"])
        for row in main_summary["seeds"]
    ]
    variants = {}
    for variant in VARIANTS:
        seeds = []
        for seed in (0, 1, 2):
            run_dir = args.outputs_root / OUTPUT_NAMES[variant] / f"seed{seed}"
            curve = read_curve(run_dir / "log.txt")
            logged_best = max(curve, key=lambda row: row["AP"])
            causal_path = args.causal_root / variant / f"seed{seed}" / "summary.json"
            causal = json.loads(causal_path.read_text(encoding="utf-8"))
            for mode in ("normal", "zero", "feature_permute", "global_mismatch"):
                if mode not in causal.get("results", {}):
                    raise ValueError(f"{causal_path}: 缺少{mode}复评")
            normal = causal["results"]["normal"]
            zero = causal["results"]["zero"]
            permuted = causal["results"]["feature_permute"]
            mismatch = causal["results"]["global_mismatch"]
            normal_ap = float(normal["AP"])
            last10 = curve[-10:]
            seeds.append(
                {
                    "seed": seed,
                    "logged_best": logged_best,
                    "independent_normal": {
                        name: float(normal[name]) for name in METRICS
                    },
                    "independent_minus_logged_best_AP": normal_ap
                    - logged_best["AP"],
                    "main_M1_AP": main_by_seed[seed],
                    "main_minus_ablation_AP": main_by_seed[seed] - normal_ap,
                    "ablation_delta_vs_rgb_AP": normal_ap - args.rgb_baseline_ap,
                    "last10_mean_AP": statistics.fmean(
                        row["AP"] for row in last10
                    ),
                    "causal": {
                        "normal_minus_zero_AP": normal_ap - float(zero["AP"]),
                        "zero_minus_rgb_baseline_AP": float(zero["AP"])
                        - args.rgb_baseline_ap,
                        "feature_permute_abs_AP_gap": abs(
                            normal_ap - float(permuted["AP"])
                        ),
                        "normal_minus_global_mismatch_AP": normal_ap
                        - float(mismatch["AP"]),
                    },
                }
            )

        paired = [row["main_minus_ablation_AP"] for row in seeds]
        aps = [row["independent_normal"]["AP"] for row in seeds]
        main_better_count = sum(value > 0.0 for value in paired)
        mean_gap_pass = statistics.fmean(paired) >= args.necessity_threshold
        direction_pass = main_better_count >= 2
        aggregate = {
            "AP": describe(aps),
            "main_minus_ablation_AP": describe(paired),
            "main_better_seed_count": main_better_count,
            "mean_gap_threshold_pass": mean_gap_pass,
            "at_least_two_matched_seeds_pass": direction_pass,
            "normal_minus_zero_AP": describe(
                [row["causal"]["normal_minus_zero_AP"] for row in seeds]
            ),
            "zero_minus_rgb_baseline_AP": describe(
                [row["causal"]["zero_minus_rgb_baseline_AP"] for row in seeds]
            ),
            "feature_permute_abs_AP_gap": describe(
                [row["causal"]["feature_permute_abs_AP_gap"] for row in seeds]
            ),
            "normal_minus_global_mismatch_AP": describe(
                [
                    row["causal"]["normal_minus_global_mismatch_AP"]
                    for row in seeds
                ]
            ),
            "structure_necessity_performance_pass": mean_gap_pass
            and direction_pass,
        }
        if variant == "sameposition":
            sensitivity_count = sum(
                row["causal"]["feature_permute_abs_AP_gap"] > 1e-6
                for row in seeds
            )
            aggregate["spatial_permutation_sensitive_seed_count"] = (
                sensitivity_count
            )
            aggregate["same_position_negative_control_pass"] = (
                aggregate["structure_necessity_performance_pass"]
                and sensitivity_count >= 2
            )
        variants[variant] = {"seeds": seeds, "aggregate": aggregate}

    k1_rows = variants["k1"]["seeds"]
    k1_causal_valid = (
        all(row["ablation_delta_vs_rgb_AP"] > 0.0 for row in k1_rows)
        and all(row["causal"]["normal_minus_zero_AP"] > 0.0 for row in k1_rows)
        and all(
            row["causal"]["normal_minus_global_mismatch_AP"] > 0.0
            for row in k1_rows
        )
        and all(
            row["causal"]["feature_permute_abs_AP_gap"] <= 1e-6
            for row in k1_rows
        )
    )
    no_reliability_rows = variants["noreliability"]["seeds"]
    fallback_tolerance = 0.0003
    main_safe_fallback = abs(statistics.fmean(main_zero_minus_rgb)) <= fallback_tolerance
    no_reliability_safe_fallback = abs(
        statistics.fmean(
            row["causal"]["zero_minus_rgb_baseline_AP"]
            for row in no_reliability_rows
        )
    ) <= fallback_tolerance
    reliability_causal_support = (
        main_safe_fallback
        and not no_reliability_safe_fallback
        and all(value > 0.0 for value in main_normal_minus_zero)
    )
    same_position_rows = variants["sameposition"]["seeds"]
    same_position_paired_positive_count = sum(
        row["causal"]["normal_minus_global_mismatch_AP"] > 0.0
        for row in same_position_rows
    )
    coordinate_free_causal_support = (
        main_summary["aggregate"]["coordinate_free_all_seeds_pass"]
        and main_summary["aggregate"]["paired_thermal_content_all_seeds_pass"]
        and variants["sameposition"]["aggregate"][
            "spatial_permutation_sensitive_seed_count"
        ]
        >= 2
        and same_position_paired_positive_count < 2
    )

    result = {
        "protocol": "M1_minimal_structure_ablations_three_seeds",
        "warning": "原test参与逐轮best选择，结果属于开发验证。",
        "necessity_threshold_AP": args.necessity_threshold,
        "fixed_rgb_baseline_AP": args.rgb_baseline_ap,
        "main_M1_AP": describe(list(main_by_seed.values())),
        "main_M1_causal_reference": {
            "normal_minus_zero_AP": describe(main_normal_minus_zero),
            "zero_minus_rgb_baseline_AP": describe(main_zero_minus_rgb),
            "normal_minus_global_mismatch_AP": describe(
                main_normal_minus_mismatch
            ),
        },
        "variants": variants,
        "decisions": {
            "multi_token_necessary": variants["k1"]["aggregate"][
                "structure_necessity_performance_pass"
            ],
            "explicit_reliability_performance_necessary": variants[
                "noreliability"
            ]["aggregate"]["structure_necessity_performance_pass"],
            "coordinate_free_choice_supported": variants["sameposition"][
                "aggregate"
            ]["same_position_negative_control_pass"],
            "K1_retains_valid_multimodal_causal_chain": k1_causal_valid,
            "explicit_reliability_causally_supported": reliability_causal_support,
            "coordinate_free_choice_causally_supported": coordinate_free_causal_support,
            "same_position_paired_positive_seed_count": same_position_paired_positive_count,
            "recommended_structure": (
                "K1_coordinate_free_with_explicit_reliability"
                if (
                    k1_causal_valid
                    and reliability_causal_support
                    and coordinate_free_causal_support
                )
                else "no_final_recommendation"
            ),
        },
        "fallback_AP_tolerance": fallback_tolerance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
