"""Summarize the M-SDTEC1-R2 joint test-as-development run.

The report compares the complete JSON-lines training curve with the frozen
reader initialization, the RGB baseline and the causal validation of the joint
best checkpoint.  It also measures state-tensor drift by model branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch


METRIC_NAMES = ("AP", "AP50", "AP75", "APS", "APM", "APL")


def read_curve(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        metrics = row.get("test_coco_eval_bbox")
        if not isinstance(metrics, list) or len(metrics) < 6:
            raise ValueError(f"invalid COCO metrics at line {line_number}")
        rows.append(
            {
                "epoch": int(row["epoch"]),
                **{
                    name: float(value)
                    for name, value in zip(METRIC_NAMES, metrics[:6])
                },
                "AR100": float(metrics[8]),
                "train_loss": float(row["train_loss"]),
                "gate": float(row["train_sdtec_gate"]),
                "quality": float(row["train_sdtec_quality"]),
                "uncertainty": float(row["train_sdtec_uncertainty"]),
                "scale_0": float(row["train_sdtec_scale_0"]),
                "scale_1": float(row["train_sdtec_scale_1"]),
                "scale_2": float(row["train_sdtec_scale_2"]),
                "token_similarity": float(row["train_sdtec_token_similarity"]),
            }
        )
    if [row["epoch"] for row in rows] != list(range(len(rows))):
        raise ValueError("training log does not contain contiguous epochs from zero")
    return rows


def segment(rows: list[dict], start: int, end: int) -> dict:
    selected = [row for row in rows if start <= row["epoch"] <= end]
    values = [row["AP"] for row in selected]
    return {
        "epochs": f"{start}-{end}",
        "mean_AP": statistics.fmean(values),
        "population_std_AP": statistics.pstdev(values),
        "max_AP": max(values),
        "mean_AP75": statistics.fmean(row["AP75"] for row in selected),
        "mean_APS": statistics.fmean(row["APS"] for row in selected),
    }


def state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint["ema"]["module"]


def branch_name(key: str) -> str:
    if key.startswith("decoder.sdtec_"):
        return "sdtec"
    if key.startswith("thermal_backbone."):
        return "thermal_backbone"
    if key.startswith("thermal_encoder."):
        return "thermal_encoder"
    if key.startswith("backbone."):
        return "rgb_backbone"
    if key.startswith("encoder."):
        return "rgb_encoder"
    if key.startswith("decoder."):
        return "rgb_decoder"
    return "other"


def state_drift(initial_path: Path, best_path: Path, key_mode: str = "all") -> dict:
    initial = state_dict(initial_path)
    best = state_dict(best_path)
    if initial.keys() != best.keys():
        raise ValueError("checkpoint state dictionaries do not have identical keys")

    accumulators: dict[str, dict] = {}
    for key, initial_value in initial.items():
        is_running_stat = key.endswith(("running_mean", "running_var"))
        if key_mode == "running_stats" and not is_running_stat:
            continue
        if key_mode == "excluding_running_stats" and is_running_stat:
            continue
        best_value = best[key]
        if not (initial_value.is_floating_point() and best_value.is_floating_point()):
            continue
        group = branch_name(key)
        acc = accumulators.setdefault(
            group,
            {
                "tensor_count": 0,
                "element_count": 0,
                "initial_l2_squared": 0.0,
                "delta_l2_squared": 0.0,
                "delta_abs_sum": 0.0,
                "max_abs_delta": 0.0,
            },
        )
        delta = best_value.float() - initial_value.float()
        acc["tensor_count"] += 1
        acc["element_count"] += delta.numel()
        acc["initial_l2_squared"] += float(initial_value.float().square().sum())
        acc["delta_l2_squared"] += float(delta.square().sum())
        acc["delta_abs_sum"] += float(delta.abs().sum())
        acc["max_abs_delta"] = max(
            acc["max_abs_delta"], float(delta.abs().max())
        )

    report = {}
    for group, acc in accumulators.items():
        initial_l2 = math.sqrt(acc.pop("initial_l2_squared"))
        delta_l2 = math.sqrt(acc.pop("delta_l2_squared"))
        abs_sum = acc.pop("delta_abs_sum")
        report[group] = {
            **acc,
            "initial_l2": initial_l2,
            "delta_l2": delta_l2,
            "relative_l2_delta": delta_l2 / max(initial_l2, 1e-12),
            "mean_abs_delta": abs_sum / acc["element_count"],
        }
    return report


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--causal-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgb-baseline-ap", type=float, default=0.5241381956306778)
    parser.add_argument("--reader-initial-ap", type=float, default=0.5262229006504129)
    args = parser.parse_args()

    rows = read_curve(args.log)
    best = max(rows, key=lambda row: row["AP"])
    causal = json.loads(args.causal_summary.read_text(encoding="utf-8"))
    result = {
        "protocol": "original_test_used_as_development_validation",
        "log": {
            "path": str(args.log.resolve()),
            "size_bytes": args.log.stat().st_size,
            "epochs": len(rows),
            "schema_keys": sorted(rows[0]),
            "missing_epoch_count": 0,
        },
        "references": {
            "rgb_baseline_AP": args.rgb_baseline_ap,
            "frozen_reader_epoch9_AP": args.reader_initial_ap,
        },
        "joint_best": best,
        "joint_best_delta_AP_vs_rgb": best["AP"] - args.rgb_baseline_ap,
        "joint_best_delta_AP_vs_reader_initial": best["AP"] - args.reader_initial_ap,
        "epochs_above_rgb": sum(row["AP"] > args.rgb_baseline_ap for row in rows),
        "epochs_above_reader_initial": sum(
            row["AP"] > args.reader_initial_ap for row in rows
        ),
        "segments": [
            segment(rows, start, min(end, len(rows) - 1))
            for start, end in (
                [(0, 4), (5, 9)]
                + [(start, start + 9) for start in range(10, len(rows), 10)]
            )
            if start < len(rows)
        ],
        "causal_best": causal,
        "state_tensor_drift_initial_to_joint_best": state_drift(
            args.initial_checkpoint, args.best_checkpoint
        ),
        "normalization_running_stat_drift": state_drift(
            args.initial_checkpoint, args.best_checkpoint, "running_stats"
        ),
        "state_tensor_drift_excluding_running_stats": state_drift(
            args.initial_checkpoint, args.best_checkpoint, "excluding_running_stats"
        ),
        "best_checkpoint": {
            "path": str(args.best_checkpoint.resolve()),
            "sha256": sha256(args.best_checkpoint),
        },
        "verdict": (
            "joint_finetuning_failed_to_improve_frozen_reader_initialization"
            if best["AP"] <= args.reader_initial_ap
            else "joint_finetuning_improved_frozen_reader_initialization"
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
