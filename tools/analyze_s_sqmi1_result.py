#!/usr/bin/env python3
"""Summarize S-QMI1 curves and inspect the learned initialization gate."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aux-run",
        type=Path,
        default=WORKSPACE / "outputs/S_SQMI1_AUX_C_GQ1_B16A2_12E_TESTDEV/seed0",
    )
    parser.add_argument(
        "--init-run",
        type=Path,
        default=WORKSPACE / "outputs/S_SQMI1_INIT_C_GQ1_B16A2_12E_TESTDEV/seed0",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_s/s_sqmi1_init_c_gq1_b16a2_12e_testdev_local.yml",
    )
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "reports/145_sqmi1_result/analysis.json",
    )
    return parser.parse_args()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics(row):
    values = row["test_coco_eval_bbox"]
    return {
        "epoch": row["epoch"],
        "ap": values[0],
        "ap50": values[1],
        "ap75": values[2],
        "aps": values[3],
        "apm": values[4],
        "ar100": values[8],
        "loss_sqmi": row.get("train_loss_sqmi"),
    }


def checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"]
    return checkpoint.get("model", checkpoint)


def inspect_checkpoint(config, checkpoint, batches):
    cfg = YAMLConfig(str(config))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    model = model.cuda().eval()

    total_queries = 0
    valid_queries = 0
    sums = {
        "mix": 0.0,
        "valid_mix": 0.0,
        "effective_delta_abs": 0.0,
        "certainty": 0.0,
        "area_ratio": 0.0,
    }
    maxima = {"mix": 0.0, "effective_delta_abs": 0.0}
    with torch.inference_mode():
        for index, (samples, _targets) in enumerate(cfg.val_dataloader):
            if index >= batches:
                break
            model(samples.cuda(non_blocking=True))
            diag = model.decoder.last_sqmi_diagnostics
            valid = diag["valid_mask"]
            count = valid.numel()
            valid_count = int(valid.sum())
            total_queries += count
            valid_queries += valid_count
            mix = diag["mix"].squeeze(-1)
            effective = diag["effective_box_delta"].abs()
            sums["mix"] += float(mix.sum())
            sums["valid_mix"] += float(mix[valid].sum()) if valid_count else 0.0
            sums["effective_delta_abs"] += float(effective.sum())
            sums["certainty"] += float(diag["certainty"].sum())
            sums["area_ratio"] += float(diag["area_ratio"].sum())
            maxima["mix"] = max(maxima["mix"], float(mix.max()))
            maxima["effective_delta_abs"] = max(
                maxima["effective_delta_abs"], float(effective.max())
            )

    result = {
        "checkpoint": str(checkpoint),
        "batches": batches,
        "queries": total_queries,
        "valid_query_ratio": valid_queries / total_queries,
        "mean_mix_all": sums["mix"] / total_queries,
        "mean_mix_valid": sums["valid_mix"] / max(valid_queries, 1),
        "max_mix": maxima["mix"],
        "mean_abs_effective_box_delta": sums["effective_delta_abs"] / (total_queries * 4),
        "max_abs_effective_box_delta": maxima["effective_delta_abs"],
        "mean_certainty": sums["certainty"] / total_queries,
        "mean_mask_area_ratio": sums["area_ratio"] / total_queries,
    }
    del model
    torch.cuda.empty_cache()
    return result


def mean_metric(curve, key):
    return sum(row[key] for row in curve) / len(curve)


def main():
    args = parse_args()
    aux_curve = [metrics(row) for row in rows(args.aux_run / "log.txt")]
    init_curve = [metrics(row) for row in rows(args.init_run / "log.txt")]
    if len(aux_curve) != 12 or len(init_curve) != 12:
        raise RuntimeError("Expected two complete 12-epoch curves")
    aux_best = max(aux_curve, key=lambda row: row["ap"])
    init_best = max(init_curve, key=lambda row: row["ap"])
    paired = [
        {
            "epoch": left["epoch"],
            "init_minus_aux_ap": right["ap"] - left["ap"],
            "init_minus_aux_ap75": right["ap75"] - left["ap75"],
            "init_minus_aux_aps": right["aps"] - left["aps"],
        }
        for left, right in zip(aux_curve, init_curve)
    ]
    result = {
        "status": "complete",
        "aux_curve": aux_curve,
        "init_curve": init_curve,
        "aux_best": aux_best,
        "init_best": init_best,
        "best_ap_delta": init_best["ap"] - aux_best["ap"],
        "same_epoch0_delta": paired[0],
        "paired_epoch_deltas": paired,
        "mean_12_delta": {
            key: mean_metric(init_curve, key) - mean_metric(aux_curve, key)
            for key in ("ap", "ap50", "ap75", "aps", "apm", "ar100")
        },
        "init_wins_ap_epochs": sum(item["init_minus_aux_ap"] > 0 for item in paired),
        "gate_best": inspect_checkpoint(args.config, args.init_run / "best_stg1.pth", args.batches),
        "gate_last": inspect_checkpoint(args.config, args.init_run / "last.pth", args.batches),
    }
    for section in ("aux_best", "init_best", "best_ap_delta"):
        value = result[section]
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"Non-finite summary value: {section}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
