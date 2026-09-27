#!/usr/bin/env python3
"""Validate the frozen 60-epoch QRL-R1 best checkpoint and causal controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from src.solver.det_engine import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "experiments/phase_s/s_bpc2_qrl_r1_detail_only_s8_s16_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT.parent
        / "runs/20_spatial_importance/S_BPC2_QRL_R1_DETAIL_ONLY/seed0/best_60e_stg1.pth",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT.parent
        / "reports/20_spatial_importance/S_BPC2_QRL_R1_DETAIL_ONLY/formal_60e_controls",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("QRL-R1 formal validation requires CUDA")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = args.batch_size
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0

    device = torch.device("cuda")
    model = cfg.model.to(device).eval()
    criterion = cfg.criterion.to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema", {}).get("module", checkpoint["model"])
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    qrl = model.decoder.qrl
    if qrl is None or not qrl.detail_only_delta:
        raise RuntimeError("checkpoint config did not construct QRL-R1")

    modes = ("learned", "zero", "shifted", "uniform")
    results = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        qrl.region_mode = mode
        mode_dir = args.output_dir / mode
        print(f"QRL-R1 60e validation mode={mode}", flush=True)
        stats, _ = evaluate(
            model,
            criterion,
            cfg.postprocessor,
            cfg.val_dataloader,
            cfg.evaluator,
            device,
            epoch=int(checkpoint.get("last_epoch", -1)),
            use_wandb=False,
            output_dir=str(mode_dir),
            num_visualization_sample_batch=0,
        )
        bbox = stats["coco_eval_bbox"]
        results[mode] = {
            "coco_eval_bbox": bbox,
            "AP": bbox[0],
            "AP50": bbox[1],
            "AP75": bbox[2],
            "APsmall": bbox[3],
            "APmedium": bbox[4],
            "APlarge": bbox[5],
            "AR100": bbox[8],
        }
        print(json.dumps({mode: results[mode]}, ensure_ascii=False), flush=True)
        torch.cuda.empty_cache()

    learned = results["learned"]
    deltas = {
        mode: {
            metric: learned[metric] - results[mode][metric]
            for metric in ("AP", "AP75", "APsmall", "AR100")
        }
        for mode in ("zero", "shifted", "uniform")
    }
    report = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_last_epoch": int(checkpoint.get("last_epoch", -1)),
        "weights": "ema.module" if "ema" in checkpoint else "model",
        "batch_size": args.batch_size,
        "modes": results,
        "learned_minus_controls": deltas,
    }
    report_path = args.output_dir / "formal_60e_control_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    qrl.region_mode = "learned"
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
