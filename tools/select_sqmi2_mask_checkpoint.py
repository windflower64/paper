#!/usr/bin/env python3
"""Select S-QMI2 stage-A checkpoint by mask-box quality, not detector AP."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from audit_s_sqmi1_box_quality import inspect


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_s/s_sqmi2_s4_mask_warmup_b16_6e_testdev_local.yml",
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=WORKSPACE / "outputs/S_SQMI2_S4_MASK_WARMUP_B16_6E_TESTDEV/seed0",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batches", type=int, default=114)
    parser.add_argument("--min-mask-iou", type=float, default=0.65)
    parser.add_argument("--min-better-ratio", type=float, default=0.20)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "reports/146_sqmi2_s4/quality_selection.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = []
    for epoch in range(args.epochs):
        checkpoint = args.run / f"checkpoint{epoch:04d}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result = inspect(args.config, checkpoint, args.batches)
        result["epoch"] = epoch
        records.append(result)
    selected = max(
        records,
        key=lambda item: (
            item["mean_iou_valid"]["mask"],
            item["mask_better_than_base_ratio"],
        ),
    )
    passed = (
        selected["mean_iou_valid"]["mask"] >= args.min_mask_iou
        and selected["mask_better_than_base_ratio"] >= args.min_better_ratio
    )
    result = {
        "status": "pass" if passed else "stop",
        "decision": (
            "allow_quality_gate_stage" if passed else "close_sqmi2_before_detection_training"
        ),
        "thresholds": {
            "min_mask_iou": args.min_mask_iou,
            "min_better_ratio": args.min_better_ratio,
        },
        "selected_epoch": selected["epoch"],
        "selected_checkpoint": selected["checkpoint"],
        "selected": selected,
        "epochs": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

