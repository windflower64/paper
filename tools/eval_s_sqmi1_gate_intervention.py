#!/usr/bin/env python3
"""Exact COCO evaluation of one S-QMI1 checkpoint with its box gate on/off."""

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from src.solver.det_engine import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_s/s_sqmi1_init_c_gq1_b16a2_12e_testdev_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=WORKSPACE
        / "outputs/S_SQMI1_INIT_C_GQ1_B16A2_12E_TESTDEV/seed0/best_stg1.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "reports/145_sqmi1_result/gate_intervention.json",
    )
    return parser.parse_args()


def checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"]
    return checkpoint.get("model", checkpoint)


def run(config, state, enabled):
    cfg = YAMLConfig(str(config))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    model.load_state_dict(state, strict=True)
    model.decoder.sqmi_apply_initialization = bool(enabled)
    model.decoder.sqmi.apply_initialization = bool(enabled)
    model = model.cuda().eval()
    criterion = cfg.criterion.cuda().eval()
    _, evaluator = evaluate(
        model,
        criterion,
        cfg.postprocessor,
        cfg.val_dataloader,
        cfg.evaluator,
        torch.device("cuda"),
        epoch=-1,
        use_wandb=False,
    )
    stats = evaluator.coco_eval["bbox"].stats.tolist()
    del model, criterion, evaluator
    torch.cuda.empty_cache()
    return stats


def main():
    args = parse_args()
    state = checkpoint_state(args.checkpoint)
    gate_on = run(args.config, state, True)
    gate_off = run(args.config, state, False)
    names = ["ap", "ap50", "ap75", "aps", "apm", "apl", "ar1", "ar10", "ar100", "ars", "arm", "arl"]
    result = {
        "checkpoint": str(args.checkpoint),
        "gate_on": dict(zip(names, gate_on)),
        "gate_off": dict(zip(names, gate_off)),
        "gate_on_minus_off": {
            name: on - off for name, on, off in zip(names, gate_on, gate_off)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
