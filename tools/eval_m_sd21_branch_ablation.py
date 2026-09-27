"""Evaluate one M-SD2.1 checkpoint with the conditioner enabled or disabled."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "enabled",
            "disabled",
            "empty_thermal",
            "s16_only",
            "s32_only",
            "mean_tokens",
            "zero_tokens",
            "no_contrast",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from src.core import YAMLConfig
    from src.misc import dist_utils
    from src.solver import TASKS
    from src.solver.det_engine import evaluate

    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)
    cfg = YAMLConfig(
        str(config),
        resume=str(checkpoint),
        output_dir=str(output.parent / f"eval_runtime_{args.mode}"),
    )
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    module = solver.ema.module if solver.ema else solver.model
    conditioner_class = (
        type(module.sd2_conditioner).__name__
        if module.sd2_conditioner is not None
        else None
    )
    if conditioner_class != "SpatiallyDecoupledThermalTokenConditioner":
        raise RuntimeError(f"unexpected conditioner: {conditioner_class}")
    if args.mode == "disabled":
        module.sd2_conditioner = None
    elif args.mode == "empty_thermal":
        module.rgbt_thermal_intervention = "zero"
    elif args.mode in ("s16_only", "s32_only"):
        module.sd2_conditioner.intervention_level_mode = args.mode
    elif args.mode == "mean_tokens":
        module.sd2_conditioner.intervention_token_mode = "mean_repeat"
    elif args.mode == "zero_tokens":
        module.sd2_conditioner.intervention_token_mode = "zero"
    elif args.mode == "no_contrast":
        module.sd2_conditioner.thermal_contrastive = False

    stats, _ = evaluate(
        module,
        solver.criterion,
        solver.postprocessor,
        solver.val_dataloader,
        solver.evaluator,
        solver.device,
        epoch=-1,
        use_wandb=False,
    )
    metrics = stats["coco_eval_bbox"]
    result = {
        "schema": "m_sd21_same_checkpoint_branch_ablation_v1",
        "mode": args.mode,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "conditioner_class_in_checkpoint": conditioner_class,
        "conditioner_active": args.mode != "disabled",
        "coco_eval_bbox": metrics,
        "AP": metrics[0],
        "AP50": metrics[1],
        "AP75": metrics[2],
        "APS": metrics[3],
        "APM": metrics[4],
        "APL": metrics[5],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    dist_utils.cleanup()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
