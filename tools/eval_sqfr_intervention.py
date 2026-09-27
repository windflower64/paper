"""Evaluate one N-SQFR1 checkpoint under same-weight S8 interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


FEATURE_MODES = ("full", "zero", "shifted")
VALID_MODES = FEATURE_MODES + ("disabled",)


def configure_sqfr_mode(model, mode):
    """Configure a same-checkpoint SQFR1 intervention and return the refiner."""
    if mode not in VALID_MODES:
        raise ValueError(f"unsupported SQFR1 mode {mode!r}")
    decoder = getattr(model, "decoder", None)
    refiner = getattr(decoder, "sqfr_refiner", None)
    if refiner is None:
        raise RuntimeError("evaluation requires a trained SQFR1 branch")
    if mode == "disabled":
        decoder.sqfr_refiner = None
    else:
        refiner.feature_mode = mode
    return refiner


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=VALID_MODES, required=True)
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
    refiner = configure_sqfr_mode(module, args.mode)

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
    residual_mean = (
        float(refiner.last_residual_abs_mean)
        if args.mode != "disabled" and refiner.last_residual_abs_mean is not None
        else None
    )
    residual_max = (
        float(refiner.last_residual_abs_max)
        if args.mode != "disabled" and refiner.last_residual_abs_max is not None
        else None
    )
    result = {
        "schema": "n_sqfr1_same_checkpoint_intervention_v1",
        "mode": args.mode,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "weight_source": "ema" if solver.ema else "model",
        "coco_eval_bbox": metrics,
        "AP": metrics[0],
        "AP50": metrics[1],
        "AP75": metrics[2],
        "APS": metrics[3],
        "APM": metrics[4],
        "APL": metrics[5],
        "AR100": metrics[8],
        "sqfr_active": args.mode != "disabled",
        "sqfr_feature_mode": None if args.mode == "disabled" else args.mode,
        "last_batch_residual_abs_mean": residual_mean,
        "last_batch_residual_abs_max": residual_max,
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
