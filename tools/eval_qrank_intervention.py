#!/usr/bin/env python3
"""Evaluate one Q-Rank1 checkpoint under same-weight score interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


VALID_MODES = ("learned", "zero", "disabled")


def configure_qrank_mode(model, mode):
    if mode not in VALID_MODES:
        raise ValueError(f"unsupported Q-Rank1 mode {mode!r}")
    decoder = getattr(model, "decoder", None)
    ranker = getattr(decoder, "qrank_score_bias", None)
    if ranker is None:
        raise RuntimeError("evaluation requires a trained Q-Rank1 module")
    if mode == "disabled":
        decoder.qrank_score_bias = None
    else:
        ranker.intervention_mode = mode
    return ranker


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
    ranker = configure_qrank_mode(module, args.mode)

    with torch.no_grad():
        bias = ranker.rank_bias.detach().float().cpu()
        layer_stats = [
            {
                "layer": index,
                "mean": float(layer.mean()),
                "std": float(layer.std()),
                "min": float(layer.min()),
                "max": float(layer.max()),
            }
            for index, layer in enumerate(bias)
        ]

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
        "schema": "q_rank1_same_checkpoint_intervention_v1",
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
        "qrank_active": args.mode != "disabled",
        "qrank_intervention_mode": None if args.mode == "disabled" else args.mode,
        "rank_bias_abs_mean": float(bias.abs().mean()),
        "rank_bias_std": float(bias.std()),
        "rank_bias_layer_stats": layer_stats,
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
