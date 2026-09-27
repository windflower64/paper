"""Re-evaluate one M-OTE2 pair checkpoint with normal, bypass and empty IR."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


LABELS = (
    "AP", "AP50", "AP75", "APS", "APM", "APL",
    "AR1", "AR10", "AR100", "ARS", "ARM", "ARL",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_digest(module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sqer-mode", choices=("checkpoint_epoch", "active", "bypass"),
        default="checkpoint_epoch",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    for path in (repo, config, checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from src.core import YAMLConfig
    from src.misc import dist_utils
    from src.solver import TASKS
    from src.solver.det_engine import evaluate

    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)
    cfg = YAMLConfig(
        str(config), resume=str(checkpoint),
        output_dir=str(output.parent / f"eval_runtime_{output.stem}"),
    )
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    module = solver.ema.module if solver.ema else solver.model
    epoch = int(solver.last_epoch)
    if hasattr(module, "set_training_epoch"):
        module.set_training_epoch(epoch)
    if hasattr(solver.criterion, "training_epoch"):
        solver.criterion.training_epoch = epoch
    if args.sqer_mode == "active":
        module.sqer_bypass = False
    elif args.sqer_mode == "bypass":
        module.sqer_bypass = True

    original_mote = module.mote_fusion
    original_sd2 = module.sd2_conditioner
    original_intervention = module.rgbt_thermal_intervention
    branch = "M-OTE2" if original_mote is not None else "M-SD2.2"
    if original_mote is None and original_sd2 is None:
        raise RuntimeError("Checkpoint has no M fusion branch")
    if original_mote is not None and original_sd2 is not None:
        raise RuntimeError("Checkpoint has two M fusion branches")

    weights_before = model_digest(module)
    metrics_by_mode = {}
    for mode in ("normal", "disabled", "empty_thermal"):
        module.mote_fusion = original_mote
        module.sd2_conditioner = original_sd2
        module.rgbt_thermal_intervention = original_intervention
        if mode == "disabled":
            module.mote_fusion = None
            module.sd2_conditioner = None
        elif mode == "empty_thermal":
            module.rgbt_thermal_intervention = "zero"

        stats, _ = evaluate(
            module, solver.criterion, solver.postprocessor,
            solver.val_dataloader, solver.evaluator, solver.device,
            epoch=epoch, use_wandb=False,
        )
        metrics = stats["coco_eval_bbox"]
        if len(metrics) != len(LABELS):
            raise RuntimeError(f"Unexpected COCO metric count: {len(metrics)}")
        metrics_by_mode[mode] = dict(zip(LABELS, metrics))

    module.mote_fusion = original_mote
    module.sd2_conditioner = original_sd2
    module.rgbt_thermal_intervention = original_intervention
    weights_after = model_digest(module)
    if weights_before != weights_after:
        raise RuntimeError("Weights changed during same-checkpoint evaluation")

    result = {
        "schema": "mote2_pair_same_weight_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "config": str(config),
        "branch": branch,
        "weight_source": "ema" if solver.ema else "model",
        "checkpoint_epoch": epoch,
        "sqer_mode": args.sqer_mode,
        "sqer_bypass_at_evaluation": bool(module.sqer_bypass),
        "images": len(solver.val_dataloader.dataset),
        "uses_test_gt_in_forward": False,
        "weights_unchanged": True,
        "metrics": metrics_by_mode,
        "normal_minus_disabled_percentage_points": {
            key: (metrics_by_mode["normal"][key] - metrics_by_mode["disabled"][key]) * 100
            for key in LABELS
        },
        "normal_minus_empty_percentage_points": {
            key: (metrics_by_mode["normal"][key] - metrics_by_mode["empty_thermal"][key]) * 100
            for key in LABELS
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    dist_utils.cleanup()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
