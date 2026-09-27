"""Compare S-QER2 normal and bypass inference with one unchanged checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from src.core import YAMLConfig
    from src.misc import dist_utils
    from src.solver import TASKS
    from src.solver.det_engine import evaluate

    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)
    cfg = YAMLConfig(str(config), resume=str(checkpoint), output_dir=str(output.parent / "runtime"))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    model = solver.ema.module if solver.ema else solver.model
    before = model_digest(model)

    normal_stats, _ = evaluate(
        model, solver.criterion, solver.postprocessor, solver.val_dataloader,
        solver.evaluator, solver.device, epoch=-1, use_wandb=False,
    )
    normal_metrics = normal_stats["coco_eval_bbox"]
    model.sqer_bypass = True
    bypass_stats, _ = evaluate(
        model, solver.criterion, solver.postprocessor, solver.val_dataloader,
        solver.evaluator, solver.device, epoch=-1, use_wandb=False,
    )
    bypass_metrics = bypass_stats["coco_eval_bbox"]
    after = model_digest(model)
    if before != after:
        raise RuntimeError("model state changed during same-weight evaluation")

    labels = ("AP", "AP50", "AP75", "APS", "APM", "APL", "AR1", "AR10", "AR100", "ARS", "ARM", "ARL")
    result = {
        "schema": "sqer2_same_weight_normal_bypass_v1",
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "weight_source": "ema" if solver.ema else "model",
        "images": len(solver.val_dataloader.dataset),
        "uses_test_ground_truth_in_forward": False,
        "model_state_unchanged": before == after,
        "sqer2_normal": dict(zip(labels, normal_metrics)),
        "sqer2_bypass": dict(zip(labels, bypass_metrics)),
        "normal_minus_bypass_percentage_points": [
            (normal - bypass) * 100 for normal, bypass in zip(normal_metrics, bypass_metrics)
        ],
        "interpretation_limit": "同权重旁路是机制诊断，不等价于从训练开始移除 SQER2 的独立对照。",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    dist_utils.cleanup()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
