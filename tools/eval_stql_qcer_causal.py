"""Full test-development evaluation of trained STQL/QCER causal modes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import MethodType

import torch


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
    parser.add_argument(
        "--mode",
        choices=(
            "normal",
            "bypass",
            "zero_available",
            "batch_shuffle",
            "unavailable",
            "token_order_shuffle",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
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
        str(args.config.resolve()),
        resume=str(checkpoint),
        output_dir=str(output.parent / f"runtime_{args.mode}"),
    )
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    module = solver.ema.module if solver.ema else solver.model
    if not getattr(module, "qcer_enabled", False) or module.qcer is None:
        raise RuntimeError("checkpoint/config does not enable QCER")

    if args.mode == "bypass":
        module.qcer_bypass = True
    elif args.mode == "zero_available":
        module.rgbt_thermal_intervention = "zero_content_valid"
    elif args.mode == "batch_shuffle":
        module.rgbt_thermal_intervention = "batch_shuffle"
    elif args.mode == "unavailable":
        original_forward = module.forward

        def unavailable_forward(self, samples, targets=None, **kwargs):
            availability = torch.zeros(
                samples.shape[0], device=samples.device, dtype=torch.bool
            )
            return original_forward(
                samples, targets, qcer_availability=availability, **kwargs
            )

        module.forward = MethodType(unavailable_forward, module)
    elif args.mode == "token_order_shuffle":
        original_tokens = module.qcer._tokens

        def shuffled_tokens(self, thermal_features):
            tokens, centers, levels = original_tokens(thermal_features)
            order = torch.arange(
                tokens.shape[1] - 1, -1, -1, device=tokens.device
            )
            return tokens[:, order], centers[order], levels[order]

        module.qcer._tokens = MethodType(shuffled_tokens, module.qcer)

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
        "schema": "stql_qcer_full_causal_v1",
        "mode": args.mode,
        "config": str(args.config.resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "images": len(solver.val_dataloader.dataset),
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
