#!/usr/bin/env python3
"""Measure one full AMP forward/loss/backward step at several batch sizes."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-batch", type=int, default=32)
    args = parser.parse_args()

    repo = args.repo.resolve(strict=True)
    sys.path.insert(0, str(repo))
    from src.core import YAMLConfig  # noqa: PLC0415

    config = repo / "experiments/dfine/dfine_n_visible_640x512.yml"
    cfg = YAMLConfig(str(config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = args.max_batch
    cfg.yaml_cfg["train_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["train_dataloader"]["collate_fn"]["base_size_repeat"] = None

    model = cfg.model
    checkpoint = torch.load(args.checkpoint, map_location="cpu")["model"]
    current = model.state_dict()
    compatible = {key: value for key, value in checkpoint.items() if key in current and current[key].shape == value.shape}
    model.load_state_dict(compatible, strict=False)
    criterion = cfg.criterion
    model = model.cuda().train()
    criterion = criterion.cuda().train()

    images, targets = next(iter(cfg.train_dataloader))
    print("loaded_batch", tuple(images.shape), flush=True)
    candidates = [
        value
        for value in (4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128)
        if value <= args.max_batch
    ]
    for batch_size in candidates:
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        batch_images = images[:batch_size].cuda(non_blocking=False)
        batch_targets = [
            {key: value.cuda() if torch.is_tensor(value) else value for key, value in target.items()}
            for target in targets[:batch_size]
        ]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(batch_images, targets=batch_targets)
            with torch.autocast("cuda", enabled=False):
                loss_dict = criterion(
                    outputs,
                    batch_targets,
                    epoch=0,
                    step=0,
                    global_step=0,
                    epoch_step=100,
                )
                loss = sum(loss_dict.values())
            loss.backward()
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(
                "batch",
                batch_size,
                "ok",
                "loss",
                round(float(loss.detach().cpu()), 5),
                "peak_mib",
                round(torch.cuda.max_memory_allocated() / 1024**2, 2),
                "step_ms",
                round(elapsed_ms, 2),
                flush=True,
            )
        except torch.cuda.OutOfMemoryError as exc:
            print("batch", batch_size, "OOM", str(exc).splitlines()[0], flush=True)
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break

    print("BATCH_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
