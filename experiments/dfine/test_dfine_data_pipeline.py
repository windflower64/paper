#!/usr/bin/env python3
"""CPU smoke test for the Anti-UAV-6K D-FINE data configurations."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.core import YAMLConfig


CONFIGS = (
    Path("experiments/dfine/dfine_n_visible_640_square.yml"),
    Path("experiments/dfine/dfine_n_visible_640x512.yml"),
)


for path in CONFIGS:
    print(f"=== {path.name} ===", flush=True)
    cfg = YAMLConfig(str(path))
    for loader_name in ("train_dataloader", "val_dataloader"):
        loader_cfg = cfg.yaml_cfg[loader_name]
        loader_cfg["total_batch_size"] = 2
        loader_cfg["num_workers"] = 0

    train_loader = cfg.train_dataloader
    images, targets = next(iter(train_loader))
    print(
        "train_batch",
        tuple(images.shape),
        images.dtype,
        float(images.min()),
        float(images.max()),
        flush=True,
    )
    for target in targets:
        boxes = target["boxes"]
        if boxes.numel():
            assert torch.isfinite(boxes).all()
            assert (boxes >= 0).all() and (boxes <= 1).all(), boxes
            assert (target["labels"] == 0).all(), target["labels"]

    val_loader = cfg.val_dataloader
    val_images, val_targets = next(iter(val_loader))
    print("val_batch", tuple(val_images.shape), val_images.dtype, flush=True)
    print(
        "dataset",
        len(train_loader.dataset),
        len(val_loader.dataset),
        "categories",
        train_loader.dataset.categories,
        flush=True,
    )
    evaluator = cfg.evaluator
    print("evaluator", type(evaluator).__name__, flush=True)

print("DATA_PIPELINE_OK", flush=True)
