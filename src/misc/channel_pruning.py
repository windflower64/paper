"""Explicit, reproducible channel pruning used by C-ID experiments.

The transform runs after A00 tuning weights are loaded but before EMA and the
optimizer are created.  It physically changes tensor shapes; it is not a mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch_pruning as tp


def _select_removed(root, spec, original):
    ratio = float(spec["ratio"])
    count = max(1, int(round(original * ratio)))
    method = str(spec["method"]).lower()
    seed = int(spec.get("index_seed", 0))

    if method == "task":
        score_file = Path(spec["score_file"])
        score_key = str(spec.get("score_key", "train_s16_task"))
        with np.load(score_file) as data:
            score = np.asarray(data[score_key], dtype=np.float64).mean(0)
        if score.shape != (original,):
            raise ValueError(
                f"Task score {score_key} shape {score.shape} != ({original},)"
            )
    elif method in ("magnitude", "weight_l1"):
        score = root.weight.detach().abs().mean((1, 2, 3)).cpu().numpy()
    elif method == "random":
        score = np.random.default_rng(seed).random(original)
    else:
        raise ValueError(f"Unsupported channel_prune method: {method}")

    removed = np.argsort(score)[:count].astype(np.int64)
    return score, sorted(removed.tolist())


def apply_channel_pruning(model, spec):
    stage = str(spec.get("stage", "s16")).lower()
    if stage != "s16":
        raise ValueError("C-ID first gate is intentionally restricted to S16")

    root = model.backbone.stages[2].blocks[0].aggregation[1].conv
    projection = model.encoder.input_proj[0].conv
    original = int(root.out_channels)
    score, removed = _select_removed(root, spec, original)
    kept = [i for i in range(original) if i not in set(removed)]

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    example = torch.zeros(1, 3, 512, 640, device=device)
    graph = tp.DependencyGraph().build_dependency(
        model.backbone, example_inputs=example
    )
    group = graph.get_pruning_group(
        root, tp.prune_conv_out_channels, idxs=removed
    )
    if not graph.check_pruning_group(group):
        raise RuntimeError("Invalid S16 dependency pruning group")
    group.prune()
    tp.prune_conv_in_channels(projection, removed)
    if was_training:
        model.train()

    if root.out_channels != len(kept) or projection.in_channels != len(kept):
        raise RuntimeError(
            f"Unexpected pruned widths root={root.out_channels}, "
            f"projection={projection.in_channels}, expected={len(kept)}"
        )

    return {
        "stage": "s16", "method": str(spec["method"]).lower(),
        "ratio": float(spec["ratio"]), "index_seed": int(spec.get("index_seed", 0)),
        "original_channels": original, "remaining_channels": len(kept),
        "removed_channels": removed, "kept_channels": kept,
        "score_min": float(np.min(score)), "score_max": float(np.max(score)),
        "score_mean": float(np.mean(score)),
        "score_file": str(spec.get("score_file", "")),
        "score_key": str(spec.get("score_key", "")),
        "physical_shape_change": True,
    }
