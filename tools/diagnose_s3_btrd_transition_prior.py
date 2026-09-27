#!/usr/bin/env python3
"""Audit whether the BTRD max-average transition prior enriches UAV rings.

This is a zero-training, Val-only mechanism probe on frozen A00 S8 features.
The proxy intentionally excludes BTRD's new learnable projection so that the
training decision is supported by information already present in the carrier.
GT boxes are used only to score the proxy; they are never model inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--stage-index", type=int, default=2)
    return parser.parse_args()


def make_target_mask(targets, height, width, image_height, image_width, device):
    mask = torch.zeros((len(targets), 1, height, width), dtype=torch.bool, device=device)
    for batch_index, target in enumerate(targets):
        boxes = target["boxes"]
        absolute_xyxy = bool(boxes.numel() and boxes.max() > 2)
        for box in boxes:
            if absolute_xyxy:
                x1, y1, x2, y2 = box
                x1, x2 = x1 / image_width * width, x2 / image_width * width
                y1, y2 = y1 / image_height * height, y2 / image_height * height
            else:
                cx, cy, box_width, box_height = box
                x1, x2 = (cx - box_width / 2) * width, (cx + box_width / 2) * width
                y1, y2 = (cy - box_height / 2) * height, (cy + box_height / 2) * height
            ix1 = max(0, min(width - 1, int(torch.floor(x1).item())))
            iy1 = max(0, min(height - 1, int(torch.floor(y1).item())))
            ix2 = max(ix1 + 1, min(width, int(torch.ceil(x2).item())))
            iy2 = max(iy1 + 1, min(height, int(torch.ceil(y2).item())))
            mask[batch_index, 0, iy1:iy2, ix1:ix2] = True
    return mask


def average_rank_auc(positive, negative):
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if not positive.size or not negative.size:
        return None
    values = np.concatenate((positive, negative))
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = ranks[: positive.size].sum()
    return float(
        (rank_sum - positive.size * (positive.size + 1) / 2)
        / (positive.size * negative.size)
    )


def average_precision(positive, negative):
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if not positive.size or not negative.size:
        return None
    values = np.concatenate((positive, negative))
    labels = np.concatenate((np.ones(positive.size), np.zeros(negative.size)))
    order = np.argsort(-values, kind="mergesort")
    labels = labels[order]
    precision = np.cumsum(labels) / np.arange(1, labels.size + 1)
    return float(precision[labels == 1].mean())


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def matched_background_values(field, background, count, seed):
    candidates = field[background]
    count = min(int(count), int(candidates.numel()))
    if count <= 0:
        return field.new_empty((0,))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randperm(candidates.numel(), generator=generator)[:count]
    return candidates[indices.to(candidates.device)]


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(max(1, min(4, args.threads // 4)))
    sys.path.insert(0, str(args.repo))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model.cpu().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = checkpoint.get("ema", {}).get("module")
    weight_source = "ema.module"
    if weights is None:
        weights = checkpoint.get("model", checkpoint)
        weight_source = "model_or_raw"
    model.load_state_dict(weights, strict=True)

    backbone = model.backbone
    if args.stage_index <= 0 or args.stage_index >= len(backbone.stages):
        raise ValueError("stage-index must name a valid stage after at least one prior stage")

    storage = {
        kernel: defaultdict(list) for kernel in (2, 3, 5)
    }
    images_seen = 0
    object_images = 0
    started = time.time()

    with torch.inference_mode():
        for batch_index, (samples, targets) in enumerate(cfg.val_dataloader):
            if batch_index >= args.max_batches:
                break
            samples = samples.cpu()
            targets = [
                {
                    key: value.cpu() if torch.is_tensor(value) else value
                    for key, value in target.items()
                }
                for target in targets
            ]
            x = backbone.stem(samples)
            for stage in backbone.stages[: args.stage_index]:
                x = stage(x)

            activation = x.float().square().mean(dim=1, keepdim=True).sqrt()
            mean = activation.mean(dim=(-2, -1), keepdim=True)
            std = activation.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            activation = (activation - mean) / std

            for kernel in (2, 3, 5):
                maximum = F.max_pool2d(activation, kernel, stride=1)
                average = F.avg_pool2d(activation, kernel, stride=1)
                transition = torch.sigmoid(
                    F.relu(torch.sigmoid(maximum) - torch.sigmoid(average))
                )
                target = make_target_mask(
                    targets,
                    transition.shape[-2],
                    transition.shape[-1],
                    samples.shape[-2],
                    samples.shape[-1],
                    transition.device,
                )
                ring = F.max_pool2d(target.float(), 3, 1, 1).bool() & ~target
                background = ~F.max_pool2d(target.float(), 5, 1, 2).bool()

                for item_index in range(len(targets)):
                    ring_count = int(ring[item_index].sum().item())
                    if ring_count == 0:
                        continue
                    object_images += int(kernel == 2)
                    item = transition[item_index, 0]
                    ring_values = item[ring[item_index, 0]]
                    target_values = item[target[item_index, 0]]
                    background_values = matched_background_values(
                        item,
                        background[item_index, 0],
                        ring_count,
                        20260814 + int(targets[item_index]["image_id"].item()) + kernel,
                    )
                    flat = item.flatten()
                    generator = torch.Generator(device="cpu").manual_seed(
                        20260901 + int(targets[item_index]["image_id"].item()) + kernel
                    )
                    shuffled = flat[
                        torch.randperm(flat.numel(), generator=generator)
                    ].reshape_as(item)
                    shuffled_ring = shuffled[ring[item_index, 0]]

                    storage[kernel]["ring"].extend(ring_values.tolist())
                    storage[kernel]["target"].extend(target_values.tolist())
                    storage[kernel]["background_matched"].extend(background_values.tolist())
                    storage[kernel]["shuffled_ring"].extend(shuffled_ring.tolist())

            images_seen += len(targets)
            print(
                f"batch={batch_index + 1}/{args.max_batches} images={images_seen}",
                flush=True,
            )

    results = {}
    for kernel, groups in storage.items():
        ring = groups["ring"]
        background = groups["background_matched"]
        results[str(kernel)] = {
            "regions": {name: describe(values) for name, values in groups.items()},
            "ring_vs_matched_background": {
                "roc_auc": average_rank_auc(ring, background),
                "average_precision": average_precision(ring, background),
                "mean_ratio": (
                    float(np.mean(ring) / max(np.mean(background), 1e-12))
                    if ring and background
                    else None
                ),
                "mean_difference": (
                    float(np.mean(ring) - np.mean(background))
                    if ring and background
                    else None
                ),
            },
            "ring_vs_shuffled_ring": {
                "roc_auc": average_rank_auc(ring, groups["shuffled_ring"]),
                "mean_difference": (
                    float(np.mean(ring) - np.mean(groups["shuffled_ring"]))
                    if ring and groups["shuffled_ring"]
                    else None
                ),
            },
        }

    output = {
        "protocol": {
            "split": "Val (possibly capped by max_batches)",
            "images_seen": images_seen,
            "object_images": object_images,
            "max_batches": args.max_batches,
            "threads": args.threads,
            "stage_index": args.stage_index,
            "weight_source": weight_source,
            "elapsed_seconds": time.time() - started,
            "warning": (
                "GT is scoring-only. This parameter-free proxy audits the BTRD core prior; "
                "it is not the final learnable BTRD gate and cannot prove final AP gain."
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
