#!/usr/bin/env python3
"""Validate SAM position and query identity learned by a REP2-MDQA checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_s/s_rep2_mdqa_w025_ft6_lr02_local.yml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def move_target(target, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }


def checkpoint_weights(checkpoint):
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"], "ema.module"
    model = checkpoint.get("model")
    if isinstance(model, dict):
        return model, "model"
    return checkpoint, "root"


def single_pair_loss(criterion, queries, pixels, target, source_index, target_index):
    device = queries.device
    indices = [
        (
            torch.tensor([source_index], device=device, dtype=torch.long),
            torch.tensor([target_index], device=device, dtype=torch.long),
        )
    ]
    return criterion._mdqa_loss(queries, pixels, [target], indices)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("REP2-MDQA validation requires CUDA")
    if args.batches <= 0:
        raise ValueError("--batches must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    cfg = YAMLConfig(str(args.config))
    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights, weight_source = checkpoint_weights(checkpoint)
    incompatible = model.load_state_dict(weights, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")

    # The head exists only on the training path. Keep dropout disabled (none in
    # this model) and freeze every batch-norm statistic during read-only use.
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

    aligned_losses = []
    shifted_losses = []
    wrong_query_losses = []
    processed_batches = 0
    skipped_samples = 0
    matched_pairs = 0
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for samples, targets in cfg.train_dataloader:
            samples = samples.to(device)
            targets = [move_target(target, device) for target in targets]
            outputs = model(samples, targets=targets)
            match_input = {
                key: value for key, value in outputs.items() if "aux" not in key
            }
            indices = criterion.matcher(match_input, targets)["indices"]
            query_embeddings = outputs["mdqa_query_embeddings"]
            pixel_features = outputs["mdqa_pixel_features"]
            num_queries = query_embeddings.shape[1]

            for batch_index, (source_indices, target_indices) in enumerate(indices):
                target = targets[batch_index]
                masks = target.get("masks")
                accepted_in_sample = 0
                for source_index, target_index in zip(
                    source_indices.tolist(), target_indices.tolist()
                ):
                    if masks is None or not bool(masks[target_index].any()):
                        continue
                    accepted_in_sample += 1
                    matched_pairs += 1
                    queries = query_embeddings[batch_index : batch_index + 1]
                    pixels = pixel_features[batch_index : batch_index + 1]
                    aligned = single_pair_loss(
                        criterion,
                        queries,
                        pixels,
                        target,
                        source_index,
                        target_index,
                    )
                    shifted_target = dict(target)
                    shifted_target["masks"] = torch.roll(
                        masks, shifts=masks.shape[-1] // 2, dims=-1
                    )
                    shifted = single_pair_loss(
                        criterion,
                        queries,
                        pixels,
                        shifted_target,
                        source_index,
                        target_index,
                    )
                    wrong_source = (source_index + num_queries // 2) % num_queries
                    wrong_query = single_pair_loss(
                        criterion,
                        queries,
                        pixels,
                        target,
                        wrong_source,
                        target_index,
                    )
                    aligned_losses.append(float(aligned))
                    shifted_losses.append(float(shifted))
                    wrong_query_losses.append(float(wrong_query))
                if accepted_in_sample == 0:
                    skipped_samples += 1

            processed_batches += 1
            if processed_batches >= args.batches:
                break

    if not aligned_losses:
        raise RuntimeError("No accepted matched SAM pairs were evaluated")
    aligned = np.asarray(aligned_losses, dtype=np.float64)
    shifted = np.asarray(shifted_losses, dtype=np.float64)
    wrong_query = np.asarray(wrong_query_losses, dtype=np.float64)
    shift_margin = shifted - aligned
    query_margin = wrong_query - aligned
    report = {
        "status": "PASS",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("last_epoch", -1)),
        "weight_source": weight_source,
        "batches": processed_batches,
        "matched_sam_pairs": matched_pairs,
        "skipped_samples": skipped_samples,
        "aligned_loss_mean": float(aligned.mean()),
        "shifted_loss_mean": float(shifted.mean()),
        "shifted_minus_aligned_mean": float(shift_margin.mean()),
        "aligned_better_than_shifted_fraction": float((shift_margin > 0).mean()),
        "wrong_query_loss_mean": float(wrong_query.mean()),
        "wrong_query_minus_matched_mean": float(query_margin.mean()),
        "matched_better_than_wrong_query_fraction": float((query_margin > 0).mean()),
        "position_interpretation": (
            "ALIGNED_MASK_PREFERRED"
            if shift_margin.mean() > 0
            else "NO_ALIGNED_MASK_PREFERENCE"
        ),
        "query_interpretation": (
            "MATCHED_QUERY_PREFERRED"
            if query_margin.mean() > 0
            else "NO_MATCHED_QUERY_PREFERENCE"
        ),
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
