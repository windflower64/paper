#!/usr/bin/env python3
"""Validate learned query-conditioned S8->S16 retention for REP3-QCSR."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_s/s_rep3_qcsr_w025_ft6_lr02_local.yml",
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


def one_pair_indices(device, source_index, target_index):
    return [
        (
            torch.tensor([source_index], device=device, dtype=torch.long),
            torch.tensor([target_index], device=device, dtype=torch.long),
        )
    ]


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("REP3-QCSR validation requires CUDA")
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

    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

    transfer_losses = []
    shifted_transfer_losses = []
    teacher_mask_losses = []
    student_mask_losses = []
    wrong_query_teacher_losses = []
    matched_pairs = 0
    skipped_samples = 0
    processed_batches = 0
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for samples, targets in cfg.train_dataloader:
            samples = samples.to(device)
            targets = [move_target(target, device) for target in targets]
            outputs = model(samples, targets=targets)
            match_input = {
                key: value for key, value in outputs.items() if "aux" not in key
            }
            matched = criterion.matcher(match_input, targets)["indices"]
            queries_all = outputs["qcsr_query_embeddings"]
            teachers_all = outputs["qcsr_teacher_pixels"]
            students_all = outputs["qcsr_student_pixels"]
            num_queries = queries_all.shape[1]

            for batch_index, (source_indices, target_indices) in enumerate(matched):
                target = targets[batch_index]
                masks = target.get("masks")
                accepted = 0
                for source_index, target_index in zip(
                    source_indices.tolist(), target_indices.tolist()
                ):
                    if masks is None or not bool(masks[target_index].any()):
                        continue
                    accepted += 1
                    matched_pairs += 1
                    query = queries_all[batch_index : batch_index + 1]
                    teacher = teachers_all[batch_index : batch_index + 1]
                    student = students_all[batch_index : batch_index + 1]
                    indices = one_pair_indices(device, source_index, target_index)
                    _, transfer = criterion._qcsr_losses(
                        query, teacher, student, [target], indices
                    )
                    shifted_target = dict(target)
                    shifted_target["masks"] = torch.roll(
                        masks, shifts=masks.shape[-1] // 2, dims=-1
                    )
                    _, shifted_transfer = criterion._qcsr_losses(
                        query, teacher, student, [shifted_target], indices
                    )
                    teacher_mask = criterion._mdqa_loss(
                        query, teacher, [target], indices
                    )
                    student_mask = criterion._mdqa_loss(
                        query, student, [target], indices
                    )
                    wrong_source = (source_index + num_queries // 2) % num_queries
                    wrong_indices = one_pair_indices(device, wrong_source, target_index)
                    wrong_teacher = criterion._mdqa_loss(
                        query, teacher, [target], wrong_indices
                    )
                    transfer_losses.append(float(transfer))
                    shifted_transfer_losses.append(float(shifted_transfer))
                    teacher_mask_losses.append(float(teacher_mask))
                    student_mask_losses.append(float(student_mask))
                    wrong_query_teacher_losses.append(float(wrong_teacher))
                if accepted == 0:
                    skipped_samples += 1

            processed_batches += 1
            if processed_batches >= args.batches:
                break

    if not transfer_losses:
        raise RuntimeError("No accepted matched SAM pairs were evaluated")
    transfer = np.asarray(transfer_losses, dtype=np.float64)
    shifted = np.asarray(shifted_transfer_losses, dtype=np.float64)
    teacher_mask = np.asarray(teacher_mask_losses, dtype=np.float64)
    student_mask = np.asarray(student_mask_losses, dtype=np.float64)
    wrong_teacher = np.asarray(wrong_query_teacher_losses, dtype=np.float64)
    shift_margin = shifted - transfer
    query_margin = wrong_teacher - teacher_mask
    student_gap = student_mask - teacher_mask
    method = (
        "REP3.1-QCSR-SN"
        if criterion.qcsr_transfer_mode == "shape_correlation"
        else "REP3-QCSR"
    )
    report = {
        "status": "PASS",
        "method": method,
        "qcsr_transfer_mode": criterion.qcsr_transfer_mode,
        "qcsr_shape_eps": criterion.qcsr_shape_eps,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("last_epoch", -1)),
        "weight_source": weight_source,
        "batches": processed_batches,
        "matched_sam_pairs": matched_pairs,
        "skipped_samples": skipped_samples,
        "aligned_transfer_loss_mean": float(transfer.mean()),
        "shifted_transfer_loss_mean": float(shifted.mean()),
        "shifted_minus_aligned_mean": float(shift_margin.mean()),
        "aligned_better_than_shifted_fraction": float((shift_margin > 0).mean()),
        "teacher_sam_loss_mean": float(teacher_mask.mean()),
        "student_sam_loss_mean": float(student_mask.mean()),
        "student_minus_teacher_sam_loss_mean": float(student_gap.mean()),
        "wrong_query_teacher_sam_loss_mean": float(wrong_teacher.mean()),
        "wrong_minus_matched_query_sam_loss_mean": float(query_margin.mean()),
        "matched_query_better_fraction": float((query_margin > 0).mean()),
        "position_interpretation": (
            "ALIGNED_REGION_RETAINED_BETTER"
            if shift_margin.mean() > 0
            else "NO_ALIGNED_RETENTION_PREFERENCE"
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
