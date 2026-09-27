#!/usr/bin/env python3
"""Post-hoc scale/shape diagnosis for trained REP3-QCSR checkpoints."""

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
    return checkpoint.get("model", checkpoint), "model"


def region_from_mask(mask, height, width, radius, floor):
    mask = F.interpolate(
        mask.float()[None, None], size=(height, width), mode="area"
    ).clamp(0.0, 1.0)
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(
        1.0 - mask, kernel_size=kernel, stride=1, padding=radius
    )
    boundary = (dilated - eroded).clamp(0.0, 1.0)
    low = F.interpolate(
        mask,
        size=(max(1, height // 2), max(1, width // 2)),
        mode="area",
    )
    reconstructed = F.interpolate(
        low, size=(height, width), mode="bilinear", align_corners=False
    )
    incoherence = (mask - reconstructed).abs().clamp(0.0, 1.0)
    return (boundary * (floor + (1.0 - floor) * incoherence))[0, 0]


def metrics(student, teacher, weight):
    mass = weight.sum().clamp_min(1e-6)
    teacher = teacher.float()
    student = student.float()
    raw_huber = (
        F.smooth_l1_loss(student, teacher, reduction="none", beta=0.5) * weight
    ).sum() / mass
    probability_mse = (
        (student.sigmoid() - teacher.sigmoid()).square() * weight
    ).sum() / mass

    def normalize(value):
        mean = (value * weight).sum() / mass
        centered = value - mean
        std = ((centered.square() * weight).sum() / mass).sqrt().clamp_min(1e-6)
        return centered / std

    teacher_z = normalize(teacher)
    student_z = normalize(student)
    shape_mse = ((student_z - teacher_z).square() * weight).sum() / mass
    shape_cosine = (student_z * teacher_z * weight).sum() / mass
    teacher_rms = ((teacher.square() * weight).sum() / mass).sqrt()
    student_rms = ((student.square() * weight).sum() / mass).sqrt()
    return {
        "raw_huber": float(raw_huber),
        "probability_mse": float(probability_mse),
        "shape_mse": float(shape_mse),
        "shape_cosine": float(shape_cosine),
        "teacher_rms": float(teacher_rms),
        "student_rms": float(student_rms),
        "student_teacher_rms_ratio": float(student_rms / teacher_rms.clamp_min(1e-6)),
    }


def summarize(records, shifted_records):
    keys = records[0].keys()
    result = {}
    for key in keys:
        aligned = np.asarray([record[key] for record in records], dtype=np.float64)
        shifted = np.asarray(
            [record[key] for record in shifted_records], dtype=np.float64
        )
        result[key] = {
            "aligned_mean": float(aligned.mean()),
            "shifted_mean": float(shifted.mean()),
            "shifted_minus_aligned_mean": float((shifted - aligned).mean()),
            "aligned_lower_fraction": float((aligned < shifted).mean()),
        }
    return result


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("REP3 transfer diagnosis requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    cfg = YAMLConfig(str(args.config))
    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights, source = checkpoint_weights(checkpoint)
    incompatible = model.load_state_dict(weights, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

    aligned_records = []
    shifted_records = []
    processed_batches = 0
    matched_pairs = 0
    scale = float(model.decoder.qcsr_dim) ** 0.5
    with torch.no_grad():
        for samples, targets in cfg.train_dataloader:
            samples = samples.to(device)
            targets = [move_target(target, device) for target in targets]
            outputs = model(samples, targets=targets)
            match_input = {
                key: value for key, value in outputs.items() if "aux" not in key
            }
            indices = criterion.matcher(match_input, targets)["indices"]
            queries = outputs["qcsr_query_embeddings"]
            teachers = outputs["qcsr_teacher_pixels"]
            students = F.interpolate(
                outputs["qcsr_student_pixels"],
                size=teachers.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            height, width = teachers.shape[-2:]
            for batch_index, (source_indices, target_indices) in enumerate(indices):
                masks = targets[batch_index].get("masks")
                for source_index, target_index in zip(
                    source_indices.tolist(), target_indices.tolist()
                ):
                    if masks is None or not bool(masks[target_index].any()):
                        continue
                    query = queries[batch_index, source_index]
                    teacher = torch.einsum(
                        "d,dhw->hw", query, teachers[batch_index]
                    ) / scale
                    student = torch.einsum(
                        "d,dhw->hw", query, students[batch_index]
                    ) / scale
                    region = region_from_mask(
                        masks[target_index],
                        height,
                        width,
                        criterion.qcsr_boundary_radius,
                        criterion.qcsr_incoherence_floor,
                    )
                    shifted_mask = torch.roll(
                        masks[target_index], shifts=masks.shape[-1] // 2, dims=-1
                    )
                    shifted_region = region_from_mask(
                        shifted_mask,
                        height,
                        width,
                        criterion.qcsr_boundary_radius,
                        criterion.qcsr_incoherence_floor,
                    )
                    if bool(region.sum() > 0) and bool(shifted_region.sum() > 0):
                        aligned_records.append(metrics(student, teacher, region))
                        shifted_records.append(metrics(student, teacher, shifted_region))
                        matched_pairs += 1
            processed_batches += 1
            if processed_batches >= args.batches:
                break

    if not aligned_records:
        raise RuntimeError("No valid QCSR pairs were diagnosed")
    report = {
        "status": "PASS",
        "method": "REP3-QCSR-SCALE-DIAGNOSIS",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("last_epoch", -1)),
        "weight_source": source,
        "batches": processed_batches,
        "matched_pairs": matched_pairs,
        "metrics": summarize(aligned_records, shifted_records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
