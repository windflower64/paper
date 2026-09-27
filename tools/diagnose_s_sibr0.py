#!/usr/bin/env python3
"""Zero-training audit for SAM-guided Incoherent Boundary Retention (SIBR)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "experiments/phase_s/s_rep1_1_spar_w025_ft6_lr02_local.yml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent
        / "reports/21_public_reproduction/S_SIBR_DIAG0/report_actual_s8_s16.json",
    )
    return parser.parse_args()


def normalize_map(value):
    flat = value.flatten(1)
    minimum = flat.amin(dim=1).view(-1, 1, 1, 1)
    maximum = flat.amax(dim=1).view(-1, 1, 1, 1)
    return (value - minimum) / (maximum - minimum).clamp_min(1e-6)


def weighted_accumulate(accumulator, name, values, weights):
    numerator = float((values * weights).sum())
    denominator = float(weights.sum())
    accumulator[name]["numerator"] += numerator
    accumulator[name]["denominator"] += denominator


def weighted_means(accumulator):
    return {
        name: values["numerator"] / max(values["denominator"], 1e-12)
        for name, values in accumulator.items()
    }


def checkpoint_weights(checkpoint):
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"], "ema.module"
    model = checkpoint.get("model")
    if isinstance(model, dict):
        return model, "model"
    return checkpoint, "root"


def size_name(mask_area):
    if mask_area < 32**2:
        return "small"
    if mask_area < 96**2:
        return "medium"
    return "large"


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SIBR-DIAG0 requires CUDA")
    if args.batches <= 0:
        raise ValueError("--batches must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    cfg = YAMLConfig(str(args.config))
    model = cfg.model
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights, weight_source = checkpoint_weights(checkpoint)
    compatible = {
        key: value
        for key, value in weights.items()
        if key in model.state_dict() and model.state_dict()[key].shape == value.shape
    }
    incompatible = model.load_state_dict(compatible, strict=False)
    unexpected_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("backbone.spar_fusion.")
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={unexpected_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    backbone = model.backbone.to(device).eval()
    captured_features = {}

    def capture(name):
        def hook(_module, _inputs, output):
            captured_features[name] = output

        return hook

    # D-FINE-N returns only S16/S32. Capture its internal stride-8 and
    # stride-16 stages explicitly; the earlier audit incorrectly renamed the
    # two returned detector tensors as S8/S16.
    handles = [
        backbone.stages[1].register_forward_hook(capture("s8")),
        backbone.stages[2].register_forward_hook(capture("s16")),
    ]
    region_accumulator = defaultdict(lambda: {"numerator": 0.0, "denominator": 0.0})
    size_accumulator = defaultdict(lambda: {"numerator": 0.0, "denominator": 0.0})
    region_mass = defaultdict(float)
    processed_batches = 0
    accepted_samples = 0
    skipped_samples = 0
    total_s8_pixels = 0
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for samples, targets in cfg.train_dataloader:
            samples = samples.to(device)
            captured_features.clear()
            backbone(samples)
            if set(captured_features) != {"s8", "s16"}:
                raise RuntimeError("failed to capture internal S8 and S16 features")
            s8 = captured_features["s8"].float()
            s16 = captured_features["s16"].float()
            if s8.shape[-2] != 2 * s16.shape[-2] or s8.shape[-1] != 2 * s16.shape[-1]:
                raise RuntimeError(
                    f"unexpected S8/S16 shapes: {tuple(s8.shape)}, {tuple(s16.shape)}"
                )

            response8 = normalize_map(s8.square().mean(dim=1, keepdim=True).sqrt())
            response16 = normalize_map(s16.square().mean(dim=1, keepdim=True).sqrt())
            response16_up = F.interpolate(
                response16, size=s8.shape[-2:], mode="bilinear", align_corners=False
            )
            retention_error = (response8 - response16_up).abs()

            for index, target in enumerate(targets):
                masks = target.get("masks")
                if masks is None or masks.numel() == 0 or not bool(masks.any()):
                    skipped_samples += 1
                    continue
                masks = masks.float().to(device)
                union = masks.amax(dim=0, keepdim=True).unsqueeze(0)
                mask_s8 = F.interpolate(union, size=s8.shape[-2:], mode="area")

                dilated = F.max_pool2d(mask_s8, kernel_size=3, stride=1, padding=1)
                eroded = -F.max_pool2d(-mask_s8, kernel_size=3, stride=1, padding=1)
                boundary = (dilated - eroded).clamp(0.0, 1.0)

                mask_s16 = F.interpolate(mask_s8, size=s16.shape[-2:], mode="area")
                reconstructed = F.interpolate(
                    mask_s16,
                    size=s8.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                incoherence = (mask_s8 - reconstructed).abs().clamp(0.0, 1.0)
                sibr = (boundary * incoherence).clamp(0.0, 1.0)
                shifted = torch.roll(sibr, shifts=sibr.shape[-1] // 2, dims=-1)
                random_control = torch.roll(
                    sibr,
                    shifts=(sibr.shape[-2] // 3, sibr.shape[-1] // 4),
                    dims=(-2, -1),
                )
                interior = (mask_s8 - boundary).clamp(0.0, 1.0)
                background = (1.0 - dilated).clamp(0.0, 1.0)
                error = retention_error[index : index + 1]

                regions = {
                    "ordinary_boundary": boundary,
                    "sibr": sibr,
                    "shifted_sibr": shifted,
                    "random_sibr": random_control,
                    "interior": interior,
                    "background": background,
                    "whole_map": torch.ones_like(sibr),
                }
                for name, weights_map in regions.items():
                    weighted_accumulate(
                        region_accumulator, name, error, weights_map
                    )
                    region_mass[name] += float(weights_map.sum())

                sample_mask_area = float(union.sum())
                weighted_accumulate(
                    size_accumulator,
                    size_name(sample_mask_area),
                    error,
                    sibr,
                )
                accepted_samples += 1
                total_s8_pixels += int(sibr.shape[-2] * sibr.shape[-1])

            processed_batches += 1
            if processed_batches >= args.batches:
                break

    if accepted_samples == 0:
        raise RuntimeError("no accepted SAM masks were audited")
    for handle in handles:
        handle.remove()

    means = weighted_means(region_accumulator)
    size_means = weighted_means(size_accumulator)
    ratios = {
        "sibr_over_ordinary_boundary": means["sibr"] / means["ordinary_boundary"],
        "sibr_over_shifted": means["sibr"] / means["shifted_sibr"],
        "sibr_over_random": means["sibr"] / means["random_sibr"],
        "sibr_over_background": means["sibr"] / means["background"],
    }
    mass_fractions = {
        name: mass / max(total_s8_pixels, 1)
        for name, mass in region_mass.items()
    }
    gate_checks = {
        "sibr_more_informative_than_full_boundary": ratios[
            "sibr_over_ordinary_boundary"
        ]
        > 1.0,
        "correct_sibr_beats_shifted": ratios["sibr_over_shifted"] > 1.0,
        "correct_sibr_beats_random": ratios["sibr_over_random"] > 1.0,
        "sibr_is_sparse": mass_fractions["sibr"] < mass_fractions[
            "ordinary_boundary"
        ],
    }
    report = {
        "status": "PASS" if all(gate_checks.values()) else "FAIL",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("last_epoch", -1)),
        "weight_source": weight_source,
        "feature_capture": "internal HGNetv2 strides 8 and 16",
        "batches": processed_batches,
        "accepted_samples": accepted_samples,
        "skipped_samples": skipped_samples,
        "feature_shapes": {"s8": list(s8.shape), "s16": list(s16.shape)},
        "retention_error_means": means,
        "retention_error_by_object_size": size_means,
        "enrichment_ratios": ratios,
        "region_mass_fraction_of_s8": mass_fractions,
        "gate_checks": gate_checks,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
