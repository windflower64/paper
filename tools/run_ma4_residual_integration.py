#!/usr/bin/env python3
"""Integrate frozen M-A4 proposal evidence as a train-selected residual only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from run_ma2_explicit_matching import build_evaluator, evaluate_scores
from run_ma4_proposal_set_reranker import Reranker, build_features, predict


ALPHA_GRID = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--test-annotations", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-modulus", type=int, default=5)
    parser.add_argument("--val-residue", type=int, default=0)
    parser.add_argument("--mismatch-offset", type=int, default=607)
    return parser.parse_args()


def parse_sequence(file_name):
    return Path(file_name).stem.rsplit("_", 1)[0]


def stable_bucket(value, modulus):
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % modulus


def load_models(weights_path):
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    hidden = int(weights["constants"]["hidden"])
    rgb_model = Reranker(int(weights["rgb_mean"].numel()))
    rgbt_model = Reranker(int(weights["rgbt_mean"].numel()))
    if hidden != 16:
        raise RuntimeError(f"Unexpected hidden width {hidden}; schema must be updated")
    rgb_model.load_state_dict(weights["rgb_model"], strict=True)
    rgbt_model.load_state_dict(weights["rgbt_model"], strict=True)
    return (
        rgb_model.eval(),
        weights["rgb_mean"],
        weights["rgb_std"],
        rgbt_model.eval(),
        weights["rgbt_mean"],
        weights["rgbt_std"],
    )


def residual_delta(rgb_model, rgb_mean, rgb_std, rgbt_model, rgbt_mean, rgbt_std, rgb, thermal):
    rgb_flat = rgb.reshape(-1, rgb.shape[-1])
    rgbt = torch.cat((rgb, thermal), dim=-1)
    rgbt_flat = rgbt.reshape(-1, rgbt.shape[-1])
    rgb_probability = predict(rgb_model, rgb_mean, rgb_std, rgb_flat).reshape(rgb.shape[:2])
    rgbt_probability = predict(rgbt_model, rgbt_mean, rgbt_std, rgbt_flat).reshape(rgb.shape[:2])
    # Difference of correctness logits isolates what the thermal feature set
    # adds beyond the equal-capacity RGB-only head.
    delta = torch.logit(rgbt_probability.clamp(1e-6, 1 - 1e-6)) - torch.logit(
        rgb_probability.clamp(1e-6, 1 - 1e-6)
    )
    return delta, rgb_probability, rgbt_probability


def integrate(cache, query_indices, delta, alpha):
    logits = cache["visible_logits"].squeeze(-1).float().clone()
    original = logits.gather(1, query_indices)
    logits.scatter_(1, query_indices, original + alpha * delta)
    return logits.sigmoid()


def main():
    args = parse_args()
    if not 0 <= args.val_residue < args.val_modulus:
        raise ValueError("invalid validation split")
    (
        rgb_model,
        rgb_mean,
        rgb_std,
        rgbt_model,
        rgbt_mean,
        rgbt_std,
    ) = load_models(args.weights)
    train_cache = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    test_cache = torch.load(args.test_cache, map_location="cpu", weights_only=False)
    train_queries, train_rgb, train_thermal, train_boxes = build_features(train_cache)
    train_delta, _rgb_prob, _rgbt_prob = residual_delta(
        rgb_model, rgb_mean, rgb_std, rgbt_model, rgbt_mean, rgbt_std, train_rgb, train_thermal
    )
    validation_indices = [
        index
        for index, name in enumerate(train_cache["file_names"])
        if stable_bucket(parse_sequence(name), args.val_modulus) == args.val_residue
    ]
    if not validation_indices:
        raise RuntimeError("empty validation subset")
    train_coco = build_evaluator(args.train_annotations)
    calibration = []
    for alpha in ALPHA_GRID:
        scores = integrate(train_cache, train_queries, train_delta, alpha)
        metrics = evaluate_scores(
            train_coco, train_cache, train_boxes, scores, validation_indices
        )
        calibration.append({"alpha": alpha, "metrics": metrics})
        print(f"alpha={alpha:.3f} val_AP={metrics['AP']:.9f}", flush=True)
    # Exact AP ties select the smallest magnitude residual.
    chosen = max(calibration, key=lambda item: (item["metrics"]["AP"], -item["alpha"]))

    test_queries, test_rgb, test_thermal, test_boxes = build_features(test_cache)
    test_delta, _rgb_prob, _rgbt_prob = residual_delta(
        rgb_model, rgb_mean, rgb_std, rgbt_model, rgbt_mean, rgbt_std, test_rgb, test_thermal
    )
    mismatch_order = torch.arange(len(test_cache["image_ids"])).roll(args.mismatch_offset)
    mismatch_queries, mismatch_rgb, mismatch_thermal, _ = build_features(
        test_cache, thermal_order=mismatch_order
    )
    if not torch.equal(test_queries, mismatch_queries) or not torch.equal(test_rgb, mismatch_rgb):
        raise RuntimeError("Thermal mismatch changed RGB features")
    mismatch_delta, _rgb_prob, _rgbt_prob = residual_delta(
        rgb_model,
        rgb_mean,
        rgb_std,
        rgbt_model,
        rgbt_mean,
        rgbt_std,
        mismatch_rgb,
        mismatch_thermal,
    )
    test_coco = build_evaluator(args.test_annotations)
    indices = list(range(len(test_cache["image_ids"])))
    baseline = test_cache["visible_logits"].squeeze(-1).float().sigmoid()
    normal = integrate(test_cache, test_queries, test_delta, chosen["alpha"])
    mismatch = integrate(test_cache, mismatch_queries, mismatch_delta, chosen["alpha"])
    metrics = {
        "visible_baseline": evaluate_scores(test_coco, test_cache, test_boxes, baseline, indices),
        "ma4_residual_normal": evaluate_scores(test_coco, test_cache, test_boxes, normal, indices),
        "ma4_residual_global_mismatch": evaluate_scores(test_coco, test_cache, test_boxes, mismatch, indices),
    }
    report = {
        "schema": "ma4_residual_integration_audit_v1",
        "purpose": "Use proposal-set Thermal evidence only as a train-selected residual on mature RGB scores.",
        "weights": str(args.weights.resolve()),
        "selection_protocol": {
            "alpha_grid": list(ALPHA_GRID),
            "validation_data": "deterministic sequence-grouped subset of train",
            "validation_images": len(validation_indices),
            "test_metrics_used_to_select_alpha": False,
            "val_modulus": args.val_modulus,
            "val_residue": args.val_residue,
        },
        "validation_calibration": calibration,
        "chosen": chosen,
        "metrics": metrics,
        "contrasts": {
            "normal_minus_baseline_AP": metrics["ma4_residual_normal"]["AP"] - metrics["visible_baseline"]["AP"],
            "normal_minus_mismatch_AP": metrics["ma4_residual_normal"]["AP"] - metrics["ma4_residual_global_mismatch"]["AP"],
        },
        "test_delta_diagnostics": {
            "normal_mean_abs": float(test_delta.abs().mean()),
            "normal_p95_abs": float(torch.quantile(test_delta.abs(), 0.95)),
            "mismatch_mean_abs": float(mismatch_delta.abs().mean()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
