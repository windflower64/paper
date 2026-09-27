#!/usr/bin/env python3
"""M-A4 proposal-set evidence audit with equal-capacity RGB/RGB-T rerankers.

Both base detectors and all boxes are frozen.  Two tiny MLPs receive the same
RGB candidate features; the RGB-T version additionally receives continuous
support features from the mapped top-100 Thermal proposal set.  The comparison
tests information value rather than a hand-picked fusion threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from analyze_ma2_detector_cache import cxcywh_to_xyxy, map_ir_xyxy, pairwise_iou
from run_ma2_explicit_matching import build_evaluator, evaluate_scores


TOPK = 100
HIDDEN = 16
EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
SEED = 20260831


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--test-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-output", type=Path, required=True)
    parser.add_argument("--val-modulus", type=int, default=5)
    parser.add_argument("--val-residue", type=int, default=0)
    parser.add_argument("--mismatch-offset", type=int, default=607)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_sequence(file_name):
    return Path(file_name).stem.rsplit("_", 1)[0]


def stable_bucket(value, modulus):
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % modulus


def normalized_targets(annotation_path):
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_info = {int(item["id"]): item for item in annotation["images"]}
    targets = defaultdict(list)
    for ann in annotation["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        image = image_info[int(ann["image_id"])]
        x, y, w, h = ann["bbox"]
        targets[int(ann["image_id"])].append(
            torch.tensor(
                [x / image["width"], y / image["height"], (x + w) / image["width"], (y + h) / image["height"]],
                dtype=torch.float32,
            )
        )
    return targets


def build_features(cache, thermal_order=None):
    """Return selected query indices, RGB features and added Thermal features."""
    visible_logits = cache["visible_logits"].squeeze(-1).float()
    visible_scores = visible_logits.sigmoid()
    visible_boxes = cxcywh_to_xyxy(cache["visible_boxes"].float()).clamp(0.0, 1.0)
    thermal_scores = cache["thermal_logits"].squeeze(-1).float().sigmoid()
    thermal_boxes = cache["thermal_boxes"].float()
    if thermal_order is not None:
        thermal_scores = thermal_scores[thermal_order]
        thermal_boxes = thermal_boxes[thermal_order]
    thermal_boxes = map_ir_xyxy(cxcywh_to_xyxy(thermal_boxes))

    topk_visible = min(TOPK, visible_scores.shape[1])
    _scores, query_indices = visible_scores.topk(topk_visible, dim=1)
    query_logits = visible_logits.gather(1, query_indices)
    query_boxes = visible_boxes.gather(1, query_indices[..., None].expand(-1, -1, 4))
    query_centers = (query_boxes[..., :2] + query_boxes[..., 2:]) / 2
    query_cxcywh = torch.cat(
        (query_centers, (query_boxes[..., 2:] - query_boxes[..., :2]).clamp_min(1e-6)),
        dim=-1,
    )
    ranks = torch.arange(topk_visible, dtype=torch.float32)[None, :, None]
    ranks = ranks.expand(query_logits.shape[0], -1, -1) / max(1, topk_visible - 1)
    rgb_features = torch.cat((query_logits[..., None], ranks, query_cxcywh), dim=-1)

    topk_thermal = min(TOPK, thermal_scores.shape[1])
    thermal_top_scores, thermal_top_indices = thermal_scores.topk(topk_thermal, dim=1)
    thermal_top_boxes = thermal_boxes.gather(
        1, thermal_top_indices[..., None].expand(-1, -1, 4)
    )
    thermal_centers = (thermal_top_boxes[..., :2] + thermal_top_boxes[..., 2:]) / 2
    distances = torch.linalg.vector_norm(
        query_centers[:, :, None, :] - thermal_centers[:, None, :, :], dim=-1
    )
    soft_support = []
    for sigma in (0.02, 0.05, 0.10):
        soft_support.append(
            (thermal_top_scores[:, None, :] * torch.exp(-0.5 * (distances / sigma) ** 2))
            .amax(dim=-1)
        )
    min_distance = distances.amin(dim=-1)
    local_support = []
    for radius in (0.03, 0.05, 0.10):
        local = thermal_top_scores[:, None, :].masked_fill(distances > radius, 0.0)
        local_support.append(local.amax(dim=-1))
    thermal_features = torch.stack(
        (*soft_support, min_distance, *local_support), dim=-1
    )
    return query_indices, rgb_features, thermal_features, visible_boxes


def build_labels(cache, query_indices, targets):
    boxes = cxcywh_to_xyxy(cache["visible_boxes"].float()).clamp(0.0, 1.0)
    labels = torch.zeros(query_indices.shape, dtype=torch.float32)
    for row, image_id in enumerate(cache["image_ids"].tolist()):
        gt_list = targets.get(int(image_id), [])
        if not gt_list:
            continue
        gt = torch.stack(gt_list)
        candidate = boxes[row, query_indices[row]]
        labels[row] = (pairwise_iou(candidate, gt).amax(dim=1) >= 0.5).float()
    return labels


class Reranker(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_reranker(
    name, train_features, train_labels, val_features, val_labels, device, epochs, patience
):
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_x = ((train_features - mean) / std).to(device)
    val_x = ((val_features - mean) / std).to(device)
    train_y = train_labels.to(device)
    val_y = val_labels.to(device)
    model = Reranker(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    positives = train_y.sum().item()
    negatives = train_y.numel() - positives
    pos_weight = torch.tensor([negatives / max(1.0, positives)], device=device)
    best = None
    best_epoch = -1
    best_loss = float("inf")
    stale = 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(train_x.shape[0], device=device)
        for start in range(0, train_x.shape[0], BATCH_SIZE):
            index = order[start : start + BATCH_SIZE]
            logits = model(train_x[index])
            loss = F.binary_cross_entropy_with_logits(
                logits, train_y[index], pos_weight=pos_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss = float(
                F.binary_cross_entropy_with_logits(
                    model(val_x), val_y, pos_weight=pos_weight
                )
            )
        print(f"{name} epoch={epoch:02d} val_bce={val_loss:.8f}", flush=True)
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best, strict=True)
    return model.cpu().eval(), mean, std, {
        "best_epoch_by_validation_bce": best_epoch,
        "best_validation_bce": best_loss,
        "train_samples": int(train_x.shape[0]),
        "validation_samples": int(val_x.shape[0]),
        "train_positive_fraction": float(train_y.mean()),
        "validation_positive_fraction": float(val_y.mean()),
        "input_features": int(train_x.shape[1]),
        "hidden": HIDDEN,
    }


def predict(model, mean, std, features):
    with torch.inference_mode():
        return model((features - mean) / std).sigmoid()


def scores_from_predictions(cache, query_indices, predictions):
    scores = torch.zeros_like(cache["visible_logits"].squeeze(-1).float())
    scores.scatter_(1, query_indices, predictions)
    return scores


def main():
    args = parse_args()
    if not 0 <= args.val_residue < args.val_modulus:
        raise ValueError("invalid validation residue")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cache = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    test_cache = torch.load(args.test_cache, map_location="cpu", weights_only=False)
    train_targets = normalized_targets(args.train_annotations)

    train_queries, train_rgb, train_thermal, _train_boxes = build_features(train_cache)
    train_labels = build_labels(train_cache, train_queries, train_targets)
    val_image_mask = torch.tensor(
        [
            stable_bucket(parse_sequence(file_name), args.val_modulus) == args.val_residue
            for file_name in train_cache["file_names"]
        ]
    )
    if not val_image_mask.any() or val_image_mask.all():
        raise RuntimeError("invalid sequence validation split")
    train_image_mask = ~val_image_mask
    split_info = {
        "train_images": int(train_image_mask.sum()),
        "validation_images": int(val_image_mask.sum()),
        "train_sequences": len(
            {parse_sequence(name) for name, keep in zip(train_cache["file_names"], train_image_mask) if keep}
        ),
        "validation_sequences": len(
            {parse_sequence(name) for name, keep in zip(train_cache["file_names"], val_image_mask) if keep}
        ),
        "val_modulus": args.val_modulus,
        "val_residue": args.val_residue,
    }

    rgb_model, rgb_mean, rgb_std, rgb_fit = fit_reranker(
        "rgb_only",
        train_rgb[train_image_mask].reshape(-1, train_rgb.shape[-1]),
        train_labels[train_image_mask].reshape(-1),
        train_rgb[val_image_mask].reshape(-1, train_rgb.shape[-1]),
        train_labels[val_image_mask].reshape(-1),
        device,
        args.epochs,
        args.patience,
    )
    rgbt_features = torch.cat((train_rgb, train_thermal), dim=-1)
    rgbt_model, rgbt_mean, rgbt_std, rgbt_fit = fit_reranker(
        "rgbt_proposal_set",
        rgbt_features[train_image_mask].reshape(-1, rgbt_features.shape[-1]),
        train_labels[train_image_mask].reshape(-1),
        rgbt_features[val_image_mask].reshape(-1, rgbt_features.shape[-1]),
        train_labels[val_image_mask].reshape(-1),
        device,
        args.epochs,
        args.patience,
    )

    test_queries, test_rgb, test_thermal, test_visible_boxes = build_features(test_cache)
    test_rgbt = torch.cat((test_rgb, test_thermal), dim=-1)
    rgb_scores = scores_from_predictions(
        test_cache, test_queries, predict(rgb_model, rgb_mean, rgb_std, test_rgb.reshape(-1, test_rgb.shape[-1])).reshape(test_queries.shape)
    )
    rgbt_scores = scores_from_predictions(
        test_cache, test_queries, predict(rgbt_model, rgbt_mean, rgbt_std, test_rgbt.reshape(-1, test_rgbt.shape[-1])).reshape(test_queries.shape)
    )
    mismatch_order = torch.arange(test_queries.shape[0]).roll(args.mismatch_offset)
    mismatch_queries, mismatch_rgb, mismatch_thermal, _ = build_features(
        test_cache, thermal_order=mismatch_order
    )
    if not torch.equal(test_queries, mismatch_queries) or not torch.equal(test_rgb, mismatch_rgb):
        raise RuntimeError("Thermal mismatch changed RGB candidate features")
    mismatch_features = torch.cat((mismatch_rgb, mismatch_thermal), dim=-1)
    mismatch_scores = scores_from_predictions(
        test_cache,
        mismatch_queries,
        predict(
            rgbt_model,
            rgbt_mean,
            rgbt_std,
            mismatch_features.reshape(-1, mismatch_features.shape[-1]),
        ).reshape(mismatch_queries.shape),
    )

    coco_gt = build_evaluator(args.test_annotations)
    indices = list(range(len(test_cache["image_ids"])))
    baseline_scores = test_cache["visible_logits"].squeeze(-1).float().sigmoid()
    metrics = {
        "visible_baseline": evaluate_scores(coco_gt, test_cache, test_visible_boxes, baseline_scores, indices),
        "equal_capacity_rgb_reranker": evaluate_scores(coco_gt, test_cache, test_visible_boxes, rgb_scores, indices),
        "rgbt_proposal_set_reranker": evaluate_scores(coco_gt, test_cache, test_visible_boxes, rgbt_scores, indices),
        "rgbt_global_mismatch": evaluate_scores(coco_gt, test_cache, test_visible_boxes, mismatch_scores, indices),
    }
    weights = {
        "schema": "ma4_proposal_set_reranker_weights_v1",
        "rgb_model": rgb_model.state_dict(),
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std,
        "rgbt_model": rgbt_model.state_dict(),
        "rgbt_mean": rgbt_mean,
        "rgbt_std": rgbt_std,
        "constants": {"topk": TOPK, "hidden": HIDDEN},
    }
    args.weights_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, args.weights_output)
    report = {
        "schema": "ma4_proposal_set_reranker_audit_v1",
        "purpose": "Does Thermal proposal-set evidence improve query correctness beyond equal-capacity RGB-only reranking?",
        "frozen_detector_protocol": True,
        "train_cache": str(args.train_cache.resolve()),
        "test_cache": str(args.test_cache.resolve()),
        "train_annotations": str(args.train_annotations.resolve()),
        "test_annotations": str(args.test_annotations.resolve()),
        "selection_protocol": {
            "validation_data": "deterministic sequence-grouped subset of train",
            "test_metrics_used_to_select_epoch_or_hyperparameters": False,
            "seed": SEED,
            "topk_per_modality": TOPK,
            "epochs_cap": args.epochs,
            "early_stopping_patience": args.patience,
            **split_info,
        },
        "rgb_only_fit": rgb_fit,
        "rgbt_fit": rgbt_fit,
        "metrics": metrics,
        "contrasts": {
            "rgb_reranker_minus_baseline_AP": metrics["equal_capacity_rgb_reranker"]["AP"] - metrics["visible_baseline"]["AP"],
            "rgbt_minus_rgb_reranker_AP": metrics["rgbt_proposal_set_reranker"]["AP"] - metrics["equal_capacity_rgb_reranker"]["AP"],
            "rgbt_minus_baseline_AP": metrics["rgbt_proposal_set_reranker"]["AP"] - metrics["visible_baseline"]["AP"],
            "rgbt_normal_minus_mismatch_AP": metrics["rgbt_proposal_set_reranker"]["AP"] - metrics["rgbt_global_mismatch"]["AP"],
        },
        "weights": str(args.weights_output.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
