"""Train a cache-level target-pair alignment gate for M-B1.

Two matched-capacity MLPs are compared:

* score_only sees candidate confidence and rank, but no coordinates;
* spatial sees native RGB/thermal boxes plus residuals from a train-fitted
  Visible-to-Thermal affine prior.

Both are trained on sequence-grouped train data.  Test labels are never used
for fitting, early stopping, normalization, affine estimation, or thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center = boxes[..., :2]
    half = boxes[..., 2:] * 0.5
    return torch.cat((center - half, center + half), dim=-1)


def iou_one_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area_a = (box[2:] - box[:2]).clamp_min(0).prod()
    area_b = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(dim=-1)
    return intersection / (area_a + area_b - intersection).clamp_min(1e-12)


def largest_thermal_box(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    boxes = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            x, y, width, height = map(float, fields[1:5])
        except ValueError:
            continue
        if 0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1:
            boxes.append(np.asarray([x, y, width, height], dtype=np.float32))
    return max(boxes, key=lambda box: float(box[2] * box[3])) if boxes else None


def load_rows(cache_path: Path, visible_coco: Path, thermal_labels: Path) -> list[dict]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    coco = json.loads(visible_coco.read_text(encoding="utf-8"))
    image_info = {int(item["id"]): item for item in coco["images"]}
    annotations: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if not annotation.get("iscrowd", 0):
            annotations[int(annotation["image_id"])].append(annotation)
    visible_scores = cache["visible_logits"].squeeze(-1).float().sigmoid()
    thermal_scores = cache["thermal_logits"].squeeze(-1).float().sigmoid()
    rows = []
    for index, image_id in enumerate(cache["image_ids"].tolist()):
        info = image_info[int(image_id)]
        candidates = annotations.get(int(image_id), [])
        visible_gt = None
        if candidates:
            annotation = max(candidates, key=lambda item: item["bbox"][2] * item["bbox"][3])
            x, y, width, height = map(float, annotation["bbox"])
            visible_gt = np.asarray(
                [
                    (x + 0.5 * width) / info["width"],
                    (y + 0.5 * height) / info["height"],
                    width / info["width"],
                    height / info["height"],
                ],
                dtype=np.float32,
            )
        stem = Path(info["file_name"]).stem
        thermal_gt = largest_thermal_box(thermal_labels / f"{stem}.txt")
        rows.append(
            {
                "stem": stem,
                "sequence": stem.rsplit("_", 1)[0],
                "visible_gt": visible_gt,
                "thermal_gt": thermal_gt,
                "visible_scores": visible_scores[index],
                "thermal_scores": thermal_scores[index],
                "visible_boxes": cache["visible_boxes"][index].float().clamp(0, 1),
                "thermal_boxes": cache["thermal_boxes"][index].float().clamp(0, 1),
            }
        )
    return rows


def fit_affine(rows: list[dict]) -> np.ndarray:
    visible = np.stack([row["visible_gt"][:2] for row in rows])
    thermal = np.stack([row["thermal_gt"][:2] for row in rows])
    design = np.concatenate((visible, np.ones((len(visible), 1))), axis=1)
    matrix, _, _, _ = np.linalg.lstsq(design, thermal, rcond=None)
    return matrix.astype(np.float32)


def apply_affine(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(points[..., :1])
    return torch.cat((points, ones), dim=-1) @ matrix


def sequence_is_validation(sequence: str, modulus: int, residue: int) -> bool:
    return zlib.crc32(sequence.encode("utf-8")) % modulus == residue


def candidate_view(row: dict, topk: int) -> dict:
    visible_order = row["visible_scores"].argsort(descending=True)[:topk]
    thermal_order = row["thermal_scores"].argsort(descending=True)[:topk]
    visible_boxes = row["visible_boxes"][visible_order]
    thermal_boxes = row["thermal_boxes"][thermal_order]
    visible_gt = torch.as_tensor(row["visible_gt"])
    thermal_gt = torch.as_tensor(row["thermal_gt"])
    visible_correct = iou_one_to_many(
        cxcywh_to_xyxy(visible_gt), cxcywh_to_xyxy(visible_boxes)
    ) >= 0.5
    thermal_correct = iou_one_to_many(
        cxcywh_to_xyxy(thermal_gt), cxcywh_to_xyxy(thermal_boxes)
    ) >= 0.5
    return {
        "visible_boxes": visible_boxes,
        "thermal_boxes": thermal_boxes,
        "visible_scores": row["visible_scores"][visible_order],
        "thermal_scores": row["thermal_scores"][thermal_order],
        "visible_correct": visible_correct,
        "thermal_correct": thermal_correct,
        "visible_gt": visible_gt,
        "thermal_gt": thermal_gt,
        "sequence": row["sequence"],
        "stem": row["stem"],
    }


def feature_grid(
    visible: dict,
    thermal: dict,
    affine: torch.Tensor,
    spatial: bool,
) -> torch.Tensor:
    rgb_boxes = visible["visible_boxes"]
    ir_boxes = thermal["thermal_boxes"]
    rgb_scores = visible["visible_scores"]
    ir_scores = thermal["thermal_scores"]
    k_rgb, k_ir = len(rgb_boxes), len(ir_boxes)
    rgb_rank = torch.linspace(0, 1, k_rgb).view(k_rgb, 1).expand(k_rgb, k_ir)
    ir_rank = torch.linspace(0, 1, k_ir).view(1, k_ir).expand(k_rgb, k_ir)
    rgb_score_grid = rgb_scores.view(k_rgb, 1).expand(k_rgb, k_ir)
    ir_score_grid = ir_scores.view(1, k_ir).expand(k_rgb, k_ir)
    score_features = torch.stack(
        (
            rgb_score_grid,
            ir_score_grid,
            rgb_score_grid * ir_score_grid,
            rgb_score_grid - ir_score_grid,
            rgb_rank,
            ir_rank,
        ),
        dim=-1,
    )
    if not spatial:
        return score_features.reshape(k_rgb * k_ir, -1)

    mapped_rgb = apply_affine(rgb_boxes[:, :2], affine)
    delta = ir_boxes[None, :, :2] - mapped_rgb[:, None, :]
    abs_delta = delta.abs()
    distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    rgb_box_grid = rgb_boxes[:, None, :].expand(k_rgb, k_ir, 4)
    ir_box_grid = ir_boxes[None, :, :].expand(k_rgb, k_ir, 4)
    eps = 1e-5
    log_size_ratio = torch.log((ir_box_grid[..., 2:] + eps) / (rgb_box_grid[..., 2:] + eps))
    spatial_features = torch.cat(
        (
            score_features,
            rgb_box_grid,
            ir_box_grid,
            mapped_rgb[:, None, :].expand(k_rgb, k_ir, 2),
            delta,
            abs_delta,
            distance,
            log_size_ratio,
        ),
        dim=-1,
    )
    return spatial_features.reshape(k_rgb * k_ir, -1)


def different_sequence_roll(views: list[dict], shift: int) -> list[dict]:
    result = []
    size = len(views)
    for index, view in enumerate(views):
        candidate = (index + shift) % size
        attempts = 0
        while views[candidate]["sequence"] == view["sequence"] and attempts < size:
            candidate = (candidate + 1) % size
            attempts += 1
        result.append(views[candidate])
    return result


def build_dataset(
    views: list[dict], affine: torch.Tensor, topk: int, spatial: bool, mismatch_shift: int
) -> tuple[torch.Tensor, torch.Tensor]:
    mismatched = different_sequence_roll(views, mismatch_shift)
    features = []
    labels = []
    for visible, true_thermal, wrong_thermal in zip(views, views, mismatched):
        true_features = feature_grid(visible, true_thermal, affine, spatial)
        true_labels = (
            visible["visible_correct"][:, None] & true_thermal["thermal_correct"][None, :]
        ).reshape(-1)
        wrong_features = feature_grid(visible, wrong_thermal, affine, spatial)
        features.extend((true_features, wrong_features))
        labels.extend((true_labels.float(), torch.zeros(len(wrong_features))))
    return torch.cat(features), torch.cat(labels)


class PairMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    rank_sum = ranks[labels].sum()
    return float((rank_sum - positive * (positive + 1) / 2) / (positive * negative))


def choose_threshold(true_scores: np.ndarray, mismatch_scores: np.ndarray) -> dict:
    candidates = np.unique(
        np.quantile(np.concatenate((true_scores, mismatch_scores)), np.linspace(0, 1, 1001))
    )
    best = None
    for threshold in candidates:
        accept = float((true_scores >= threshold).mean())
        reject = float((mismatch_scores < threshold).mean())
        balanced = 0.5 * (accept + reject)
        candidate = (balanced, reject, float(threshold))
        if best is None or candidate > best[0]:
            best = (candidate, float(threshold), accept, reject)
    assert best is not None
    return {
        "threshold": best[1],
        "balanced_accuracy": best[0][0],
        "true_accept_rate": best[2],
        "mismatch_reject_rate": best[3],
    }


def train_model(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    hidden: int,
    epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[PairMLP, dict, torch.Tensor, torch.Tensor]:
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0).clamp_min(1e-5)
    train_x = (train_features - mean) / std
    val_x = (val_features - mean) / std
    model = PairMLP(train_x.shape[1], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    positive = float(train_labels.sum())
    negative = float(len(train_labels) - positive)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative / max(positive, 1.0), device=device)
    )
    generator = torch.Generator().manual_seed(20260831)
    best_state = None
    best_epoch = -1
    best_val = math.inf
    remaining = patience
    batch_size = 4096
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            logits = model(train_x[indices].to(device))
            loss = criterion(logits, train_labels[indices].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x.to(device))
            val_loss = float(
                nn.functional.binary_cross_entropy_with_logits(
                    val_logits, val_labels.to(device)
                ).cpu()
            )
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            remaining = patience
        else:
            remaining -= 1
            if remaining <= 0:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_bce": best_val,
        "train_pairs": len(train_labels),
        "train_positive_fraction": float(train_labels.mean()),
        "input_features": train_x.shape[1],
        "hidden": hidden,
    }, mean, std


def evaluate_images(
    model: PairMLP,
    views: list[dict],
    affine: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    topk: int,
    spatial: bool,
    mismatch_shift: int,
    device: torch.device,
) -> dict:
    mismatched = different_sequence_roll(views, mismatch_shift)
    true_max_scores = []
    mismatch_max_scores = []
    selected_visible_correct = []
    selected_thermal_correct = []
    selected_both_correct = []
    thermal_center_errors = []
    all_pair_scores = []
    all_pair_labels = []
    model.eval()
    with torch.no_grad():
        for visible, true_thermal, wrong_thermal in zip(views, views, mismatched):
            true_features = feature_grid(visible, true_thermal, affine, spatial)
            mismatch_features = feature_grid(visible, wrong_thermal, affine, spatial)
            true_logits = model(((true_features - mean) / std).to(device)).cpu()
            mismatch_logits = model(((mismatch_features - mean) / std).to(device)).cpu()
            true_scores = true_logits.sigmoid()
            mismatch_scores = mismatch_logits.sigmoid()
            true_max_scores.append(float(true_scores.max()))
            mismatch_max_scores.append(float(mismatch_scores.max()))
            selected = int(true_scores.argmax())
            rgb_index = selected // topk
            ir_index = selected % topk
            rgb_ok = bool(visible["visible_correct"][rgb_index])
            ir_ok = bool(true_thermal["thermal_correct"][ir_index])
            selected_visible_correct.append(rgb_ok)
            selected_thermal_correct.append(ir_ok)
            selected_both_correct.append(rgb_ok and ir_ok)
            thermal_center_errors.append(
                float(
                    torch.linalg.vector_norm(
                        true_thermal["thermal_boxes"][ir_index, :2]
                        - true_thermal["thermal_gt"][:2]
                    )
                )
            )
            labels = (
                visible["visible_correct"][:, None]
                & true_thermal["thermal_correct"][None, :]
            ).reshape(-1)
            all_pair_scores.extend(true_scores.tolist())
            all_pair_labels.extend(labels.tolist())
            all_pair_scores.extend(mismatch_scores.tolist())
            all_pair_labels.extend([False] * len(mismatch_scores))
    true_array = np.asarray(true_max_scores)
    mismatch_array = np.asarray(mismatch_max_scores)
    center_array = np.asarray(thermal_center_errors)
    return {
        "images": len(views),
        "selected_visible_correct_fraction": float(np.mean(selected_visible_correct)),
        "selected_thermal_correct_fraction": float(np.mean(selected_thermal_correct)),
        "selected_both_correct_fraction": float(np.mean(selected_both_correct)),
        "selected_thermal_center_error": {
            "mean": float(center_array.mean()),
            "p50": float(np.quantile(center_array, 0.5)),
            "p90": float(np.quantile(center_array, 0.9)),
        },
        "pair_classification_auc": binary_auc(
            np.asarray(all_pair_labels), np.asarray(all_pair_scores)
        ),
        "image_pair_verification_auc": binary_auc(
            np.concatenate((np.ones(len(true_array), dtype=bool), np.zeros(len(mismatch_array), dtype=bool))),
            np.concatenate((true_array, mismatch_array)),
        ),
        "true_max_scores": true_array,
        "mismatch_max_scores": mismatch_array,
    }


def threshold_metrics(result: dict, threshold: float) -> dict:
    true_scores = result["true_max_scores"]
    mismatch_scores = result["mismatch_max_scores"]
    accept = float((true_scores >= threshold).mean())
    reject = float((mismatch_scores < threshold).mean())
    return {
        "threshold_from_train_validation": threshold,
        "balanced_accuracy": 0.5 * (accept + reject),
        "true_accept_rate": accept,
        "mismatch_reject_rate": reject,
    }


def clean_result(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in {"true_max_scores", "mismatch_max_scores"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--visible-train-coco", type=Path, required=True)
    parser.add_argument("--visible-test-coco", type=Path, required=True)
    parser.add_argument("--thermal-train-labels", type=Path, required=True)
    parser.add_argument("--thermal-test-labels", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val-modulus", type=int, default=5)
    parser.add_argument("--val-residue", type=int, default=0)
    parser.add_argument("--mismatch-shift", type=int, default=607)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_rows = load_rows(args.train_cache, args.visible_train_coco, args.thermal_train_labels)
    test_rows = load_rows(args.test_cache, args.visible_test_coco, args.thermal_test_labels)
    paired_train = [row for row in train_rows if row["visible_gt"] is not None and row["thermal_gt"] is not None]
    paired_test = [row for row in test_rows if row["visible_gt"] is not None and row["thermal_gt"] is not None]
    fit_rows = [
        row
        for row in paired_train
        if not sequence_is_validation(row["sequence"], args.val_modulus, args.val_residue)
    ]
    val_rows = [
        row
        for row in paired_train
        if sequence_is_validation(row["sequence"], args.val_modulus, args.val_residue)
    ]
    affine_np = fit_affine(fit_rows)
    affine = torch.as_tensor(affine_np)
    fit_views = [candidate_view(row, args.topk) for row in fit_rows]
    val_views = [candidate_view(row, args.topk) for row in val_rows]
    test_views = [candidate_view(row, args.topk) for row in paired_test]

    results = {}
    saved = {"affine": affine, "topk": args.topk, "models": {}}
    for name, spatial in (("score_only", False), ("spatial_alignment", True)):
        fit_x, fit_y = build_dataset(
            fit_views, affine, args.topk, spatial, args.mismatch_shift
        )
        val_x, val_y = build_dataset(
            val_views, affine, args.topk, spatial, args.mismatch_shift
        )
        model, fit_report, mean, std = train_model(
            fit_x,
            fit_y,
            val_x,
            val_y,
            args.hidden,
            args.epochs,
            args.patience,
            device,
        )
        validation = evaluate_images(
            model,
            val_views,
            affine,
            mean,
            std,
            args.topk,
            spatial,
            args.mismatch_shift,
            device,
        )
        threshold = choose_threshold(
            validation["true_max_scores"], validation["mismatch_max_scores"]
        )
        test = evaluate_images(
            model,
            test_views,
            affine,
            mean,
            std,
            args.topk,
            spatial,
            args.mismatch_shift,
            device,
        )
        results[name] = {
            "fit": fit_report,
            "validation": {
                **clean_result(validation),
                "selected_pair_threshold": threshold,
            },
            "test": {
                **clean_result(test),
                "threshold_metrics": threshold_metrics(test, threshold["threshold"]),
            },
        }
        saved["models"][name] = {
            "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "mean": mean,
            "std": std,
            "spatial": spatial,
            "input_features": fit_x.shape[1],
            "threshold": threshold["threshold"],
        }

    score_test = results["score_only"]["test"]
    spatial_test = results["spatial_alignment"]["test"]
    verification_gain = (
        spatial_test["image_pair_verification_auc"]
        - score_test["image_pair_verification_auc"]
    )
    selection_gain = (
        spatial_test["selected_both_correct_fraction"]
        - score_test["selected_both_correct_fraction"]
    )
    passed = (
        spatial_test["image_pair_verification_auc"] >= 0.80
        and verification_gain >= 0.10
        and spatial_test["selected_both_correct_fraction"] >= 0.75
    )
    report = {
        "schema": "mb1_target_alignment_p1_v1",
        "status": "PASS" if passed else "FAIL",
        "purpose_zh": "验证只有看到空间信息的目标级对齐器能否区分正确配对与错配。",
        "protocol": {
            "train_cache": str(args.train_cache.resolve()),
            "test_cache": str(args.test_cache.resolve()),
            "topk_per_modality": args.topk,
            "train_paired_images": len(paired_train),
            "fit_images": len(fit_rows),
            "validation_images": len(val_rows),
            "test_images": len(paired_test),
            "sequence_grouped_validation": True,
            "test_used_for_training_selection_or_threshold": False,
            "mismatch_requires_different_sequence": True,
            "device": str(device),
            "seed": args.seed,
        },
        "train_fitted_visible_to_thermal_affine": affine_np.tolist(),
        "results": results,
        "contrasts": {
            "spatial_minus_score_only_pair_verification_auc": verification_gain,
            "spatial_minus_score_only_selected_both_correct_fraction": selection_gain,
        },
        "pass_rule": {
            "spatial_pair_verification_auc_min": 0.80,
            "spatial_auc_gain_over_score_only_min": 0.10,
            "spatial_selected_both_correct_fraction_min": 0.75,
        },
        "decision_zh": (
            "空间P1通过后，下一步才允许把对齐置信度接入四状态路由和有界融合；"
            "若失败，则缓存中的框/分数不足，必须缓存S8/S16局部特征训练可变形对齐。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(saved, args.weights)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
