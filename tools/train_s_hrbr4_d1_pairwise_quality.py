#!/usr/bin/env python3
"""S-HRBR4-D1：冻结A00与HRBR1，训练成对ROI质量选择器。"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=Path("E:/two_paper/outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth"),
    )
    parser.add_argument(
        "--refiner-checkpoint",
        type=Path,
        default=Path("E:/two_paper/outputs/S_HRBR1_REFINEBOX_OFFICIAL_FPN_SEED0/best.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "E:/two_paper/reports/24_high_resolution_box_refinement/"
            "S_HRBR4_D1_PAIRWISE_QUALITY_SEED0"
        ),
    )
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--head-epochs", type=int, default=80)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    return parser.parse_args()


class PairwiseQualitySelector(nn.Module):
    """共享候选编码器 + 质量辅助头 + 成对相对优劣头。"""

    def __init__(self, roi_dim: int = 64, geometry_dim: int = 7) -> None:
        super().__init__()
        candidate_dim = roi_dim + geometry_dim
        self.encoder = nn.Sequential(
            nn.Linear(candidate_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 64),
            nn.GELU(),
        )
        self.quality = nn.Linear(64, 1)
        pair_dim = 64 * 4 + geometry_dim * 2
        self.pair = nn.Sequential(
            nn.Linear(pair_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, base_roi, refined_roi, base_geometry, refined_geometry):
        base_encoded = self.encoder(torch.cat((base_roi, base_geometry), dim=-1))
        refined_encoded = self.encoder(torch.cat((refined_roi, refined_geometry), dim=-1))
        quality_base = torch.sigmoid(self.quality(base_encoded)).squeeze(-1)
        quality_refined = torch.sigmoid(self.quality(refined_encoded)).squeeze(-1)
        pair_input = torch.cat(
            (
                base_encoded,
                refined_encoded,
                refined_encoded - base_encoded,
                (refined_encoded - base_encoded).abs(),
                base_geometry,
                refined_geometry,
            ),
            dim=-1,
        )
        benefit_logit = self.pair(pair_input).squeeze(-1)
        return benefit_logit, quality_base, quality_refined


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_metadata_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise RuntimeError(f"没有元数据记录可写入: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2), dim=-1
    )


def aligned_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes_a[..., :2], boxes_b[..., :2])
    bottom_right = torch.minimum(boxes_a[..., 2:], boxes_b[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(-1)
    area_a = (boxes_a[..., 2:] - boxes_a[..., :2]).clamp_min(0).prod(-1)
    area_b = (boxes_b[..., 2:] - boxes_b[..., :2]).clamp_min(0).prod(-1)
    return intersection / (area_a + area_b - intersection).clamp_min(1e-9)


def normalize_targets_cxcywh(targets, image_height: int, image_width: int):
    normalized = []
    scale = torch.tensor(
        [image_width, image_height, image_width, image_height],
        dtype=torch.float32,
        device=targets[0]["boxes"].device,
    )
    for target in targets:
        source = target["boxes"]
        boxes = source.as_subclass(torch.Tensor).float()
        format_name = str(getattr(source, "format", "")).lower()
        max_value = float(boxes.max()) if boxes.numel() else 0.0
        if "xyxy" in format_name or (not format_name and max_value > 1.5):
            x1, y1, x2, y2 = boxes.unbind(-1)
            boxes = torch.stack(
                ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1
            )
        if boxes.numel() and float(boxes.max()) > 1.5:
            boxes = boxes / scale
        converted = dict(target)
        converted["boxes"] = boxes
        normalized.append(converted)
    return normalized


def load_frozen_models(args, cfg):
    from train_s_hrbr1_refinebox import BackboneFeatureTap, RefineBoxHead, load_frozen_detector

    detector, weight_source = load_frozen_detector(cfg, args.detector_checkpoint)
    tap = BackboneFeatureTap(detector.backbone)
    channels = tuple(detector.backbone._out_channels[index] for index in (0, 1, 2, 3))
    refiner = RefineBoxHead(channels, d_model=64, roi_size=7, refine_steps=3).cuda()
    saved = torch.load(args.refiner_checkpoint, map_location="cpu", weights_only=False)
    refiner.load_state_dict(saved.get("refiner", saved), strict=True)
    refiner.force_identity = False
    refiner.eval().requires_grad_(False)
    return detector, refiner, tap, weight_source


@torch.inference_mode()
def forward_candidates(detector, refiner, tap, samples, topk: int, precision: str):
    tap.clear()
    with torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16"):
        outputs = detector(samples)
    backbone_features = tuple(feature.float() for feature in tap.features())
    pyramid = refiner.fpn(backbone_features)
    probabilities = outputs["pred_logits"].float().sigmoid()
    query_scores, query_labels = probabilities.max(-1)
    selected = query_scores.topk(min(topk, query_scores.shape[1]), dim=1).indices
    batch_ids = torch.arange(samples.shape[0], device=samples.device)[:, None]
    base_boxes = outputs["pred_boxes"].float()[batch_ids, selected]
    flat_batch = batch_ids.expand_as(selected).flatten()
    refined_flat = base_boxes.flatten(0, 1)
    for _ in range(refiner.refine_steps):
        refined_flat = refiner.refine_once(pyramid, refined_flat, flat_batch)
    refined_boxes = refined_flat.view_as(base_boxes)

    def roi_vectors(boxes):
        pooled = refiner.pool(pyramid, boxes.flatten(0, 1), flat_batch)
        encoded = refiner.residual(pooled)
        return F.adaptive_avg_pool2d(encoded, 1).flatten(1).view(
            samples.shape[0], selected.shape[1], -1
        )

    base_roi = roi_vectors(base_boxes)
    refined_roi = roi_vectors(refined_boxes)
    scores = query_scores.gather(1, selected)
    labels = query_labels.gather(1, selected)
    if probabilities.shape[-1] > 1:
        top2 = probabilities.topk(2, dim=-1).values
        margins = (top2[..., 0] - top2[..., 1]).gather(1, selected)
    else:
        margins = scores
    count = selected.shape[1]
    rank = torch.arange(count, device=samples.device, dtype=torch.float32)
    rank = rank.view(1, count).expand(samples.shape[0], -1) / max(1, count - 1)
    base_geometry = torch.cat((base_boxes, scores[..., None], margins[..., None], rank[..., None]), -1)
    refined_geometry = torch.cat(
        (refined_boxes, scores[..., None], margins[..., None], rank[..., None]), -1
    )
    return {
        "outputs": outputs,
        "selected": selected,
        "batch_ids": batch_ids,
        "base_boxes": base_boxes,
        "refined_boxes": refined_boxes,
        "base_roi": base_roi,
        "refined_roi": refined_roi,
        "base_geometry": base_geometry,
        "refined_geometry": refined_geometry,
        "labels": labels,
        "image_height": int(backbone_features[0].shape[-2] * 4),
        "image_width": int(backbone_features[0].shape[-1] * 4),
    }


def matched_positions(bundle, targets, matcher):
    outputs = bundle["outputs"]
    selected = bundle["selected"]
    normalized_targets = normalize_targets_cxcywh(
        targets, bundle["image_height"], bundle["image_width"]
    )
    core = {
        "pred_logits": outputs["pred_logits"].float(),
        "pred_boxes": outputs["pred_boxes"].float(),
    }
    matches = matcher(core, normalized_targets)["indices"]
    result = []
    for image_index, ((query_ids, target_ids), target) in enumerate(
        zip(matches, normalized_targets)
    ):
        positions = {int(query): position for position, query in enumerate(selected[image_index].tolist())}
        for query_tensor, target_tensor in zip(query_ids, target_ids):
            query_id = int(query_tensor.item())
            target_id = int(target_tensor.item())
            if query_id not in positions:
                continue
            position = positions[query_id]
            predicted_label = int(bundle["labels"][image_index, position].item())
            target_label = int(target["labels"][target_id].item())
            if predicted_label != target_label:
                continue
            truth = target["boxes"][target_id]
            before = float(
                aligned_iou(
                    box_cxcywh_to_xyxy(bundle["base_boxes"][image_index, position]),
                    box_cxcywh_to_xyxy(truth),
                ).item()
            )
            after = float(
                aligned_iou(
                    box_cxcywh_to_xyxy(bundle["refined_boxes"][image_index, position]),
                    box_cxcywh_to_xyxy(truth),
                ).item()
            )
            result.append(
                {
                    "image_index": image_index,
                    "position": position,
                    "query_id": query_id,
                    "target_id": target_id,
                    "image_id": int(target["image_id"].item()),
                    "iou_before": before,
                    "iou_after": after,
                    "delta_iou": after - before,
                    "beneficial": int(after > before),
                }
            )
    return result


@torch.inference_mode()
def extract_training_cache(detector, refiner, tap, loader, matcher, args):
    from train_s_hrbr1_refinebox import move_targets

    arrays = {name: [] for name in ("base_roi", "refined_roi", "base_geometry", "refined_geometry")}
    metadata = []
    started = time.time()
    for batch_index, (samples, targets_cpu) in enumerate(loader):
        if args.max_train_batches is not None and batch_index >= args.max_train_batches:
            break
        samples = samples.cuda(non_blocking=True)
        targets = move_targets(targets_cpu, "cuda")
        bundle = forward_candidates(detector, refiner, tap, samples, args.topk, args.precision)
        positions = matched_positions(bundle, targets, matcher)
        for item in positions:
            image_index, position = item["image_index"], item["position"]
            for name in arrays:
                arrays[name].append(bundle[name][image_index, position].detach().cpu())
            metadata.append({key: value for key, value in item.items() if key not in ("image_index", "position")})
        if batch_index == 0 or (batch_index + 1) % 20 == 0:
            print(
                f"extract=train batch={batch_index + 1:03d}/{len(loader)} "
                f"pairs={len(metadata)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    cache = {name: torch.stack(values) for name, values in arrays.items()}
    cache["iou_before"] = torch.tensor([item["iou_before"] for item in metadata], dtype=torch.float32)
    cache["iou_after"] = torch.tensor([item["iou_after"] for item in metadata], dtype=torch.float32)
    cache["beneficial"] = torch.tensor([item["beneficial"] for item in metadata], dtype=torch.float32)
    cache["image_id"] = torch.tensor([item["image_id"] for item in metadata], dtype=torch.long)
    return cache, metadata


def selector_forward(selector, cache, indices, device):
    values = [cache[name][indices].to(device) for name in (
        "base_roi", "refined_roi", "base_geometry", "refined_geometry"
    )]
    return selector(*values)


@torch.inference_mode()
def predict_cache(selector, cache, indices, device):
    selector.eval()
    logits, quality_base, quality_refined = selector_forward(selector, cache, indices, device)
    return (
        torch.sigmoid(logits).cpu().numpy(),
        quality_base.cpu().numpy(),
        quality_refined.cpu().numpy(),
    )


def train_selector(cache, args, output_dir: Path):
    device = torch.device("cuda")
    image_ids = cache["image_id"].numpy()
    hold_mask = (image_ids % 5) == 0
    if hold_mask.sum() < 100 or (~hold_mask).sum() < 500:
        raise RuntimeError(f"内部划分过小: fit={(~hold_mask).sum()} hold={hold_mask.sum()}")
    fit_indices = torch.from_numpy(np.flatnonzero(~hold_mask)).long()
    hold_indices = torch.from_numpy(np.flatnonzero(hold_mask)).long()
    selector = PairwiseQualitySelector().to(device)
    trainable = sum(parameter.numel() for parameter in selector.parameters())
    labels_fit = cache["beneficial"][fit_indices]
    positives = float(labels_fit.sum())
    negatives = float(len(labels_fit) - positives)
    pos_weight = torch.tensor(negatives / max(1.0, positives), device=device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.head_epochs)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    best_state = None
    best_key = (-float("inf"), -float("inf"))
    delta_hold = (
        cache["iou_after"][hold_indices] - cache["iou_before"][hold_indices]
    ).numpy()
    labels_hold = cache["beneficial"][hold_indices].numpy().astype(np.int64)
    for epoch in range(args.head_epochs):
        selector.train()
        permutation = fit_indices[torch.randperm(len(fit_indices), generator=generator)]
        running = 0.0
        count = 0
        for start in range(0, len(permutation), args.head_batch_size):
            indices = permutation[start : start + args.head_batch_size]
            logits, quality_base, quality_refined = selector_forward(selector, cache, indices, device)
            target_label = cache["beneficial"][indices].to(device)
            target_before = cache["iou_before"][indices].to(device)
            target_after = cache["iou_after"][indices].to(device)
            classification = F.binary_cross_entropy_with_logits(
                logits, target_label, pos_weight=pos_weight
            )
            quality = F.smooth_l1_loss(quality_base, target_before) + F.smooth_l1_loss(
                quality_refined, target_after
            )
            gain = F.smooth_l1_loss(
                quality_refined - quality_base, target_after - target_before
            )
            loss = classification + 2.0 * quality + 10.0 * gain
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(indices)
            count += len(indices)
        scheduler.step()
        probabilities, _, _ = predict_cache(selector, cache, hold_indices, device)
        auc = float(roc_auc_score(labels_hold, probabilities))
        gated_delta = float((delta_hold * (probabilities >= 0.5)).mean())
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": running / max(1, count),
            "hold_auc": auc,
            "hold_average_precision": float(average_precision_score(labels_hold, probabilities)),
            "hold_gate_coverage_at_0_5": float((probabilities >= 0.5).mean()),
            "hold_gated_mean_delta_at_0_5": gated_delta,
        }
        history.append(record)
        key = (auc, gated_delta)
        if key > best_key:
            best_key = key
            best_state = copy.deepcopy(selector.state_dict())
            torch.save(
                {
                    "selector": best_state,
                    "epoch": epoch,
                    "record": record,
                    "trainable_parameters": trainable,
                },
                output_dir / "best_selector.pth",
            )
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(json.dumps(record, ensure_ascii=False), flush=True)
        save_json(output_dir / "head_history.json", history)
    if best_state is None:
        raise RuntimeError("质量选择器没有生成best状态")
    selector.load_state_dict(best_state)
    probabilities, quality_before, quality_after = predict_cache(
        selector, cache, hold_indices, device
    )
    candidates = np.linspace(0.10, 0.90, 161)
    threshold_rows = []
    for threshold in candidates:
        chosen = probabilities >= threshold
        coverage = float(chosen.mean())
        if 0.10 <= coverage <= 0.90:
            threshold_rows.append(
                (float((delta_hold * chosen).mean()), float(threshold), coverage)
            )
    if not threshold_rows:
        threshold = 0.5
    else:
        _, threshold, _ = max(threshold_rows, key=lambda row: row[0])
    hold_chosen = probabilities >= threshold
    hold_report = {
        "fit_pairs": int(len(fit_indices)),
        "hold_pairs": int(len(hold_indices)),
        "best_epoch": int(max(history, key=lambda item: item["hold_auc"])["epoch"]),
        "auc": float(roc_auc_score(labels_hold, probabilities)),
        "average_precision": float(average_precision_score(labels_hold, probabilities)),
        "positive_rate": float(labels_hold.mean()),
        "threshold": float(threshold),
        "coverage": float(hold_chosen.mean()),
        "gated_mean_delta_iou": float((delta_hold * hold_chosen).mean()),
        "all_refined_mean_delta_iou": float(delta_hold.mean()),
        "quality_base_mae": float(
            np.abs(quality_before - cache["iou_before"][hold_indices].numpy()).mean()
        ),
        "quality_refined_mae": float(
            np.abs(quality_after - cache["iou_after"][hold_indices].numpy()).mean()
        ),
    }
    return selector, threshold, hold_report, trainable


@torch.inference_mode()
def evaluate_selector(detector, refiner, tap, selector, threshold, loader, matcher, postprocessor, args):
    from train_s_hrbr1_refinebox import collect_detections, coco_metrics, move_targets

    selector.eval()
    records = []
    detections = {name: [] for name in ("baseline", "all_refined", "pairwise_gate", "oracle_gate")}
    category_ids = sorted(loader.dataset.coco.getCatIds())
    started = time.time()
    for batch_index, (samples, targets_cpu) in enumerate(loader):
        if args.max_val_batches is not None and batch_index >= args.max_val_batches:
            break
        samples = samples.cuda(non_blocking=True)
        targets = move_targets(targets_cpu, "cuda")
        bundle = forward_candidates(detector, refiner, tap, samples, args.topk, args.precision)
        logits, quality_base, quality_refined = selector(
            bundle["base_roi"], bundle["refined_roi"],
            bundle["base_geometry"], bundle["refined_geometry"]
        )
        probabilities = torch.sigmoid(logits)
        learned_mask = probabilities >= threshold
        oracle_mask = torch.zeros_like(learned_mask)
        positions = matched_positions(bundle, targets, matcher)
        for item in positions:
            image_index, position = item["image_index"], item["position"]
            oracle_mask[image_index, position] = bool(item["beneficial"])
            records.append(
                {
                    "image_id": item["image_id"],
                    "query_id": item["query_id"],
                    "target_id": item["target_id"],
                    "iou_before": item["iou_before"],
                    "iou_after": item["iou_after"],
                    "delta_iou": item["delta_iou"],
                    "beneficial": item["beneficial"],
                    "benefit_probability": float(probabilities[image_index, position].item()),
                    "predicted_quality_before": float(quality_base[image_index, position].item()),
                    "predicted_quality_after": float(quality_refined[image_index, position].item()),
                    "selected": int(learned_mask[image_index, position].item()),
                }
            )
        output_variants = {}
        for name, mask in (
            ("baseline", torch.zeros_like(learned_mask)),
            ("all_refined", torch.ones_like(learned_mask)),
            ("pairwise_gate", learned_mask),
            ("oracle_gate", oracle_mask),
        ):
            chosen = torch.where(
                mask[..., None], bundle["refined_boxes"], bundle["base_boxes"]
            )
            variant = {
                "pred_logits": bundle["outputs"]["pred_logits"].float().clone(),
                "pred_boxes": bundle["outputs"]["pred_boxes"].float().clone(),
            }
            variant["pred_boxes"][bundle["batch_ids"], bundle["selected"]] = chosen
            output_variants[name] = variant
        sizes = torch.stack([target["orig_size"] for target in targets])
        for name, variant in output_variants.items():
            results = postprocessor(variant, sizes)
            collect_detections(detections[name], targets, results, category_ids)
        if batch_index == 0 or (batch_index + 1) % 10 == 0:
            print(
                f"evaluate=val batch={batch_index + 1:03d}/{len(loader)} "
                f"pairs={len(records)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    labels = np.asarray([record["beneficial"] for record in records], dtype=np.int64)
    probabilities_np = np.asarray([record["benefit_probability"] for record in records])
    delta = np.asarray([record["delta_iou"] for record in records])
    chosen = probabilities_np >= threshold
    before = np.asarray([record["iou_before"] for record in records])
    after = np.asarray([record["iou_after"] for record in records])
    if before.mean() < 0.50:
        raise RuntimeError(f"验证匹配平均IoU异常: {before.mean()}")
    classifier = {
        "roc_auc": float(roc_auc_score(labels, probabilities_np)),
        "average_precision": float(average_precision_score(labels, probabilities_np)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, chosen.astype(np.int64))),
        "positive_rate": float(labels.mean()),
        "coverage": float(chosen.mean()),
        "all_refined_mean_delta_iou": float(delta.mean()),
        "gated_mean_delta_iou": float((delta * chosen).mean()),
        "selected_mean_delta_iou": float(delta[chosen].mean()) if chosen.any() else None,
        "rejected_mean_delta_iou": float(delta[~chosen].mean()) if (~chosen).any() else None,
        "oracle_mean_delta_iou": float(np.maximum(delta, 0.0).mean()),
        "mean_iou_before": float(before.mean()),
        "mean_iou_after": float(after.mean()),
    }
    coco = {}
    if args.max_val_batches is None:
        for name, values in detections.items():
            coco[name] = coco_metrics(loader.dataset.coco, values)
        baseline = coco["baseline"]
        for name in ("all_refined", "pairwise_gate", "oracle_gate"):
            coco[name]["delta_vs_baseline"] = {
                metric: coco[name][metric] - baseline[metric] for metric in baseline
            }
        coco["pairwise_gate"]["delta_vs_all_refined"] = {
            metric: coco["pairwise_gate"][metric] - coco["all_refined"][metric]
            for metric in baseline
        }
    return classifier, coco, records


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    report_path = args.output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError(f"正式结果已存在，拒绝覆盖: {report_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.core import YAMLConfig

    if not torch.cuda.is_available():
        raise RuntimeError("S-HRBR4-D1需要CUDA")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = args.train_batch_size
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = args.val_batch_size
    detector, refiner, tap, weight_source = load_frozen_models(args, cfg)
    matcher = cfg.criterion.cuda().eval().matcher
    train_cache, train_metadata = extract_training_cache(
        detector, refiner, tap, cfg.train_dataloader, matcher, args
    )
    save_metadata_csv(args.output_dir / "train_pair_metadata.csv", train_metadata)
    selector, threshold, hold_report, trainable = train_selector(
        train_cache, args, args.output_dir
    )
    val_classifier, coco, val_records = evaluate_selector(
        detector, refiner, tap, selector, threshold, cfg.val_dataloader,
        matcher, cfg.postprocessor, args
    )
    save_metadata_csv(args.output_dir / "val_pair_metadata.csv", val_records)
    gate = {
        "auc_at_least_0_60": val_classifier["roc_auc"] >= 0.60,
        "ap_gain_at_least_0_0005_vs_hrbr1": None,
        "ap75_not_lower_than_hrbr1": None,
        "aps_not_lower_than_hrbr1": None,
        "pass": False,
    }
    if coco:
        gate["ap_gain_at_least_0_0005_vs_hrbr1"] = (
            coco["pairwise_gate"]["AP"] - coco["all_refined"]["AP"] >= 0.0005
        )
        gate["ap75_not_lower_than_hrbr1"] = (
            coco["pairwise_gate"]["AP75"] >= coco["all_refined"]["AP75"]
        )
        gate["aps_not_lower_than_hrbr1"] = (
            coco["pairwise_gate"]["APS"] >= coco["all_refined"]["APS"]
        )
        gate["pass"] = bool(all(value for key, value in gate.items() if key != "pass"))
    report = {
        "experiment": "S-HRBR4-D1-Pairwise-ROI-Quality-Selector",
        "purpose": "冻结A00与HRBR1，从成对P2 ROI语义特征选择校准前/后框",
        "protocol": {
            "config": str(args.config.resolve()),
            "detector_checkpoint": str(args.detector_checkpoint.resolve()),
            "refiner_checkpoint": str(args.refiner_checkpoint.resolve()),
            "weight_source": weight_source,
            "seed": args.seed,
            "train_batch_size": args.train_batch_size,
            "val_batch_size": args.val_batch_size,
            "topk": args.topk,
            "head_epochs": args.head_epochs,
            "head_batch_size": args.head_batch_size,
            "learning_rate": args.learning_rate,
            "trainable_parameters": trainable,
            "train_fit_hold_split": "image_id modulo 5; 80% fit / 20% hold",
            "gt_used_at_inference": False,
            "sam_used": False,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
        },
        "counts": {"train_pairs": len(train_metadata), "val_pairs": len(val_records)},
        "internal_hold": hold_report,
        "validation_classifier": val_classifier,
        "coco": coco,
        "gate": gate,
    }
    save_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
