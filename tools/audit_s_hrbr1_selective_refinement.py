#!/usr/bin/env python3
"""S-HRBR4-D0：审计HRBR1逐目标收益，并验证可观测选择门是否成立。

该脚本不训练检测器或RefineBox。它固定A00与HRBR1 best：
1. 在训练集上用匹配正样本拟合一个小型随机森林诊断器；
2. 在完整验证集上判断“是否采用HRBR1校准”能否由推理时可观测量预测；
3. 同时输出无门控、学习门控和使用GT的oracle门控COCO指标。

GT IoU只用于监督和审计，绝不进入门控器的输入特征。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=Path(
            "E:/two_paper/outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth"
        ),
    )
    parser.add_argument(
        "--refiner-checkpoint",
        type=Path,
        default=Path(
            "E:/two_paper/outputs/S_HRBR1_REFINEBOX_OFFICIAL_FPN_SEED0/best.pth"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "E:/two_paper/reports/24_high_resolution_box_refinement/"
            "S_HRBR4_D0_SELECTIVE_AUDIT_V3"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    # 与采用的HRBR1 seed0权重保持完全相同的评测随机种子。
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    return parser.parse_args()


FEATURE_NAMES = [
    "score",
    "score_margin",
    "rank_fraction",
    "center_x",
    "center_y",
    "width",
    "height",
    "log_area",
    "log_aspect",
    "border_distance",
    "fpn_level",
    "correction_l1",
    "correction_l2",
    "correction_logit_l1",
    "step1_l1",
    "step2_l1",
    "step3_l1",
]


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise RuntimeError(f"没有记录可写入: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def tensor_iou_xyxy(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    top_left = torch.maximum(box_a[:2], box_b[:2])
    bottom_right = torch.minimum(box_a[2:], box_b[2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod()
    area_a = (box_a[2:] - box_a[:2]).clamp_min(0).prod()
    area_b = (box_b[2:] - box_b[:2]).clamp_min(0).prod()
    union = area_a + area_b - intersection
    return float((intersection / union.clamp_min(1e-9)).item())


def normalize_targets_cxcywh(
    targets: list[dict], image_height: int, image_width: int
) -> list[dict]:
    """统一训练/验证target格式；验证集原始接口是绝对XYXY。"""
    normalized_targets = []
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
        elif "cxcywh" not in format_name and max_value > 1.5:
            raise RuntimeError(f"无法判断绝对框格式: {format_name!r}")
        if boxes.numel() and float(boxes.max()) > 1.5:
            boxes = boxes / scale
        converted = dict(target)
        converted["boxes"] = boxes
        normalized_targets.append(converted)
    return normalized_targets


def observable_features(
    logits: torch.Tensor,
    selected: torch.Tensor,
    base_boxes: torch.Tensor,
    sequence: list[torch.Tensor],
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """返回[B,K,F]特征、预测标签和分数；所有量在推理时可见。"""
    probabilities = logits.float().sigmoid()
    query_scores, query_labels = probabilities.max(-1)
    scores = query_scores.gather(1, selected)
    labels = query_labels.gather(1, selected)
    if probabilities.shape[-1] > 1:
        top2 = probabilities.topk(2, dim=-1).values
        margins_all = top2[..., 0] - top2[..., 1]
    else:
        margins_all = query_scores
    margins = margins_all.gather(1, selected)

    batch_size, count, _ = base_boxes.shape
    final = sequence[-1].view(batch_size, count, 4)
    steps = [item.view(batch_size, count, 4) for item in sequence]
    previous = base_boxes
    step_l1 = []
    for item in steps:
        step_l1.append((item - previous).abs().sum(-1))
        previous = item

    correction = final - base_boxes
    eps = 1e-8
    width = base_boxes[..., 2].clamp_min(eps)
    height = base_boxes[..., 3].clamp_min(eps)
    area = width * height
    scale_pixels = torch.sqrt(area * float(image_height * image_width))
    fpn_level = torch.floor(4.0 + torch.log2(scale_pixels / 224.0 + eps)).clamp(2, 5)
    border = torch.stack(
        (
            base_boxes[..., 0],
            base_boxes[..., 1],
            1.0 - base_boxes[..., 0],
            1.0 - base_boxes[..., 1],
        ),
        dim=-1,
    ).amin(-1)
    rank = torch.arange(count, device=base_boxes.device, dtype=torch.float32)
    rank = rank.view(1, count).expand(batch_size, -1) / max(1, count - 1)

    from train_s_hrbr1_refinebox import inverse_sigmoid

    logit_correction = (
        inverse_sigmoid(final) - inverse_sigmoid(base_boxes)
    ).abs().sum(-1)
    values = [
        scores,
        margins,
        rank,
        base_boxes[..., 0],
        base_boxes[..., 1],
        width,
        height,
        torch.log(area + eps),
        torch.log(width / height),
        border,
        fpn_level,
        correction.abs().sum(-1),
        torch.sqrt((correction.square()).sum(-1)),
        logit_correction,
    ]
    values.extend(step_l1)
    matrix = torch.stack(values, dim=-1).detach().cpu().numpy().astype(np.float32)
    return matrix, labels.detach(), scores.detach()


def record_matches(
    outputs,
    match_targets,
    raw_targets,
    selected,
    base_boxes,
    sequence,
    features,
    predicted_labels,
    matcher,
    split: str,
    batch_number: int,
) -> tuple[list[dict], list[dict[tuple[int, int], bool]]]:
    """记录Hungarian匹配正样本，并返回每张图的oracle采用映射。"""
    core = {
        "pred_logits": outputs["pred_logits"].float(),
        "pred_boxes": outputs["pred_boxes"].float(),
    }
    matched = matcher(core, match_targets)["indices"]
    sequence_final = sequence[-1].view_as(base_boxes)
    records: list[dict] = []
    oracle_maps: list[dict[tuple[int, int], bool]] = []
    for image_index, ((query_ids, target_ids), target, raw_target) in enumerate(
        zip(matched, match_targets, raw_targets)
    ):
        query_ids = query_ids.to(selected.device)
        target_ids = target_ids.to(selected.device)
        position_by_query = {
            int(query_id): position
            for position, query_id in enumerate(selected[image_index].tolist())
        }
        oracle_map: dict[tuple[int, int], bool] = {}
        # COCO数据接口的orig_size顺序是[width, height]。
        orig_w, orig_h = [int(value) for value in raw_target["orig_size"].tolist()]
        for query_id_tensor, target_id_tensor in zip(query_ids, target_ids):
            query_id = int(query_id_tensor.item())
            target_id = int(target_id_tensor.item())
            if query_id not in position_by_query:
                continue
            position = position_by_query[query_id]
            base = base_boxes[image_index, position]
            refined = sequence_final[image_index, position]
            truth = target["boxes"][target_id].float()
            base_xyxy = box_cxcywh_to_xyxy(base)
            refined_xyxy = box_cxcywh_to_xyxy(refined)
            truth_xyxy = box_cxcywh_to_xyxy(truth)
            before = tensor_iou_xyxy(base_xyxy, truth_xyxy)
            after = tensor_iou_xyxy(refined_xyxy, truth_xyxy)
            delta = after - before
            target_label = int(target["labels"][target_id].item())
            predicted_label = int(predicted_labels[image_index, position].item())
            class_correct = predicted_label == target_label
            observable = features[image_index, position]
            record = {
                "split": split,
                "batch": batch_number,
                "image_id": int(target["image_id"].item()),
                "query_id": query_id,
                "target_id": target_id,
                "target_label": target_label,
                "predicted_label": predicted_label,
                "class_correct": int(class_correct),
                "orig_height": orig_h,
                "orig_width": orig_w,
            }
            record.update(
                {name: float(value) for name, value in zip(FEATURE_NAMES, observable)}
            )
            if "area" in raw_target and len(raw_target["area"]) > target_id:
                gt_area = float(raw_target["area"][target_id].item())
            else:
                gt_area = float(truth[2].item() * orig_w * truth[3].item() * orig_h)
            record.update(
                {
                    "gt_box_area": gt_area,
                    "iou_before": before,
                    "iou_after": after,
                    "delta_iou": delta,
                    "beneficial": int(delta > 0.0 and class_correct),
                    "cross_up_50": int(before < 0.50 <= after and class_correct),
                    "cross_down_50": int(after < 0.50 <= before and class_correct),
                    "cross_up_75": int(before < 0.75 <= after and class_correct),
                    "cross_down_75": int(after < 0.75 <= before and class_correct),
                }
            )
            records.append(record)
            oracle_map[(image_index, position)] = bool(delta > 0.0 and class_correct)
        oracle_maps.append(oracle_map)
    return records, oracle_maps


def box_cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = box.unbind(-1)
    return torch.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        dim=-1,
    )


def make_classifier(seed: int, trees: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=trees,
        max_depth=6,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def choose_threshold_oof(
    x: np.ndarray,
    labels: np.ndarray,
    delta_iou: np.ndarray,
    seed: int,
    trees: int,
) -> tuple[float, dict]:
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    probabilities = cross_val_predict(
        make_classifier(seed, trees),
        x,
        labels,
        cv=folds,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    candidates = np.linspace(0.25, 0.75, 101)
    rows = []
    for threshold in candidates:
        selected = probabilities >= threshold
        coverage = float(selected.mean())
        if not 0.10 <= coverage <= 0.90:
            continue
        gated_mean_delta = float((delta_iou * selected).mean())
        rows.append((gated_mean_delta, threshold, coverage))
    if not rows:
        threshold = 0.5
        coverage = float((probabilities >= threshold).mean())
        objective = float((delta_iou * (probabilities >= threshold)).mean())
    else:
        objective, threshold, coverage = max(rows, key=lambda item: item[0])
    diagnostics = {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "positive_rate": float(labels.mean()),
        "chosen_threshold": float(threshold),
        "chosen_coverage": float(coverage),
        "oof_gated_mean_delta_iou": float(objective),
        "oof_all_refined_mean_delta_iou": float(delta_iou.mean()),
    }
    return float(threshold), diagnostics


def summarize_records(records: list[dict]) -> dict:
    if not records:
        return {"count": 0}
    deltas = np.asarray([record["delta_iou"] for record in records])
    before = np.asarray([record["iou_before"] for record in records])
    after = np.asarray([record["iou_after"] for record in records])
    return {
        "count": len(records),
        "mean_iou_before": float(before.mean()),
        "mean_iou_after": float(after.mean()),
        "mean_delta_iou": float(deltas.mean()),
        "median_delta_iou": float(np.median(deltas)),
        "positive_rate": float((deltas > 0).mean()),
        "negative_rate": float((deltas < 0).mean()),
        "cross_up_50": int(sum(record["cross_up_50"] for record in records)),
        "cross_down_50": int(sum(record["cross_down_50"] for record in records)),
        "cross_up_75": int(sum(record["cross_up_75"] for record in records)),
        "cross_down_75": int(sum(record["cross_down_75"] for record in records)),
    }


def grouped_summaries(records: list[dict]) -> dict:
    groups: dict[str, dict[str, list[dict]]] = {
        "gt_size": defaultdict(list),
        "initial_iou": defaultdict(list),
        "fpn_level": defaultdict(list),
    }
    for record in records:
        area = record["gt_box_area"]
        size_name = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
        groups["gt_size"][size_name].append(record)
        iou = record["iou_before"]
        if iou < 0.50:
            iou_name = "lt_0.50"
        elif iou < 0.75:
            iou_name = "0.50_0.75"
        elif iou < 0.90:
            iou_name = "0.75_0.90"
        else:
            iou_name = "ge_0.90"
        groups["initial_iou"][iou_name].append(record)
        groups["fpn_level"][f"P{int(record['fpn_level'])}"].append(record)
    return {
        group_name: {
            key: summarize_records(value) for key, value in sorted(parts.items())
        }
        for group_name, parts in groups.items()
    }


def load_models(args, cfg):
    from train_s_hrbr1_refinebox import (
        BackboneFeatureTap,
        RefineBoxHead,
        load_frozen_detector,
    )

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
def run_split(
    *,
    split,
    detector,
    refiner,
    tap,
    loader,
    matcher,
    topk,
    precision,
    max_batches,
    classifier=None,
    threshold=None,
    postprocessor=None,
):
    from train_s_hrbr1_refinebox import (
        collect_detections,
        move_targets,
    )

    records: list[dict] = []
    detections = {name: [] for name in ("baseline", "all_refined", "learned_gate", "oracle_gate")}
    category_ids = sorted(loader.dataset.coco.getCatIds())
    use_amp = precision == "fp16"
    started = time.time()
    for batch_index, (samples, targets_cpu) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        samples = samples.cuda(non_blocking=True)
        targets = move_targets(targets_cpu, "cuda")
        tap.clear()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            outputs = detector(samples)
        backbone_features = tuple(feature.float() for feature in tap.features())
        query_scores = outputs["pred_logits"].float().sigmoid().max(-1).values
        selected = query_scores.topk(min(topk, query_scores.shape[1]), dim=1).indices
        batch_ids = torch.arange(samples.shape[0], device=samples.device)[:, None]
        base_boxes = outputs["pred_boxes"].float()[batch_ids, selected]
        flat_batch = batch_ids.expand_as(selected).flatten()
        sequence_flat = refiner(backbone_features, base_boxes.flatten(0, 1), flat_batch)
        image_height = int(backbone_features[0].shape[-2] * 4)
        image_width = int(backbone_features[0].shape[-1] * 4)
        match_targets = normalize_targets_cxcywh(targets, image_height, image_width)
        feature_matrix, predicted_labels, _ = observable_features(
            outputs["pred_logits"], selected, base_boxes, sequence_flat,
            image_height, image_width,
        )
        batch_records, oracle_maps = record_matches(
            outputs, match_targets, targets, selected, base_boxes, sequence_flat,
            feature_matrix, predicted_labels, matcher, split, batch_index,
        )
        records.extend(batch_records)

        if classifier is not None:
            batch_size, count, feature_count = feature_matrix.shape
            probabilities = classifier.predict_proba(
                feature_matrix.reshape(-1, feature_count)
            )[:, 1].reshape(batch_size, count)
            learned_mask = torch.from_numpy(probabilities >= threshold).to(selected.device)
            oracle_mask = torch.zeros_like(learned_mask)
            for image_index, mapping in enumerate(oracle_maps):
                for (_stored_image, position), use_refined in mapping.items():
                    oracle_mask[image_index, position] = use_refined
            final_boxes = sequence_flat[-1].view_as(base_boxes)
            output_variants = {}
            for name, mask in (
                ("baseline", torch.zeros_like(learned_mask)),
                ("all_refined", torch.ones_like(learned_mask)),
                ("learned_gate", learned_mask),
                ("oracle_gate", oracle_mask),
            ):
                chosen = torch.where(mask[..., None], final_boxes, base_boxes)
                variant = {
                    "pred_logits": outputs["pred_logits"].float().clone(),
                    "pred_boxes": outputs["pred_boxes"].float().clone(),
                }
                variant["pred_boxes"][batch_ids, selected] = chosen
                output_variants[name] = variant
            sizes = torch.stack([target["orig_size"] for target in targets])
            for name, variant in output_variants.items():
                results = postprocessor(variant, sizes)
                collect_detections(detections[name], targets, results, category_ids)

        if batch_index == 0 or (batch_index + 1) % 20 == 0:
            print(
                f"split={split} batch={batch_index + 1:03d}/{len(loader)} "
                f"matched_records={len(records)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    return records, detections


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    report_path = args.output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError(f"结果已存在，拒绝覆盖: {report_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.core import YAMLConfig
    from train_s_hrbr1_refinebox import coco_metrics

    if not torch.cuda.is_available():
        raise RuntimeError("S-HRBR4-D0需要CUDA")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = args.batch_size
    # HRBR1正式训练器没有覆盖val batch，沿用配置默认64才能严格复现其绝对指标。
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = args.val_batch_size
    detector, refiner, tap, weight_source = load_models(args, cfg)
    matcher = cfg.criterion.cuda().eval().matcher
    train_loader = cfg.train_dataloader
    val_loader = cfg.val_dataloader

    train_records, _ = run_split(
        split="train",
        detector=detector,
        refiner=refiner,
        tap=tap,
        loader=train_loader,
        matcher=matcher,
        topk=args.topk,
        precision=args.precision,
        max_batches=args.max_train_batches,
    )
    train_usable = [record for record in train_records if record["class_correct"]]
    x_train = np.asarray(
        [[record[name] for name in FEATURE_NAMES] for record in train_usable],
        dtype=np.float32,
    )
    y_train = np.asarray([record["beneficial"] for record in train_usable], dtype=np.int64)
    delta_train = np.asarray([record["delta_iou"] for record in train_usable])
    if len(np.unique(y_train)) != 2:
        raise RuntimeError("训练审计记录缺少正类或负类，无法拟合选择门")
    threshold, oof = choose_threshold_oof(
        x_train, y_train, delta_train, args.seed, args.trees
    )
    classifier = make_classifier(args.seed, args.trees)
    classifier.fit(x_train, y_train)
    print(json.dumps({"oof": oof}, ensure_ascii=False, indent=2), flush=True)

    val_records, detections = run_split(
        split="val",
        detector=detector,
        refiner=refiner,
        tap=tap,
        loader=val_loader,
        matcher=matcher,
        topk=args.topk,
        precision=args.precision,
        max_batches=args.max_val_batches,
        classifier=classifier,
        threshold=threshold,
        postprocessor=cfg.postprocessor,
    )
    val_usable = [record for record in val_records if record["class_correct"]]
    val_record_summary = summarize_records(val_usable)
    if val_record_summary.get("mean_iou_before", 0.0) < 0.50:
        raise RuntimeError(
            "验证匹配框平均初始IoU低于0.50，疑似target格式或匹配接口错误: "
            f"{val_record_summary}"
        )
    x_val = np.asarray(
        [[record[name] for name in FEATURE_NAMES] for record in val_usable],
        dtype=np.float32,
    )
    y_val = np.asarray([record["beneficial"] for record in val_usable], dtype=np.int64)
    delta_val = np.asarray([record["delta_iou"] for record in val_usable])
    probabilities = classifier.predict_proba(x_val)[:, 1]
    selected_val = probabilities >= threshold
    val_classifier = {
        "roc_auc": float(roc_auc_score(y_val, probabilities)),
        "average_precision": float(average_precision_score(y_val, probabilities)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_val, selected_val.astype(np.int64))
        ),
        "positive_rate": float(y_val.mean()),
        "selected_coverage": float(selected_val.mean()),
        "selected_mean_delta_iou": float(delta_val[selected_val].mean())
        if selected_val.any()
        else None,
        "rejected_mean_delta_iou": float(delta_val[~selected_val].mean())
        if (~selected_val).any()
        else None,
        "gated_mean_delta_iou": float((delta_val * selected_val).mean()),
        "all_refined_mean_delta_iou": float(delta_val.mean()),
        "oracle_mean_delta_iou": float(np.maximum(delta_val, 0.0).mean()),
    }

    coco = {}
    if args.max_val_batches is None:
        coco_gt = val_loader.dataset.coco
        for name, values in detections.items():
            coco[name] = coco_metrics(coco_gt, values)
        baseline = coco["baseline"]
        for name in ("all_refined", "learned_gate", "oracle_gate"):
            coco[name]["delta_vs_baseline"] = {
                metric: coco[name][metric] - baseline[metric] for metric in baseline
            }
        coco["learned_gate"]["delta_vs_all_refined"] = {
            metric: coco["learned_gate"][metric] - coco["all_refined"][metric]
            for metric in baseline
        }

    feature_importance = sorted(
        (
            {"feature": name, "importance": float(importance)}
            for name, importance in zip(FEATURE_NAMES, classifier.feature_importances_)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    gate = {
        "auc_at_least_0_60": val_classifier["roc_auc"] >= 0.60,
        "gated_iou_better_than_all": (
            val_classifier["gated_mean_delta_iou"]
            > val_classifier["all_refined_mean_delta_iou"]
        ),
        "coco_ap_gain_at_least_0_0005_vs_all": None,
        "pass": False,
    }
    if coco:
        ap_delta = coco["learned_gate"]["AP"] - coco["all_refined"]["AP"]
        gate["coco_ap_gain_at_least_0_0005_vs_all"] = ap_delta >= 0.0005
        gate["pass"] = bool(
            gate["auc_at_least_0_60"]
            and gate["gated_iou_better_than_all"]
            and gate["coco_ap_gain_at_least_0_0005_vs_all"]
        )

    report = {
        "experiment": "S-HRBR4-D0-Selective-Refinement-Audit",
        "purpose": "判断HRBR1收益是否能由推理时可观测量预测；不使用SAM，不训练检测器",
        "protocol": {
            "config": str(args.config.resolve()),
            "detector_checkpoint": str(args.detector_checkpoint.resolve()),
            "refiner_checkpoint": str(args.refiner_checkpoint.resolve()),
            "weight_source": weight_source,
            "batch_size": args.batch_size,
            "val_batch_size": args.val_batch_size,
            "topk": args.topk,
            "precision": args.precision,
            "seed": args.seed,
            "classifier": "RandomForest(max_depth=6,min_samples_leaf=20,class_weight=balanced_subsample)",
            "threshold_source": "训练集五折OOF最大化整体门控mean_delta_iou，覆盖率限制10%-90%",
            "gt_not_used_as_input": True,
            "feature_names": FEATURE_NAMES,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
        },
        "counts": {
            "train_records": len(train_records),
            "train_class_correct": len(train_usable),
            "val_records": len(val_records),
            "val_class_correct": len(val_usable),
        },
        "training_oof": oof,
        "validation_classifier": val_classifier,
        "validation_records": val_record_summary,
        "validation_groups": grouped_summaries(val_usable),
        "feature_importance": feature_importance,
        "coco": coco,
        "gate": gate,
    }
    save_csv(args.output_dir / "train_records.csv", train_records)
    save_csv(args.output_dir / "val_records.csv", val_records)
    save_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
