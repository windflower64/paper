"""审计最终解码查询的置信度与定位质量是否仍存在错序。

本脚本只读取既有配置和权重，不修改模型，也不使用真值改变正式预测。
它回答的是最终解码结果的排序问题，与编码器 Top-K 候选是否覆盖目标是两件事。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torchvision
from torchvision.ops import box_iou


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        dim=-1,
    )


def normalized_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.as_subclass(torch.Tensor) if hasattr(boxes, "as_subclass") else boxes
    plain = boxes.as_subclass(torch.Tensor) if hasattr(boxes, "as_subclass") else boxes
    box_format = getattr(boxes, "format", None)
    if box_format is not None:
        format_name = getattr(box_format, "value", str(box_format)).lower()
        if format_name != "cxcywh":
            plain = torchvision.ops.box_convert(
                plain, in_fmt=format_name, out_fmt="cxcywh"
            )
    canvas_size = getattr(boxes, "canvas_size", None)
    if canvas_size is None:
        canvas_size = getattr(boxes, "spatial_size", None)
    if canvas_size is not None and float(plain.max()) > 1.0:
        height, width = canvas_size
        plain = plain / plain.new_tensor([width, height, width, height])
    return plain


def checkpoint_state(checkpoint: dict) -> dict:
    if "ema" in checkpoint:
        ema = checkpoint["ema"]
        return ema["module"] if isinstance(ema, dict) and "module" in ema else ema
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from src.core import YAMLConfig

    config = YAMLConfig(str(args.config.resolve()))
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = config.model
    checkpoint = torch.load(
        args.checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    incompatible = model.load_state_dict(checkpoint_state(checkpoint), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint/config mismatch: "
            f"missing={incompatible.missing_keys[:10]}, "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )

    device = torch.device(args.device)
    model.to(device).eval()
    loader = config.val_dataloader

    cutoffs = (1, 3, 10, 30, 100, 300)
    best_iou_by_cutoff: dict[int, list[float]] = {cutoff: [] for cutoff in cutoffs}
    oracle_ranks: list[float] = []
    top1_ious: list[float] = []
    oracle_ious: list[float] = []
    iou_regrets: list[float] = []
    correlations: list[float] = []
    recoverable_005 = 0
    recoverable_010 = 0
    processed_images = 0
    target_images = 0
    empty_images = 0

    with torch.inference_mode():
        for samples, targets in loader:
            if args.max_images and processed_images >= args.max_images:
                break
            samples = samples.to(device)
            outputs = model(samples)
            logits = outputs["pred_logits"]
            boxes = outputs["pred_boxes"]
            scores = logits.sigmoid().amax(dim=-1)

            for batch_index in range(samples.shape[0]):
                if args.max_images and processed_images >= args.max_images:
                    break
                processed_images += 1
                gt_boxes = normalized_cxcywh(targets[batch_index]["boxes"]).to(device)
                if gt_boxes.numel() == 0:
                    empty_images += 1
                    continue
                target_images += 1

                ious = box_iou(
                    cxcywh_to_xyxy(boxes[batch_index]),
                    cxcywh_to_xyxy(gt_boxes),
                ).amax(dim=1)
                order = scores[batch_index].argsort(descending=True)
                oracle_index = int(ious.argmax())
                oracle_rank = int((order == oracle_index).nonzero()[0]) + 1
                top1_iou = float(ious[order[0]].cpu())
                oracle_iou = float(ious[oracle_index].cpu())
                regret = oracle_iou - top1_iou

                oracle_ranks.append(float(oracle_rank))
                top1_ious.append(top1_iou)
                oracle_ious.append(oracle_iou)
                iou_regrets.append(regret)
                recoverable_005 += int(regret >= 0.05)
                recoverable_010 += int(regret >= 0.10)

                score_values = scores[batch_index].float()
                if score_values.std() > 0 and ious.float().std() > 0:
                    correlation = torch.corrcoef(
                        torch.stack((score_values, ious.float()))
                    )[0, 1]
                    if torch.isfinite(correlation):
                        correlations.append(float(correlation.cpu()))

                for cutoff in cutoffs:
                    actual = min(cutoff, order.numel())
                    best_iou_by_cutoff[cutoff].append(
                        float(ious[order[:actual]].max().cpu())
                    )

    if target_images == 0:
        raise RuntimeError("no target images were evaluated")

    result = {
        "status": "PASS",
        "purpose": "measure_final_decoder_score_localization_ranking_gap",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "processed_images": processed_images,
        "target_images": target_images,
        "empty_images": empty_images,
        "mean_top1_score_query_iou": safe_mean(top1_ious),
        "mean_oracle_query_iou": safe_mean(oracle_ious),
        "mean_iou_regret": safe_mean(iou_regrets),
        "median_iou_regret": percentile(iou_regrets, 0.50),
        "recoverable_gap_ge_005_ratio": recoverable_005 / target_images,
        "recoverable_gap_ge_010_ratio": recoverable_010 / target_images,
        "oracle_query_score_rank": {
            "mean": safe_mean(oracle_ranks),
            "median": percentile(oracle_ranks, 0.50),
            "p90": percentile(oracle_ranks, 0.90),
            "within_top1_ratio": sum(rank <= 1 for rank in oracle_ranks) / target_images,
            "within_top3_ratio": sum(rank <= 3 for rank in oracle_ranks) / target_images,
            "within_top10_ratio": sum(rank <= 10 for rank in oracle_ranks) / target_images,
        },
        "mean_score_iou_pearson_all_queries": safe_mean(correlations),
        "mean_best_iou_by_score_cutoff": {
            str(cutoff): safe_mean(values)
            for cutoff, values in best_iou_by_cutoff.items()
        },
        "interpretation_boundary": (
            "This is an oracle headroom diagnostic using ground truth only for analysis; "
            "it is not a deployable reranking result and does not report COCO AP."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
