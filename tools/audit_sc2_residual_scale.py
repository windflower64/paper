#!/usr/bin/env python3
"""SC2固定best的无训练残差缩放审计。

所有alpha共享每个batch的同一次GQ1前向、同一组候选框、同一组特征和同一个SC2校准结果，
只改变最终框在GQ1原框与SC2框之间的线性插值比例。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from train_s_hrbr1_refinebox import (
    BackboneFeatureTap,
    RefineBoxHead,
    collect_detections,
    load_frozen_detector,
    move_targets,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument(
        "--config",
        type=Path,
        default=repo / "experiments/phase_sc/sc2_gq1_local_hrbr_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=workspace
        / "runs/30_channel/C_PAT_GQ1_S32_R4/seed0/best_stg1.pth",
    )
    parser.add_argument(
        "--refiner-checkpoint",
        type=Path,
        default=workspace / "outputs/SC2_GQ1_LOCAL_HRBR_SEED0/best.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace
        / "reports/SC联合系列/SC2_RESIDUAL_SCALE_AUDIT/result.json",
    )
    parser.add_argument("--scales", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    return parser.parse_args()


def parse_scales(specification: str) -> tuple[float, ...]:
    try:
        scales = tuple(float(item.strip()) for item in specification.split(","))
    except ValueError as error:
        raise ValueError(f"无法解析scales: {specification}") from error
    if not scales or any(scale < 0.0 or scale > 1.0 for scale in scales):
        raise ValueError("scales必须位于[0, 1]")
    if len(scales) != len(set(scales)):
        raise ValueError("scales不能重复")
    return scales


def coco_metrics_with_iou_curve(coco_gt, detections):
    from faster_coco_eval import COCOeval_faster

    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = (
        "AP", "AP50", "AP75", "APS", "APM", "APL",
        "AR1", "AR10", "AR100", "ARS", "ARM", "ARL",
    )
    metrics = {name: float(value) for name, value in zip(names, evaluator.stats)}
    precision = evaluator.eval["precision"]
    curve = {}
    for index, threshold in enumerate(evaluator.params.iouThrs):
        values = precision[index, :, :, 0, -1]
        valid = values[values > -1]
        curve[f"{float(threshold):.2f}"] = (
            float(valid.mean()) if valid.size else float("nan")
        )
    return metrics, curve


@torch.inference_mode()
def evaluate_scales(
    detector,
    refiner,
    tap,
    loader,
    postprocessor,
    topk: int,
    precision: str,
    scales: tuple[float, ...],
):
    detector.eval()
    refiner.eval()
    coco = loader.dataset.coco
    category_ids = sorted(coco.getCatIds())
    baseline_detections = []
    detections = {scale: [] for scale in scales}
    use_amp = precision == "fp16"

    for batch_index, (samples, targets) in enumerate(loader):
        samples = samples.cuda(non_blocking=True)
        targets_cuda = move_targets(targets, "cuda")
        tap.clear()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            outputs = detector(samples)
        features = tuple(feature.float() for feature in tap.features())
        logits = outputs["pred_logits"].float()
        original_boxes = outputs["pred_boxes"].float()
        scores = logits.sigmoid().max(-1).values
        selected = scores.topk(min(topk, scores.shape[1]), dim=1).indices
        batch_ids = torch.arange(samples.shape[0], device=samples.device)[:, None]
        selected_boxes = original_boxes[batch_ids, selected]
        flat_original = selected_boxes.flatten(0, 1)
        flat_batch = batch_ids.expand_as(selected).flatten()
        fully_refined = refiner(features, flat_original, flat_batch)[-1]
        sizes = torch.stack([target["orig_size"] for target in targets_cuda])

        baseline_results = postprocessor(
            {"pred_logits": logits, "pred_boxes": original_boxes}, sizes
        )
        collect_detections(
            baseline_detections, targets_cuda, baseline_results, category_ids
        )

        for scale in scales:
            blended = (flat_original + scale * (fully_refined - flat_original)).clamp(
                0.0, 1.0
            )
            adjusted_boxes = original_boxes.clone()
            adjusted_boxes[batch_ids, selected] = blended.view_as(selected_boxes)
            results = postprocessor(
                {"pred_logits": logits, "pred_boxes": adjusted_boxes}, sizes
            )
            collect_detections(
                detections[scale], targets_cuda, results, category_ids
            )

        if batch_index == 0 or (batch_index + 1) % 10 == 0:
            print(f"batch={batch_index + 1:03d}/{len(loader)}", flush=True)

    baseline_metrics, baseline_curve = coco_metrics_with_iou_curve(
        coco, baseline_detections
    )
    baseline_payload = dict(baseline_metrics)
    baseline_payload["AP_by_IoU"] = baseline_curve
    scale_metrics = {}
    for scale in scales:
        metrics, curve = coco_metrics_with_iou_curve(coco, detections[scale])
        metrics["delta"] = {
            key: metrics[key] - baseline_metrics[key] for key in baseline_metrics
        }
        metrics["AP_by_IoU"] = curve
        metrics["delta_AP_by_IoU"] = {
            key: curve[key] - baseline_curve[key] for key in baseline_curve
        }
        scale_metrics[f"{scale:.2f}"] = metrics
    return baseline_payload, scale_metrics


def main() -> None:
    args = parse_args()
    scales = parse_scales(args.scales)
    sys.path.insert(0, str(args.repo))
    from src.core import YAMLConfig

    if not torch.cuda.is_available():
        raise RuntimeError("SC2残差缩放审计需要CUDA")

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    detector, weight_source = load_frozen_detector(cfg, args.checkpoint)
    stage_indices = (0, 1, 2)
    tap = BackboneFeatureTap(detector.backbone, stage_indices=stage_indices)
    channels = tuple(detector.backbone._out_channels[index] for index in stage_indices)
    refiner = RefineBoxHead(channels).cuda()
    saved = torch.load(
        args.refiner_checkpoint, map_location="cpu", weights_only=False
    )
    refiner.load_state_dict(saved.get("refiner", saved), strict=True)
    refiner.force_identity = False

    started = time.time()
    baseline, results = evaluate_scales(
        detector,
        refiner,
        tap,
        cfg.val_dataloader,
        cfg.postprocessor,
        args.topk,
        args.precision,
        scales,
    )
    payload = {
        "experiment": "SC2-Residual-Scale-Audit",
        "purpose": "固定GQ1和SC2 best，共享同次前向审计框残差缩放",
        "config": str(args.config.resolve()),
        "detector_checkpoint": str(args.checkpoint.resolve()),
        "detector_weight_source": weight_source,
        "refiner_checkpoint": str(args.refiner_checkpoint.resolve()),
        "refiner_epoch": int(saved.get("epoch", -1)),
        "feature_stage_indices": list(stage_indices),
        "feature_strides": [4, 8, 16],
        "topk": args.topk,
        "scales": list(scales),
        "blend_rule": "final = gq1 + alpha * (sc2 - gq1)",
        "baseline": baseline,
        "results": results,
        "elapsed_seconds": time.time() - started,
    }
    save_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
