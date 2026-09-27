#!/usr/bin/env python3
"""S-HRBR5-D0.1：固定FULL权重，在同一次检测器前向中干预P2细节。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.core import YAMLConfig
from train_s_hrbr1_refinebox import (
    BackboneFeatureTap,
    RefineBoxHead,
    coco_metrics,
    collect_detections,
    load_frozen_detector,
    move_targets,
)


MODES = ("full", "lowpass", "shifted_detail")
METRICS = ("AP", "AP50", "AP75", "APS", "APM", "AR100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=ROOT.parent
        / "outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth",
    )
    parser.add_argument(
        "--refiner-checkpoint",
        type=Path,
        default=ROOT.parent / "outputs/S_HRBR5_D0_FULL_SEED0/best.pth",
    )
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent
        / "reports/24_high_resolution_box_refinement"
        / "S_HRBR5_D0_1_FROZEN_INTERVENTION/result.json",
    )
    return parser.parse_args()


def metric_subset(metrics: dict) -> dict:
    return {name: float(metrics[name]) for name in METRICS}


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("S-HRBR5-D0.1 完整评测需要 CUDA")
    for path in (args.config, args.detector_checkpoint, args.refiner_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    torch.manual_seed(0)
    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    detector, weight_source = load_frozen_detector(cfg, args.detector_checkpoint)
    detector.eval()
    val_loader = cfg.val_dataloader
    postprocessor = cfg.postprocessor
    tap = BackboneFeatureTap(detector.backbone)
    channels = tuple(
        detector.backbone._out_channels[index] for index in (0, 1, 2, 3)
    )

    saved = torch.load(args.refiner_checkpoint, map_location="cpu", weights_only=False)
    metadata = saved.get("metadata", {})
    if metadata.get("feature_mode") != "full":
        raise RuntimeError(
            "D0.1必须固定FULL训练得到的最佳权重，当前checkpoint元数据不是full"
        )
    refiner = RefineBoxHead(
        channels,
        d_model=int(metadata.get("d_model", 64)),
        roi_size=int(metadata.get("roi_size", 7)),
        refine_steps=int(metadata.get("refine_steps", 3)),
        feature_mode="full",
    ).cuda()
    refiner.load_state_dict(saved["refiner"], strict=True)
    refiner.force_identity = False
    refiner.eval()

    coco = val_loader.dataset.coco
    category_ids = sorted(coco.getCatIds())
    baseline_detections = []
    detections = {mode: [] for mode in MODES}
    started = time.time()
    use_amp = args.precision == "fp16"

    for batch_index, (samples, targets) in enumerate(val_loader):
        samples = samples.cuda(non_blocking=True)
        targets_cuda = move_targets(targets, "cuda")
        tap.clear()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            outputs = detector(samples)
        features = tuple(feature.float() for feature in tap.features())
        logits = outputs["pred_logits"].float()
        detector_boxes = outputs["pred_boxes"].float()
        scores = logits.sigmoid().max(-1).values
        selected = scores.topk(min(args.topk, scores.shape[1]), dim=1).indices
        image_ids = torch.arange(samples.shape[0], device=samples.device)[:, None]
        boxes = detector_boxes[image_ids, selected]
        flat_boxes = boxes.flatten(0, 1)
        flat_batch = image_ids.expand_as(selected).flatten()
        sizes = torch.stack([target["orig_size"] for target in targets_cuda])

        baseline_results = postprocessor(
            {"pred_logits": logits, "pred_boxes": detector_boxes}, sizes
        )
        collect_detections(
            baseline_detections, targets_cuda, baseline_results, category_ids
        )

        # 三个模式共享本批次完全相同的检测器输出、候选框、特征和后处理参数。
        for mode in MODES:
            refiner.feature_mode = mode
            refined = refiner(features, flat_boxes, flat_batch)[-1]
            adjusted_boxes = detector_boxes.clone()
            adjusted_boxes[image_ids, selected] = refined.view_as(boxes)
            results = postprocessor(
                {"pred_logits": logits, "pred_boxes": adjusted_boxes}, sizes
            )
            collect_detections(detections[mode], targets_cuda, results, category_ids)

        if batch_index == 0 or (batch_index + 1) % 20 == 0:
            print(
                f"batch={batch_index + 1}/{len(val_loader)} "
                f"images={(batch_index + 1) * samples.shape[0]}",
                flush=True,
            )

    baseline = coco_metrics(coco, baseline_detections)
    results = {}
    for mode in MODES:
        metrics = coco_metrics(coco, detections[mode])
        results[mode] = {
            "absolute": metric_subset(metrics),
            "paired_delta": {
                name: float(metrics[name] - baseline[name]) for name in METRICS
            },
        }

    comparisons = {}
    for control in ("lowpass", "shifted_detail"):
        comparisons[f"full_minus_{control}"] = {
            name: results["full"]["absolute"][name]
            - results[control]["absolute"][name]
            for name in METRICS
        }

    full_low = comparisons["full_minus_lowpass"]
    full_shift = comparisons["full_minus_shifted_detail"]
    gate = {
        "full_ap_over_lowpass_at_least_0_001": full_low["AP"] >= 0.001,
        "full_ap_over_shifted_at_least_0_001": full_shift["AP"] >= 0.001,
        "full_ap75_over_both": min(full_low["AP75"], full_shift["AP75"]) > 0,
        "full_aps_over_both": min(full_low["APS"], full_shift["APS"]) > 0,
    }
    gate["pass"] = all(gate.values())
    report = {
        "experiment": "S-HRBR5-D0.1-FROZEN-WEIGHT-SAME-PASS-INTERVENTION",
        "protocol": {
            "detector_checkpoint": str(args.detector_checkpoint.resolve()),
            "detector_weight_source": weight_source,
            "refiner_checkpoint": str(args.refiner_checkpoint.resolve()),
            "refiner_best_epoch": int(saved["epoch"]),
            "same_detector_forward_for_all_modes": True,
            "same_refiner_weights_for_all_modes": True,
            "topk": args.topk,
            "validation_batches": len(val_loader),
            "elapsed_seconds": time.time() - started,
        },
        "baseline": metric_subset(baseline),
        "results": results,
        "comparisons": comparisons,
        "preregistered_gate": gate,
        "interpretation": (
            "通过：FULL校准头在冻结权重后明确依赖位置正确的P2细节；这只证明模型依赖，尚不能单独证明创新性。"
            if gate["pass"]
            else "未通过：FULL校准头冻结后仍未表现出对正确P2细节的稳定依赖，关闭HRBR边缘恢复叙事并保留HRBR1性能模块。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
