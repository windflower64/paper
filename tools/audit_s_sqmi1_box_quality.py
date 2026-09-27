#!/usr/bin/env python3
"""Compare learned query-mask boxes with D-FINE proposals and matched GT."""

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_s/s_sqmi1_init_c_gq1_b16a2_12e_testdev_local.yml",
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=WORKSPACE / "outputs/S_SQMI1_INIT_C_GQ1_B16A2_12E_TESTDEV/seed0",
    )
    parser.add_argument("--batches", type=int, default=114)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "reports/145_sqmi1_result/box_quality.json",
    )
    return parser.parse_args()


def state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"]
    return checkpoint.get("model", checkpoint)


def cxcywh_to_xyxy(box):
    center, size = box[..., :2], box[..., 2:]
    return torch.cat((center - size * 0.5, center + size * 0.5), dim=-1)


def xyxy_to_normalized_cxcywh(boxes, height, width):
    boxes = boxes / boxes.new_tensor([width, height, width, height])
    top_left, bottom_right = boxes[..., :2], boxes[..., 2:]
    return torch.cat(((top_left + bottom_right) * 0.5, bottom_right - top_left), dim=-1)


def aligned_iou(left, right):
    left = cxcywh_to_xyxy(left)
    right = cxcywh_to_xyxy(right)
    top_left = torch.maximum(left[..., :2], right[..., :2])
    bottom_right = torch.minimum(left[..., 2:], right[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    left_area = (left[..., 2:] - left[..., :2]).clamp_min(0).prod(dim=-1)
    right_area = (right[..., 2:] - right[..., :2]).clamp_min(0).prod(dim=-1)
    return intersection / (left_area + right_area - intersection).clamp_min(1e-9)


def inspect(config, checkpoint, batches):
    cfg = YAMLConfig(str(config))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    model.load_state_dict(state(checkpoint), strict=True)
    model = model.cuda().eval()
    criterion = cfg.criterion.cuda().eval()

    totals = {"base": 0.0, "mask": 0.0, "refined": 0.0, "final": 0.0}
    valid_totals = {"base": 0.0, "mask": 0.0, "refined": 0.0, "final": 0.0}
    matched = 0
    topk_matched = 0
    valid_matched = 0
    mask_better = 0
    refined_better = 0
    with torch.inference_mode():
        for batch_index, (samples, targets) in enumerate(cfg.val_dataloader):
            if batch_index >= batches:
                break
            samples = samples.cuda(non_blocking=True)
            targets = [
                {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in target.items()}
                for target in targets
            ]
            outputs = model(samples)
            normalized_targets = []
            for target in targets:
                normalized = dict(target)
                normalized["boxes"] = xyxy_to_normalized_cxcywh(
                    target["boxes"], samples.shape[-2], samples.shape[-1]
                )
                normalized_targets.append(normalized)
            indices = criterion.matcher(outputs, normalized_targets)["indices"]
            diag = model.decoder.last_sqmi_diagnostics
            count = diag["mask_boxes"].shape[1]
            for image_index, (source_indices, target_indices) in enumerate(indices):
                matched += source_indices.numel()
                keep = source_indices < count
                if not bool(keep.any()):
                    continue
                source = source_indices[keep]
                target_index = target_indices[keep]
                gt = normalized_targets[image_index]["boxes"][target_index]
                base = diag["base_boxes"][image_index, source]
                mask = diag["mask_boxes"][image_index, source]
                refined = diag["refined_boxes"][image_index, source]
                final = outputs["pred_boxes"][image_index, source]
                valid = diag["valid_mask"][image_index, source]
                values = {
                    "base": aligned_iou(base, gt),
                    "mask": aligned_iou(mask, gt),
                    "refined": aligned_iou(refined, gt),
                    "final": aligned_iou(final, gt),
                }
                number = source.numel()
                topk_matched += number
                valid_matched += int(valid.sum())
                for name, value in values.items():
                    totals[name] += float(value.sum())
                    valid_totals[name] += float(value[valid].sum()) if bool(valid.any()) else 0.0
                mask_better += int((values["mask"][valid] > values["base"][valid]).sum())
                refined_better += int((values["refined"][valid] > values["base"][valid]).sum())

    result = {
        "checkpoint": str(checkpoint),
        "batches": batches,
        "matched_targets": matched,
        "matched_in_topk": topk_matched,
        "topk_coverage": topk_matched / max(matched, 1),
        "valid_matched": valid_matched,
        "valid_ratio": valid_matched / max(topk_matched, 1),
        "mean_iou_topk": {name: value / max(topk_matched, 1) for name, value in totals.items()},
        "mean_iou_valid": {name: value / max(valid_matched, 1) for name, value in valid_totals.items()},
        "mask_better_than_base_ratio": mask_better / max(valid_matched, 1),
        "refined_better_than_base_ratio": refined_better / max(valid_matched, 1),
    }
    del model, criterion
    torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    result = {
        "best": inspect(args.config, args.run / "best_stg1.pth", args.batches),
        "last": inspect(args.config, args.run / "last.pth", args.batches),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
