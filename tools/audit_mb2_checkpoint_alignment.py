"""Audit learned M-B2 target alignment and gating on the original test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def cxcywh_to_xyxy(boxes):
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2, boxes[..., :2] + boxes[..., 2:] / 2), -1)


def pair_iou(boxes, target):
    left_top = torch.maximum(boxes[:, :2], target[:2])
    right_bottom = torch.minimum(boxes[:, 2:], target[2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(-1)
    area_boxes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(-1)
    area_target = (target[2:] - target[:2]).clamp_min(0).prod()
    return intersection / (area_boxes + area_target - intersection).clamp_min(1e-12)


def describe(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array), "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5)), "p90": float(np.quantile(array, 0.9)),
    }


def normalized_xyxy(box, height, width):
    value = torch.as_tensor(box).float()
    return value / value.new_tensor([width, height, width, height])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mismatch-offset", type=int, default=670)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"]
    scenarios = {"normal": 0, "global_mismatch": args.mismatch_offset}
    result = {}
    for name, offset in scenarios.items():
        cfg = YAMLConfig(str(args.config.resolve()))
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        cfg.yaml_cfg["val_dataloader"]["dataset"]["infrared_index_offset"] = offset
        model = cfg.model
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        loader = cfg.val_dataloader
        mapped_errors, aligned_errors, gates, deltas, visible_ious = [], [], [], [], []
        for samples, targets in loader:
            samples = samples.to(device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(samples)
            diagnostics = model.decoder.sdtec_aligned_calibrator.last_diagnostics
            predicted_xyxy = cxcywh_to_xyxy(outputs["pred_boxes"].float())
            scores = outputs["pred_logits"].float().sigmoid().amax(-1)
            for batch, target in enumerate(targets):
                if len(target["boxes"]) != 1 or len(target["infrared_boxes"]) != 1:
                    continue
                height, width = samples.shape[-2:]
                visible_gt = normalized_xyxy(target["boxes"][0], height, width).to(device)
                thermal_gt = normalized_xyxy(target["infrared_boxes"][0], height, width).to(device)
                candidate_iou = pair_iou(predicted_xyxy[batch], visible_gt)
                # Prefer a real RGB target proposal; confidence breaks duplicate-IoU ties.
                query = int(torch.argmax(candidate_iou + 1e-4 * scores[batch]))
                visible_ious.append(float(candidate_iou[query]))
                thermal_center = (thermal_gt[:2] + thermal_gt[2:]) / 2
                mapped = diagnostics["mapped_centres"][batch, query].float()
                aligned = diagnostics["aligned_centres"][batch, query].float()
                mapped_errors.append(float(torch.linalg.vector_norm(mapped - thermal_center)))
                aligned_errors.append(float(torch.linalg.vector_norm(aligned - thermal_center)))
                gates.append(float(diagnostics["gate"][batch, query]))
                deltas.append(float(diagnostics["delta"][batch, query].float().abs().mean()))
        result[name] = {
            "visible_selected_iou": describe(visible_ious),
            "coarse_mapped_center_error": describe(mapped_errors),
            "learned_aligned_center_error": describe(aligned_errors),
            "selected_query_gate": describe(gates),
            "selected_query_abs_logit_delta": describe(deltas),
        }
        del model, loader
        torch.cuda.empty_cache()
    normal = result["normal"]
    mismatch = result["global_mismatch"]
    report = {
        "schema": "mb2_checkpoint_alignment_audit_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "test_used_for_training_or_threshold": False,
        "results": result,
        "contrasts": {
            "normal_aligned_minus_coarse_p50_error": (
                normal["learned_aligned_center_error"]["p50"]
                - normal["coarse_mapped_center_error"]["p50"]
            ),
            "normal_minus_mismatch_gate_mean": (
                normal["selected_query_gate"]["mean"]
                - mismatch["selected_query_gate"]["mean"]
            ),
            "normal_minus_mismatch_delta_mean": (
                normal["selected_query_abs_logit_delta"]["mean"]
                - mismatch["selected_query_abs_logit_delta"]["mean"]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
