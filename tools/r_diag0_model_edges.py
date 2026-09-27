#!/usr/bin/env python3
"""GPU R-DIAG0B: matched-query edge errors, FDR uncertainty and perturbation AP."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SIDES = ("left", "top", "right", "bottom")
SIZE_BINS = (("lt8", 0, 8), ("8to16", 8, 16), ("16to32", 16, 32),
             ("32to48", 32, 48), ("ge48", 48, float("inf")))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--weight-source", choices=("ema", "model"), default="ema")
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
            "q25": float(np.quantile(x, .25)), "median": float(np.median(x)),
            "q75": float(np.quantile(x, .75))}


def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def size_bin(box, height, width):
    edge = math.sqrt(float(box[2] * width * box[3] * height))
    return next(name for name, lo, hi in SIZE_BINS if lo <= edge < hi)


def val_targets_to_normalized(targets, input_h, input_w):
    converted = []
    for target in targets:
        boxes = target["boxes"].clone().float()
        boxes[:, [0, 2]] /= input_w
        boxes[:, [1, 3]] /= input_h
        boxes = torch.stack(((boxes[:, 0] + boxes[:, 2]) / 2,
                             (boxes[:, 1] + boxes[:, 3]) / 2,
                             boxes[:, 2] - boxes[:, 0],
                             boxes[:, 3] - boxes[:, 1]), dim=-1)
        converted.append({"boxes": boxes, "labels": target["labels"]})
    return converted


def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), -1)


def xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), -1)


def perturb_boxes(boxes, side, sign, mode, input_h, input_w):
    xyxy = cxcywh_to_xyxy(boxes).clone()
    index = SIDES.index(side)
    if mode == "pixel_1":
        delta = torch.full_like(xyxy[..., index], 1 / (input_w if index in (0, 2) else input_h))
    else:
        delta = 0.05 * (xyxy[..., 2] - xyxy[..., 0] if index in (0, 2)
                        else xyxy[..., 3] - xyxy[..., 1])
    xyxy[..., index] += sign * delta
    xyxy[..., [0, 2]] = xyxy[..., [0, 2]].clamp(0, 1)
    xyxy[..., [1, 3]] = xyxy[..., [1, 3]].clamp(0, 1)
    xyxy[..., 2] = torch.maximum(xyxy[..., 2], xyxy[..., 0] + 1e-6)
    xyxy[..., 3] = torch.maximum(xyxy[..., 3], xyxy[..., 1] + 1e-6)
    return xyxy_to_cxcywh(xyxy)


def add_detections(store, results, targets, category_ids):
    for target, result in zip(targets, results):
        boxes = result["boxes"].detach().cpu().clone()
        boxes[:, 2:] -= boxes[:, :2]
        for box, score, label in zip(boxes.tolist(), result["scores"].tolist(),
                                     result["labels"].tolist()):
            store.append({"image_id": int(target["image_id"]),
                          "category_id": int(category_ids[int(label)]),
                          "bbox": box, "score": float(score)})


def coco_metrics(coco_gt, detections, evaluator_cls):
    evaluator = evaluator_cls(coco_gt, coco_gt.loadRes(detections), "bbox")
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate(); evaluator.accumulate()
    precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]
    valid_mean = lambda x: float(x[x > -1].mean()) if np.any(x > -1) else None
    i75 = int(np.argmin(np.abs(evaluator.params.iouThrs - .75)))
    return {"AP": valid_mean(precision[:, :, :, 0, -1]),
            "AP75": valid_mean(precision[i75, :, :, 0, -1]),
            "AR100": valid_mean(recall[:, :, 0, -1])}


def main():
    a = parse_args()
    sys.path.insert(0, str(a.repo))
    from src.core import YAMLConfig
    from faster_coco_eval import COCOeval_faster

    cfg = YAMLConfig(str(a.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model.cuda().eval()
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    weights = checkpoint["ema"]["module"] if a.weight_source == "ema" else checkpoint["model"]
    model.load_state_dict(weights, strict=True)

    # Diagnostic-only full decoder exposure. Backbone/encoder and every BN stay in eval mode.
    model.decoder.train()
    model.decoder.num_denoising = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    loader, post, matcher = cfg.val_dataloader, cfg.postprocessor, cfg.criterion.matcher
    coco_gt = loader.dataset.coco
    category_ids = sorted(coco_gt.getCatIds())
    detections = defaultdict(list)
    edge_values = defaultdict(list)
    entropy_error = defaultdict(lambda: ([], []))
    kl_values = defaultdict(list)

    intervention_keys = [(mode, side, sign) for mode in ("pixel_1", "relative_5pct")
                         for side in SIDES for sign in (-1, 1)]
    with torch.inference_mode():
        for samples, targets in loader:
            samples = samples.cuda()
            targets_gpu = [{k: v.cuda() if torch.is_tensor(v) else v for k, v in t.items()}
                           for t in targets]
            input_h, input_w = samples.shape[-2:]
            normalized = val_targets_to_normalized(targets_gpu, input_h, input_w)
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
                outputs = model(samples)
            indices = matcher(outputs, normalized)["indices"]
            orig_sizes = torch.stack([t["orig_size"] for t in targets_gpu])
            add_detections(detections["baseline"], post(outputs, orig_sizes), targets_gpu, category_ids)
            for mode, side, sign in intervention_keys:
                changed = {"pred_logits": outputs["pred_logits"],
                           "pred_boxes": perturb_boxes(outputs["pred_boxes"], side, sign,
                                                       mode, input_h, input_w)}
                add_detections(detections[f"{mode}/{side}/{sign:+d}"],
                               post(changed, orig_sizes), targets_gpu, category_ids)

            layers = list(outputs.get("aux_outputs", [])) + [outputs]
            final_prob = F.softmax(outputs["pred_corners"].reshape(
                *outputs["pred_corners"].shape[:2], 4, -1).float(), -1)
            for batch_idx, (query_idx_cpu, target_idx_cpu) in enumerate(indices):
                query_idx, target_idx = query_idx_cpu.cuda(), target_idx_cpu.cuda()
                pred_xyxy = cxcywh_to_xyxy(outputs["pred_boxes"][batch_idx, query_idx]).float()
                gt = normalized[batch_idx]["boxes"][target_idx].float()
                gt_xyxy = cxcywh_to_xyxy(gt)
                scale = torch.tensor([input_w, input_h, input_w, input_h], device=gt.device)
                abs_error = (pred_xyxy - gt_xyxy).abs() * scale
                relative_scale = torch.stack((gt[:, 2], gt[:, 3], gt[:, 2], gt[:, 3]), -1)
                rel_error = (pred_xyxy - gt_xyxy).abs() / relative_scale.clamp_min(1e-9)
                groups = [size_bin(box, input_h, input_w) for box in gt]
                for local_idx, group in enumerate(groups):
                    for side_idx, side in enumerate(SIDES):
                        for group_name in ("all", group):
                            edge_values[("abs_px", side, group_name)].append(
                                float(abs_error[local_idx, side_idx]))
                            edge_values[("relative", side, group_name)].append(
                                float(rel_error[local_idx, side_idx]))

                for layer_idx, layer in enumerate(layers):
                    corner_logits = layer["pred_corners"][batch_idx, query_idx].reshape(-1, 4, 33).float()
                    probability = F.softmax(corner_logits, -1)
                    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1) / math.log(33)
                    maximum = probability.max(-1).values
                    final_match = final_prob[batch_idx, query_idx]
                    kl = (final_match * (final_match.clamp_min(1e-12).log() -
                          probability.clamp_min(1e-12).log())).sum(-1)
                    for local_idx, group in enumerate(groups):
                        for side_idx, side in enumerate(SIDES):
                            for group_name in ("all", group):
                                edge_values[(f"entropy_l{layer_idx}", side, group_name)].append(
                                    float(entropy[local_idx, side_idx]))
                                edge_values[(f"maxprob_l{layer_idx}", side, group_name)].append(
                                    float(maximum[local_idx, side_idx]))
                                kl_values[(f"layer{layer_idx}_to_final", side, group_name)].append(
                                    float(kl[local_idx, side_idx]))
                            entropy_error[side][0].append(float(entropy[local_idx, side_idx]))
                            entropy_error[side][1].append(float(abs_error[local_idx, side_idx]))

    metric_summary = {"baseline": coco_metrics(coco_gt, detections["baseline"], COCOeval_faster)}
    for key in sorted(k for k in detections if k != "baseline"):
        metric_summary[key] = coco_metrics(coco_gt, detections[key], COCOeval_faster)
    paired = {}
    for mode in ("pixel_1", "relative_5pct"):
        paired[mode] = {}
        for side in SIDES:
            plus, minus = metric_summary[f"{mode}/{side}/+1"], metric_summary[f"{mode}/{side}/-1"]
            paired[mode][side] = {name: float((plus[name] + minus[name]) / 2 -
                                                    metric_summary["baseline"][name])
                                  for name in ("AP", "AP75", "AR100")}

    result = {
        "checkpoint": str(a.checkpoint), "weight_source": a.weight_source,
        "test_used": False, "baseline_metrics": metric_summary["baseline"],
        "paired_side_perturbation_delta": paired,
        "edge_statistics": {"/".join(key): stats(value) for key, value in edge_values.items()},
        "cross_layer_kl": {"/".join(key): stats(value) for key, value in kl_values.items()},
        "entropy_abs_error_pearson": {side: pearson(*entropy_error[side]) for side in SIDES},
        "decision_note": "Geometry-only asymmetry is insufficient; inspect relative errors, normalized entropy and relative perturbation before enabling R-FDR/R-LSD.",
    }
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
