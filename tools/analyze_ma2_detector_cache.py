#!/usr/bin/env python3
"""Audit M-A2 frozen proposals before any fusion parameter is selected."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


# Fitted on train annotations only.  Points are row vectors:
# [x_ir, y_ir, 1] @ IR_TO_VISIBLE -> [x_visible, y_visible].
IR_TO_VISIBLE = torch.tensor(
    [
        [0.8143358614759916, 0.05263852735114608],
        [0.011159181806002685, 1.181970253466458],
        [0.11291937665160638, -0.1689762267210995],
    ],
    dtype=torch.float32,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def cxcywh_to_xyxy(boxes):
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def map_ir_xyxy(boxes, matrix=IR_TO_VISIBLE):
    x1, y1, x2, y2 = boxes.unbind(-1)
    corners = torch.stack(
        (
            torch.stack((x1, y1), -1),
            torch.stack((x2, y1), -1),
            torch.stack((x1, y2), -1),
            torch.stack((x2, y2), -1),
        ),
        dim=-2,
    )
    ones = torch.ones_like(corners[..., :1])
    mapped = torch.cat((corners, ones), dim=-1) @ matrix
    lower = mapped.amin(dim=-2)
    upper = mapped.amax(dim=-2)
    return torch.cat((lower, upper), -1).clamp(0.0, 1.0)


def pairwise_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lower = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    upper = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (upper - lower).clamp_min(0).prod(-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(-1)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp_min(1e-12)


def quantiles(values):
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.numel() == 0:
        return {"count": 0}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "p10": float(torch.quantile(values, 0.10)),
        "p25": float(torch.quantile(values, 0.25)),
        "p50": float(torch.quantile(values, 0.50)),
        "p75": float(torch.quantile(values, 0.75)),
        "p90": float(torch.quantile(values, 0.90)),
    }


def coco_detections(image_ids, orig_sizes, scores, xyxy):
    detections = []
    for index, image_id in enumerate(image_ids.tolist()):
        # D-FINE targets store orig_size as [width, height].
        width, height = orig_sizes[index].tolist()
        boxes = xyxy[index].clone()
        boxes[:, 0::2] *= width
        boxes[:, 1::2] *= height
        boxes[:, 2:] -= boxes[:, :2]
        for box, score in zip(boxes.tolist(), scores[index].tolist()):
            if box[2] <= 0 or box[3] <= 0 or not math.isfinite(score):
                continue
            detections.append(
                {
                    "image_id": int(image_id),
                    "category_id": 0,
                    "bbox": box,
                    "score": float(score),
                }
            )
    return detections


def coco_eval(annotation_path, detections):
    from faster_coco_eval import COCO, COCOeval_faster

    coco_gt = COCO(str(annotation_path), print_function=lambda *_args, **_kwargs: None)
    coco_dt = coco_gt.loadRes(detections, min_score=0.0)
    evaluator = COCOeval_faster(
        coco_gt, coco_dt, "bbox", print_function=lambda *_args, **_kwargs: None
    )
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = (
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    )
    return {name: float(value) for name, value in zip(names, evaluator.stats)}


def gt_audit(annotation, cache, visible_xyxy, thermal_mapped_xyxy):
    anns_by_image = defaultdict(list)
    image_info = {int(image["id"]): image for image in annotation["images"]}
    for ann in annotation["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        image = image_info[int(ann["image_id"])]
        x, y, w, h = ann["bbox"]
        anns_by_image[int(ann["image_id"])].append(
            torch.tensor(
                [x / image["width"], y / image["height"], (x + w) / image["width"], (y + h) / image["height"]],
                dtype=torch.float32,
            )
        )

    visible_scores = cache["visible_logits"].squeeze(-1).sigmoid()
    thermal_scores = cache["thermal_logits"].squeeze(-1).sigmoid()
    topks = (1, 3, 5, 10, 100)
    result = {}
    for modality, boxes_all, scores_all in (
        ("visible", visible_xyxy, visible_scores),
        ("thermal_mapped", thermal_mapped_xyxy, thermal_scores),
    ):
        by_k = {}
        for topk in topks:
            max_ious = []
            min_center_distances = []
            for index, image_id in enumerate(cache["image_ids"].tolist()):
                gt_list = anns_by_image.get(int(image_id), [])
                if not gt_list:
                    continue
                gt = torch.stack(gt_list)
                indices = scores_all[index].topk(min(topk, scores_all.shape[1])).indices
                candidates = boxes_all[index, indices]
                ious = pairwise_iou(gt, candidates)
                max_ious.extend(ious.amax(dim=1).tolist())
                gt_centers = (gt[:, :2] + gt[:, 2:]) / 2
                centers = (candidates[:, :2] + candidates[:, 2:]) / 2
                distances = torch.cdist(gt_centers, centers)
                min_center_distances.extend(distances.amin(dim=1).tolist())
            ious = torch.tensor(max_ious)
            distances = torch.tensor(min_center_distances)
            by_k[str(topk)] = {
                "gt_count": int(ious.numel()),
                "recall_iou_0.3": float((ious >= 0.3).float().mean()),
                "recall_iou_0.5": float((ious >= 0.5).float().mean()),
                "center_recall_0.03": float((distances <= 0.03).float().mean()),
                "center_recall_0.05": float((distances <= 0.05).float().mean()),
                "center_recall_0.10": float((distances <= 0.10).float().mean()),
                "max_iou": quantiles(ious),
                "min_center_distance": quantiles(distances),
            }
        result[modality] = by_k
    return result


def main():
    args = parse_args()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    annotation = json.loads(args.annotations.read_text(encoding="utf-8"))
    image_ids = cache["image_ids"].long()
    annotation_ids = {int(image["id"]) for image in annotation["images"]}
    if set(image_ids.tolist()) != annotation_ids:
        raise RuntimeError("Cache image ids do not exactly match annotation image ids")

    visible_scores = cache["visible_logits"].squeeze(-1).sigmoid()
    thermal_scores = cache["thermal_logits"].squeeze(-1).sigmoid()
    visible_xyxy = cxcywh_to_xyxy(cache["visible_boxes"]).clamp(0.0, 1.0)
    thermal_xyxy = cxcywh_to_xyxy(cache["thermal_boxes"])
    thermal_mapped_xyxy = map_ir_xyxy(thermal_xyxy)

    visible_detections = coco_detections(
        image_ids, cache["orig_sizes"], visible_scores, visible_xyxy
    )
    thermal_detections = coco_detections(
        image_ids, cache["orig_sizes"], thermal_scores, thermal_mapped_xyxy
    )
    positive_ids = {int(ann["image_id"]) for ann in annotation["annotations"] if not ann.get("iscrowd", 0)}
    positive_mask = torch.tensor([int(value) in positive_ids for value in image_ids.tolist()])
    visible_top1 = visible_scores.amax(1)
    thermal_top1 = thermal_scores.amax(1)

    report = {
        "schema": "ma2_detector_cache_audit_v1",
        "cache": str(args.cache.resolve()),
        "annotations": str(args.annotations.resolve()),
        "samples": int(image_ids.numel()),
        "positive_images": int(positive_mask.sum()),
        "empty_images": int((~positive_mask).sum()),
        "visible_top1_score": {
            "all": quantiles(visible_top1),
            "positive": quantiles(visible_top1[positive_mask]),
            "empty": quantiles(visible_top1[~positive_mask]),
        },
        "thermal_top1_score": {
            "all": quantiles(thermal_top1),
            "positive": quantiles(thermal_top1[positive_mask]),
            "empty": quantiles(thermal_top1[~positive_mask]),
        },
        "coco": {
            "visible_baseline": coco_eval(args.annotations, visible_detections),
            "thermal_mapped": coco_eval(args.annotations, thermal_detections),
        },
        "gt_candidate_recall": gt_audit(
            annotation, cache, visible_xyxy, thermal_mapped_xyxy
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
