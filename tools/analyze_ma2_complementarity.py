#!/usr/bin/env python3
"""Measure whether mapped Thermal proposals complement RGB ranking failures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from analyze_ma2_detector_cache import cxcywh_to_xyxy, map_ir_xyxy, pairwise_iou


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    annotation = json.loads(args.annotations.read_text(encoding="utf-8"))
    image_info = {int(item["id"]): item for item in annotation["images"]}
    targets = defaultdict(list)
    for ann in annotation["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        image = image_info[int(ann["image_id"])]
        x, y, w, h = ann["bbox"]
        targets[int(ann["image_id"])].append(
            torch.tensor(
                [x / image["width"], y / image["height"], (x + w) / image["width"], (y + h) / image["height"]]
            )
        )

    visible_scores = cache["visible_logits"].squeeze(-1).float().sigmoid()
    thermal_scores = cache["thermal_logits"].squeeze(-1).float().sigmoid()
    visible_boxes = cxcywh_to_xyxy(cache["visible_boxes"].float()).clamp(0.0, 1.0)
    thermal_boxes = map_ir_xyxy(cxcywh_to_xyxy(cache["thermal_boxes"].float()))

    positive = 0
    rgb_top1_correct = 0
    rgb_top3_reachable = 0
    rgb_top100_reachable = 0
    rgb_ranking_failures = []
    thermal_near_gt = {key: 0 for key in (1, 3, 10, 100)}
    complementary = {
        key: {"rgb_top1_miss": 0, "rgb_top3_miss_but_top100_reachable": 0}
        for key in (1, 3, 10, 100)
    }
    paired_candidate_support = {key: 0 for key in (1, 3, 10, 100)}

    for row, image_id in enumerate(cache["image_ids"].tolist()):
        gt_list = targets.get(int(image_id), [])
        if not gt_list:
            continue
        # Anti-UAV has at most one visible target per retained frame.  Keep the
        # implementation explicit in case a future split has several boxes.
        for gt in gt_list:
            positive += 1
            ious = pairwise_iou(gt[None], visible_boxes[row]).squeeze(0)
            score_order = visible_scores[row].argsort(descending=True)
            correct_order_positions = torch.nonzero(ious[score_order] >= 0.5).flatten()
            if correct_order_positions.numel() == 0:
                continue
            first_rank = int(correct_order_positions[0]) + 1
            if first_rank == 1:
                rgb_top1_correct += 1
            if first_rank <= 3:
                rgb_top3_reachable += 1
            if first_rank <= 100:
                rgb_top100_reachable += 1
            if first_rank > 1 and first_rank <= 100:
                rgb_ranking_failures.append((row, gt, score_order[first_rank - 1]))

            thermal_order = thermal_scores[row].argsort(descending=True)
            gt_center = (gt[:2] + gt[2:]) / 2
            for topk in (1, 3, 10, 100):
                ir_indices = thermal_order[:topk]
                ir_boxes = thermal_boxes[row, ir_indices]
                ir_centers = (ir_boxes[:, :2] + ir_boxes[:, 2:]) / 2
                near_gt = torch.linalg.vector_norm(ir_centers - gt_center, dim=1).amin() <= 0.05
                if near_gt:
                    thermal_near_gt[topk] += 1
                if first_rank > 1 and first_rank <= 100 and near_gt:
                    complementary[topk]["rgb_top1_miss"] += 1
                if first_rank > 3 and first_rank <= 100 and near_gt:
                    complementary[topk]["rgb_top3_miss_but_top100_reachable"] += 1

                correct_box = visible_boxes[row, score_order[first_rank - 1]]
                correct_center = (correct_box[:2] + correct_box[2:]) / 2
                near_correct_candidate = (
                    torch.linalg.vector_norm(ir_centers - correct_center, dim=1).amin() <= 0.03
                )
                if first_rank > 1 and first_rank <= 100 and near_correct_candidate:
                    paired_candidate_support[topk] += 1

    report = {
        "schema": "ma2_complementarity_audit_v1",
        "cache": str(args.cache.resolve()),
        "annotations": str(args.annotations.resolve()),
        "center_radius_for_thermal_gt_support": 0.05,
        "center_radius_for_matched_rgb_candidate_support": 0.03,
        "positive_targets": positive,
        "rgb_reachability": {
            "top1_correct": rgb_top1_correct,
            "top1_correct_fraction": rgb_top1_correct / positive,
            "top3_reachable": rgb_top3_reachable,
            "top3_reachable_fraction": rgb_top3_reachable / positive,
            "top100_reachable": rgb_top100_reachable,
            "top100_reachable_fraction": rgb_top100_reachable / positive,
            "ranking_failures_top1_to_top100": len(rgb_ranking_failures),
            "ranking_failures_fraction": len(rgb_ranking_failures) / positive,
        },
        "thermal_center_support_of_all_visible_gt": {
            str(topk): {
                "count": thermal_near_gt[topk],
                "fraction": thermal_near_gt[topk] / positive,
            }
            for topk in (1, 3, 10, 100)
        },
        "thermal_support_on_rgb_ranking_failures": {
            str(topk): {
                **complementary[topk],
                "fraction_of_all_positive": complementary[topk]["rgb_top1_miss"] / positive,
                "fraction_of_ranking_failures": complementary[topk]["rgb_top1_miss"]
                / max(1, len(rgb_ranking_failures)),
                "correct_rgb_candidate_center_supported_count": paired_candidate_support[topk],
                "correct_rgb_candidate_center_supported_fraction_of_ranking_failures": paired_candidate_support[topk]
                / max(1, len(rgb_ranking_failures)),
            }
            for topk in (1, 3, 10, 100)
        },
        "interpretation": {
            "question": "Can a mapped Thermal proposal identify the RGB candidate that would repair a ranking failure?",
            "warning": "This is a GT-based headroom audit only; it is not an implementable test-time fusion rule.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
