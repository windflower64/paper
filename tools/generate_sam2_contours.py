#!/usr/bin/env python3
"""Generate high-confidence UAV pseudo-contours from GT boxes with SAM 2.1.

This script is a diagnostic data-preparation tool.  Its masks are never treated
as ground-truth segmentation.  It uses region-adaptive magnification and rejects
prompt-unstable masks before any detector intervention is attempted.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--sam2-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--main-crop-scale", type=float, default=8.0)
    parser.add_argument("--check-crop-scale", type=float, default=10.0)
    parser.add_argument("--pred-iou-threshold", type=float, default=0.55)
    parser.add_argument("--stability-threshold", type=float, default=0.75)
    parser.add_argument("--min-area-ratio", type=float, default=0.08)
    parser.add_argument("--max-area-ratio", type=float, default=1.20)
    parser.add_argument("--min-inside-ratio", type=float, default=0.80)
    parser.add_argument("--min-mask-box-iou", type=float, default=0.20)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--audit-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def square_crop(width: int, height: int, box: np.ndarray, scale: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.tolist()
    side = max(32.0, max(x2 - x1, y2 - y1) * scale)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = int(math.floor(cx - side / 2))
    top = int(math.floor(cy - side / 2))
    right = int(math.ceil(cx + side / 2))
    bottom = int(math.ceil(cy + side / 2))
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    return max(0, left), max(0, top), min(width, right), min(height, bottom)


def jitter_boxes(box: np.ndarray, crop_width: int, crop_height: int) -> list[np.ndarray]:
    x1, y1, x2, y2 = box.tolist()
    width, height = x2 - x1, y2 - y1
    variants = [box.copy()]
    for expand, shift_x, shift_y in ((0.08, 0, 0), (-0.05, 0, 0), (0, 0.03, -0.02)):
        dx, dy = width * expand / 2, height * expand / 2
        sx, sy = width * shift_x, height * shift_y
        variants.append(
            np.asarray(
                [
                    max(0, x1 - dx + sx),
                    max(0, y1 - dy + sy),
                    min(crop_width - 1, x2 + dx + sx),
                    min(crop_height - 1, y2 + dy + sy),
                ],
                dtype=np.float32,
            )
        )
    return variants


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.logical_or(mask_a, mask_b).sum()
    return float(np.logical_and(mask_a, mask_b).sum() / union) if union else 0.0


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def annotation_box_to_image_box(
    annotation_box: np.ndarray,
    annotation_width: float,
    annotation_height: float,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, float, float]:
    """Map annotation coordinates to the actual image loaded from disk."""
    if annotation_width <= 0 or annotation_height <= 0:
        raise ValueError("Annotation image dimensions must be positive")
    scale_x = float(image_width) / float(annotation_width)
    scale_y = float(image_height) / float(annotation_height)
    x1, y1, x2, y2 = annotation_box.tolist()
    image_box = np.asarray(
        [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
        dtype=np.float32,
    )
    return image_box, scale_x, scale_y


def component_at_center(mask: np.ndarray, center: tuple[int, int]) -> tuple[np.ndarray, bool]:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    x, y = center
    if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
        return np.zeros_like(mask), False
    label = int(labels[y, x])
    if label == 0:
        return np.zeros_like(mask), False
    return labels == label, count > 1


def paste_mask(mask: np.ndarray, crop: tuple[int, int, int, int], image_shape: tuple[int, int]) -> np.ndarray:
    output = np.zeros(image_shape, dtype=bool)
    left, top, right, bottom = crop
    output[top:bottom, left:right] = mask[: bottom - top, : right - left]
    return output


def predict_crop(predictor, image: np.ndarray, box: np.ndarray, crop: tuple[int, int, int, int], variants: bool):
    left, top, right, bottom = crop
    crop_image = image[top:bottom, left:right]
    local_box = box - np.asarray([left, top, left, top], dtype=np.float32)
    center = np.asarray([[(local_box[0] + local_box[2]) / 2, (local_box[1] + local_box[3]) / 2]], dtype=np.float32)
    boxes = jitter_boxes(local_box, crop_image.shape[1], crop_image.shape[0]) if variants else [local_box]
    predictor.set_image(crop_image)
    masks, scores = [], []
    for candidate_box in boxes:
        prediction, score, _ = predictor.predict(
            point_coords=center,
            point_labels=np.asarray([1], dtype=np.int32),
            box=candidate_box,
            multimask_output=False,
        )
        masks.append(paste_mask(prediction[0].astype(bool), crop, image.shape[:2]))
        scores.append(float(score[0]))
    return masks, scores


def mask_box(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def expanded_inside_ratio(mask: np.ndarray, box: np.ndarray, expansion: float = 0.15) -> float:
    x1, y1, x2, y2 = box.tolist()
    width, height = x2 - x1, y2 - y1
    x1 = max(0, int(math.floor(x1 - width * expansion)))
    y1 = max(0, int(math.floor(y1 - height * expansion)))
    x2 = min(mask.shape[1], int(math.ceil(x2 + width * expansion)))
    y2 = min(mask.shape[0], int(math.ceil(y2 + height * expansion)))
    area = int(mask.sum())
    return float(mask[y1:y2, x1:x2].sum() / area) if area else 0.0


def save_overlay(image: np.ndarray, mask: np.ndarray, box: np.ndarray, record: dict, path: Path) -> None:
    x1, y1, x2, y2 = box.tolist()
    pad = max(x2 - x1, y2 - y1) * 1.5
    left, top = max(0, int(x1 - pad)), max(0, int(y1 - pad))
    right, bottom = min(image.shape[1], int(x2 + pad)), min(image.shape[0], int(y2 + pad))
    crop = Image.fromarray(image[top:bottom, left:right]).convert("RGB").resize((384, 384))
    local_mask = Image.fromarray((mask[top:bottom, left:right] * 255).astype(np.uint8)).resize((384, 384), Image.Resampling.NEAREST)
    rgb = np.asarray(crop).copy()
    binary = np.asarray(local_mask) > 0
    contour = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    rgb[binary] = (0.65 * rgb[binary] + 0.35 * np.asarray([40, 230, 80])).astype(np.uint8)
    rgb[contour] = np.asarray([255, 220, 0], dtype=np.uint8)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    sx, sy = 384 / max(1, right - left), 384 / max(1, bottom - top)
    draw.rectangle(((x1 - left) * sx, (y1 - top) * sy, (x2 - left) * sx, (y2 - top) * sy), outline=(0, 255, 255), width=2)
    label = f"{'PASS' if record['accepted'] else 'REJECT'} score={record['pred_iou']:.3f} stab={record['stability']:.3f}"
    draw.rectangle((0, 0, 384, 24), fill=(0, 0, 0))
    draw.text((4, 4), label, fill=(255, 255, 255))
    canvas.save(path, quality=90)


def summarize(records: list[dict], elapsed: float) -> dict:
    accepted = [item for item in records if item["accepted"]]
    reasons = Counter(reason for item in records for reason in item["rejection_reasons"])
    bins = defaultdict(lambda: {"total": 0, "accepted": 0})
    for item in records:
        side = item["resized_max_side_640x512"]
        name = "lt16" if side < 16 else "16to32" if side < 32 else "32to48" if side < 48 else "ge48"
        bins[name]["total"] += 1
        bins[name]["accepted"] += int(item["accepted"])
    for values in bins.values():
        values["acceptance_rate"] = values["accepted"] / max(1, values["total"])
    return {
        "records": len(records),
        "accepted": len(accepted),
        "acceptance_rate": len(accepted) / max(1, len(records)),
        "elapsed_seconds": elapsed,
        "rejection_reasons": dict(reasons),
        "size_bins": dict(bins),
        "decision_gate": {
            "minimum_acceptance_rate": 0.70,
            "automatic_pass": len(accepted) / max(1, len(records)) >= 0.70,
            "warning": "Automatic pass is necessary but not sufficient; the 200-mask visual audit is mandatory.",
        },
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = args.output_dir / "masks"
    records_dir = args.output_dir / "records"
    overlays_dir = args.output_dir / "audit_overlays"
    for directory in (masks_dir, records_dir, overlays_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in data["images"]}
    annotations = data["annotations"]
    if args.max_images is not None:
        annotations = annotations[: args.max_images]

    import sys

    sys.path.insert(0, str(args.sam2_repo))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not torch.cuda.is_available():
        raise RuntimeError("SAM2 full generation requires a CUDA GPU; prepare only while AutoDL is in no-card mode.")
    model = build_sam2(args.model_config, str(args.checkpoint), device="cuda")
    predictor = SAM2ImagePredictor(model)

    records = []
    audit_indices = set(np.linspace(0, max(0, len(annotations) - 1), min(args.audit_limit, len(annotations)), dtype=int).tolist())
    started = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, ann in enumerate(annotations):
            image_info = images[int(ann["image_id"])]
            record_path = records_dir / f"{int(ann['image_id']):06d}.json"
            if record_path.exists() and not args.overwrite:
                record = json.loads(record_path.read_text(encoding="utf-8"))
                records.append(record)
                continue
            image = np.asarray(Image.open(args.image_root / image_info["file_name"]).convert("RGB"))
            x, y, width, height = [float(value) for value in ann["bbox"]]
            annotation_box = np.asarray([x, y, x + width, y + height], dtype=np.float32)
            box, scale_x, scale_y = annotation_box_to_image_box(
                annotation_box,
                float(image_info.get("width", image.shape[1])),
                float(image_info.get("height", image.shape[0])),
                image.shape[1],
                image.shape[0],
            )
            width = float(box[2] - box[0])
            height = float(box[3] - box[1])
            main_crop = square_crop(image.shape[1], image.shape[0], box, args.main_crop_scale)
            check_crop = square_crop(image.shape[1], image.shape[0], box, args.check_crop_scale)
            main_masks, main_scores = predict_crop(predictor, image, box, main_crop, variants=True)
            check_masks, check_scores = predict_crop(predictor, image, box, check_crop, variants=False)
            candidates = main_masks + check_masks
            scores = main_scores + check_scores
            consensus = np.sum(np.stack(candidates, axis=0), axis=0) >= math.ceil(len(candidates) / 2)
            center = (int(round((box[0] + box[2]) / 2)), int(round((box[1] + box[3]) / 2)))
            consensus, center_component = component_at_center(consensus, center)
            stability = float(np.median([iou(candidates[0], item) for item in candidates[1:]]))
            predicted_iou = float(scores[0])
            area_ratio = float(consensus.sum() / max(1.0, width * height))
            inside_ratio = expanded_inside_ratio(consensus, box)
            output_box = mask_box(consensus)
            mask_bbox_iou = box_iou(output_box, box) if output_box is not None else 0.0
            reasons = []
            if predicted_iou < args.pred_iou_threshold:
                reasons.append("low_pred_iou")
            if stability < args.stability_threshold:
                reasons.append("prompt_or_crop_unstable")
            if not center_component:
                reasons.append("center_not_in_component")
            if not args.min_area_ratio <= area_ratio <= args.max_area_ratio:
                reasons.append("implausible_mask_box_area_ratio")
            if inside_ratio < args.min_inside_ratio:
                reasons.append("mask_leaks_outside_box")
            if mask_bbox_iou < args.min_mask_box_iou:
                reasons.append("mask_bbox_mismatch")
            resized_width = width * 640 / image.shape[1]
            resized_height = height * 512 / image.shape[0]
            record = {
                "image_id": int(ann["image_id"]),
                "annotation_id": int(ann["id"]),
                "file_name": image_info["file_name"],
                "bbox_xyxy": box.tolist(),
                "bbox_xyxy_annotation": annotation_box.tolist(),
                "annotation_image_size": [
                    int(image_info.get("width", image.shape[1])),
                    int(image_info.get("height", image.shape[0])),
                ],
                "actual_image_size": [int(image.shape[1]), int(image.shape[0])],
                "bbox_scale_xy": [scale_x, scale_y],
                "pred_iou": predicted_iou,
                "all_pred_iou": scores,
                "stability": stability,
                "mask_box_area_ratio": area_ratio,
                "inside_expanded_box_ratio": inside_ratio,
                "mask_bbox_iou": mask_bbox_iou,
                "center_component": bool(center_component),
                "accepted": not reasons,
                "rejection_reasons": reasons,
                "resized_width_640x512": resized_width,
                "resized_height_640x512": resized_height,
                "resized_max_side_640x512": max(resized_width, resized_height),
                "mask_path": str(masks_dir / f"{int(ann['image_id']):06d}.png"),
            }
            Image.fromarray((consensus * 255).astype(np.uint8)).save(record["mask_path"], optimize=True)
            record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            if index in audit_indices:
                save_overlay(image, consensus, box, record, overlays_dir / f"{int(ann['image_id']):06d}.webp")
            records.append(record)
            if index == 0 or (index + 1) % 50 == 0:
                rate = sum(item["accepted"] for item in records) / len(records)
                print(f"processed={index + 1}/{len(annotations)} acceptance={rate:.3f}", flush=True)

    records.sort(key=lambda item: item["image_id"])
    (args.output_dir / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize(records, time.time() - started)
    (args.output_dir / "quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
