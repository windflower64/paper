"""Audit the fixed SAM cache used by STQL and save deterministic examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    records_path = args.mask_root / "records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in coco["images"]}
    annotations = defaultdict(list)
    for row in coco["annotations"]:
        annotations[int(row["image_id"])].append(row)
    records_by_id = {int(row["image_id"]): row for row in records}

    target_counts = Counter(len(annotations[image_id]) for image_id in images)
    accepted_by_targets = Counter()
    missing_records = []
    missing_masks = []
    unique_values = Counter()
    component_counts = []
    mask_box_area_ratios = []
    inside_box_ratios = []
    mask_bbox_ious = []
    accepted_ids = []

    for image_id, image_info in images.items():
        record = records_by_id.get(image_id)
        if record is None:
            missing_records.append(image_id)
            continue
        if not bool(record.get("accepted", False)):
            continue
        accepted_by_targets[len(annotations[image_id])] += 1
        mask_path = args.mask_root / "masks" / f"{image_id:06d}.png"
        if not mask_path.is_file():
            missing_masks.append(image_id)
            continue
        mask_raw = np.asarray(Image.open(mask_path).convert("L"))
        unique_values.update(int(value) for value in np.unique(mask_raw))
        mask = mask_raw > 0
        count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        component_counts.append(int(count - 1))
        accepted_ids.append(image_id)
        anns = annotations[image_id]
        if len(anns) == 1:
            x, y, width, height = anns[0]["bbox"]
            x0 = max(0, int(np.floor(x)))
            y0 = max(0, int(np.floor(y)))
            x1 = min(mask.shape[1], int(np.ceil(x + width)))
            y1 = min(mask.shape[0], int(np.ceil(y + height)))
            mask_area = int(mask.sum())
            box_area = max(float(width * height), 1e-12)
            intersection = int(mask[y0:y1, x0:x1].sum())
            mask_box_area_ratios.append(mask_area / box_area)
            inside_box_ratios.append(intersection / max(mask_area, 1))
            mask_bbox_ious.append(float(record.get("mask_bbox_iou", 0.0)))

    if missing_records or missing_masks:
        raise RuntimeError(
            f"cache incomplete: missing_records={missing_records[:10]}, "
            f"missing_masks={missing_masks[:10]}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output_dir / "sam_examples"
    visual_dir.mkdir(parents=True, exist_ok=True)
    if accepted_ids:
        positions = np.linspace(0, len(accepted_ids) - 1, num=min(5, len(accepted_ids)))
        chosen = [accepted_ids[int(round(value))] for value in positions]
        for image_id in chosen:
            info = images[image_id]
            image = Image.open(args.image_root / info["file_name"]).convert("RGB")
            mask = np.asarray(
                Image.open(args.mask_root / "masks" / f"{image_id:06d}.png").convert("L")
            ) > 0
            rgb = np.asarray(image).copy()
            overlay = rgb.copy()
            overlay[mask] = np.array([255, 64, 64], dtype=np.uint8)
            rgb = (0.62 * rgb + 0.38 * overlay).astype(np.uint8)
            rendered = Image.fromarray(rgb)
            draw = ImageDraw.Draw(rendered)
            for ann in annotations[image_id]:
                x, y, width, height = ann["bbox"]
                draw.rectangle((x, y, x + width, y + height), outline=(0, 255, 0), width=2)
            rendered.save(visual_dir / f"{image_id:06d}_{info['file_name']}.png")

    audit = {
        "schema": "stql_sam_cache_audit_v1",
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": sha256(args.annotations),
        "records": str(records_path.resolve()),
        "records_sha256": sha256(records_path),
        "images": len(images),
        "records_count": len(records),
        "targets_per_image": {str(key): value for key, value in sorted(target_counts.items())},
        "accepted_total": len(accepted_ids),
        "accepted_by_target_count": {
            str(key): value for key, value in sorted(accepted_by_targets.items())
        },
        "eligible_rule": "accepted AND exactly one RGB target",
        "eligible_instances": accepted_by_targets.get(1, 0),
        "ambiguous_multi_target_accepted": sum(
            count for targets, count in accepted_by_targets.items() if targets > 1
        ),
        "missing_records": len(missing_records),
        "missing_accepted_masks": len(missing_masks),
        "png_mode_after_load": "L/uint8",
        "png_unique_values": dict(sorted(unique_values.items())),
        "connected_components": summarize(component_counts),
        "mask_area_over_gt_box_area": summarize(mask_box_area_ratios),
        "mask_pixels_inside_gt_box_ratio": summarize(inside_box_ratios),
        "recorded_mask_bbox_iou": summarize(mask_bbox_ious),
        "visualized_image_ids": chosen if accepted_ids else [],
    }
    (args.output_dir / "sam_cache_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
