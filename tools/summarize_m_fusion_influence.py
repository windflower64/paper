"""Summarize paired M-strength predictions without changing model weights."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODES = ("off", "trained", "double")


def iou_with_one_gt(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    upper_left = np.maximum(boxes[:, :2], gt[:2])
    lower_right = np.minimum(boxes[:, 2:], gt[2:])
    extent = np.maximum(lower_right - upper_left, 0)
    intersection = extent[:, 0] * extent[:, 1]
    box_extent = np.maximum(boxes[:, 2:] - boxes[:, :2], 0)
    box_area = box_extent[:, 0] * box_extent[:, 1]
    gt_extent = np.maximum(gt[2:] - gt[:2], 0)
    gt_area = gt_extent[0] * gt_extent[1]
    return intersection / np.maximum(box_area + gt_area - intersection, 1e-12)


def image_metrics(boxes: np.ndarray, scores: np.ndarray, gt: np.ndarray | None) -> dict:
    result = {"top_score": float(scores.max())}
    if gt is None:
        return result
    overlaps = iou_with_one_gt(boxes, gt)
    for threshold in (0.5, 0.75):
        matched = overlaps >= threshold
        result[f"matched_score_{threshold}"] = float(scores[matched].max(initial=0))
        result[f"hit02_{threshold}"] = bool(np.any(matched & (scores >= 0.2)))
    high_score = scores >= 0.2
    result["best_iou_at_score02"] = float(overlaps[high_score].max(initial=0))
    return result


def group_summary(rows: list[dict], mode: str) -> dict:
    if not rows:
        return {"images": 0}
    result = {"images": len(rows)}
    for threshold in (0.5, 0.75):
        key = f"hit02_{threshold}"
        recovered = sum(not row["off"][key] and row[mode][key] for row in rows if row["rgb_positive"])
        lost = sum(row["off"][key] and not row[mode][key] for row in rows if row["rgb_positive"])
        result[f"hit_iou{threshold}_score02"] = {"recovered": recovered, "lost": lost}
        positive = [row for row in rows if row["rgb_positive"]]
        if positive:
            score_key = f"matched_score_{threshold}"
            result[f"matched_score_delta_iou{threshold}_mean"] = float(
                np.mean([row[mode][score_key] - row["off"][score_key] for row in positive])
            )
    empty = [row for row in rows if not row["rgb_positive"]]
    if empty:
        result["rgb_empty_top_score"] = {
            "off_mean": float(np.mean([row["off"]["top_score"] for row in empty])),
            "mode_mean": float(np.mean([row[mode]["top_score"] for row in empty])),
            "off_above_02": sum(row["off"]["top_score"] >= 0.2 for row in empty),
            "mode_above_02": sum(row[mode]["top_score"] >= 0.2 for row in empty),
            "off_above_05": sum(row["off"]["top_score"] >= 0.5 for row in empty),
            "mode_above_05": sum(row[mode]["top_score"] >= 0.5 for row in empty),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("A5", "E1", "E2"), required=True)
    args = parser.parse_args()
    folder = args.input_dir.resolve()
    destination = folder / f"{args.arm}_per_image_summary.json"
    if destination.exists():
        raise FileExistsError(destination)
    annotations_path = ROOT / "data/antiuav6k_common/annotations/instances_visible_common_test.json"
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    gt_by_image: dict[int, list[dict]] = {int(image["id"]): [] for image in annotations["images"]}
    for annotation in annotations["annotations"]:
        if not annotation.get("iscrowd", 0):
            gt_by_image[int(annotation["image_id"])].append(annotation)
    predictions = {}
    for mode in MODES:
        with np.load(folder / f"{args.arm}_{mode}_predictions.npz") as archive:
            predictions[mode] = {key: archive[key] for key in archive.files}
    reference = predictions["off"]
    ids = reference["image_ids"]
    for mode in MODES[1:]:
        if not np.array_equal(ids, predictions[mode]["image_ids"]):
            raise RuntimeError("Mode image order differs")
    rows = []
    for index, image_id in enumerate(ids.tolist()):
        annotations_for_image = gt_by_image[image_id]
        if len(annotations_for_image) > 1:
            raise RuntimeError("This audit assumes at most one RGB GT per image")
        rgb_positive = bool(annotations_for_image)
        ir_positive = bool(reference["ir_counts"][index])
        if rgb_positive:
            x, y, width, height = annotations_for_image[0]["bbox"]
            gt = np.asarray([x, y, x + width, y + height], dtype=np.float32)
            area = width * height
            size = "small" if area < 32 * 32 else "medium" if area < 96 * 96 else "large"
        else:
            gt, size = None, "empty"
        row = {
            "image_id": image_id,
            "rgb_positive": rgb_positive,
            "ir_positive": ir_positive,
            "state": "both" if rgb_positive and ir_positive else
                     "rgb_only" if rgb_positive else
                     "ir_only" if ir_positive else "neither",
            "size": size,
        }
        for mode in MODES:
            pred = predictions[mode]
            row[mode] = image_metrics(pred["boxes"][index], pred["scores"][index], gt)
        if rgb_positive:
            score = row["off"]["matched_score_0.5"]
            row["rgb_difficulty"] = "hard" if score < 0.2 else "easy" if score >= 0.5 else "intermediate"
        rows.append(row)

    summary = {
        "schema": "m_fusion_per_image_v1",
        "arm": args.arm,
        "images": len(rows),
        "note": "Fixed score thresholds and per-image hit counts are diagnostics, not COCO AP.",
        "states": {},
        "modes": {},
    }
    for state in ("both", "rgb_only", "ir_only", "neither"):
        summary["states"][state] = sum(row["state"] == state for row in rows)
    groups = {
        "all": rows,
        "both": [row for row in rows if row["state"] == "both"],
        "rgb_only": [row for row in rows if row["state"] == "rgb_only"],
        "ir_only": [row for row in rows if row["state"] == "ir_only"],
        "neither": [row for row in rows if row["state"] == "neither"],
        "rgb_hard": [row for row in rows if row.get("rgb_difficulty") == "hard"],
        "rgb_easy": [row for row in rows if row.get("rgb_difficulty") == "easy"],
        "small": [row for row in rows if row["size"] == "small"],
        "medium": [row for row in rows if row["size"] == "medium"],
    }
    for mode in MODES[1:]:
        summary["modes"][mode] = {name: group_summary(group, mode) for name, group in groups.items()}
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    detail = folder / f"{args.arm}_per_image_rows.csv"
    with detail.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image_id", "state", "size", "rgb_difficulty", "mode", "top_score", "matched_score_iou50", "matched_score_iou75", "hit02_iou50", "hit02_iou75"))
        for row in rows:
            for mode in MODES:
                item = row[mode]
                writer.writerow((row["image_id"], row["state"], row["size"], row.get("rgb_difficulty", ""), mode, item["top_score"], item.get("matched_score_0.5", ""), item.get("matched_score_0.75", ""), item.get("hit02_0.5", ""), item.get("hit02_0.75", "")))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
