"""Paired prediction audit: R2 SGC2-only versus R2 SGC2+SQER2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO))
from tools import diagnose_s_sgc1_per_image as shared

OUT = ROOT / "reports/M_R2_SAM_FACTORIAL_20E_TESTDEV/sqer_contribution"
EXISTING = ROOT / "reports/M_R2_SAM_FACTORIAL_20E_TESTDEV/instance_audit/with_sam_predictions.npz"
ANNOTATION = ROOT / "data/antiuav6k_common/annotations/instances_visible_common_test.json"
RUN = ROOT / "outputs/M_R2_SAM_FACTORIAL_20E_TESTDEV/sgc_only_seed0"
RUNS = {
    "sgc_only": ("experiments/phase_m/m_r2_sgc_only_main.yml",
                 "outputs/M_R2_SAM_FACTORIAL_20E_TESTDEV/sgc_only_seed0/best_stg1.pth")
}


def predict():
    shared.OUT = OUT
    shared.RUNS = RUNS
    shared.predict("sgc_only")


def load_arrays(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def box_errors(box, gt):
    gw = max(float(gt[2] - gt[0]), 1e-6)
    gh = max(float(gt[3] - gt[1]), 1e-6)
    bw = max(float(box[2] - box[0]), 1e-6)
    bh = max(float(box[3] - box[1]), 1e-6)
    center = ((float(box[0] + box[2] - gt[0] - gt[2]) / (2 * gw)) ** 2
              + (float(box[1] + box[3] - gt[1] - gt[3]) / (2 * gh)) ** 2) ** .5
    size = (abs(float(np.log(bw / gw))) + abs(float(np.log(bh / gh)))) / 2
    return center, size


def row_stats(boxes, scores, gt):
    overlap = shared.ious(boxes, gt)
    visible = scores >= .2
    top = int(scores.argmax())
    best = int(np.argmax(np.where(visible, overlap, -1))) if visible.any() else None
    matched = np.flatnonzero(visible & (overlap >= .5))
    ranked_match = int(matched[np.argmax(scores[matched])]) if len(matched) else None
    selected = {"top": top, "oracle_at_02": best, "matched_score_at_02": ranked_match}
    result = {"top_score": float(scores[top]), "max_iou_at_02": float(overlap[visible].max(initial=0)),
              "max_score_iou_05": float(scores[overlap >= .5].max(initial=0)),
              "max_score_iou_075": float(scores[overlap >= .75].max(initial=0)),
              "max_score_nonmatch_iou05": float(scores[overlap < .5].max(initial=0)),
              "top_is_iou05": bool(overlap[top] >= .5),
              "hit_05": bool(np.any(visible & (overlap >= .5))),
              "hit_075": bool(np.any(visible & (overlap >= .75))),
              "hit_09": bool(np.any(visible & (overlap >= .9)))}
    for name, index in selected.items():
        result[name] = None if index is None else {
            "iou": float(overlap[index]), "score": float(scores[index]),
            "center_error_gt_norm": box_errors(boxes[index], gt)[0],
            "size_log_error": box_errors(boxes[index], gt)[1],
        }
    return result


def mean(values):
    return float(np.mean(values)) if values else None


def summarize_group(rows):
    out = {"images": len(rows), "arms": {}, "paired": {}}
    for arm in ("sgc_only", "full"):
        values = [row[arm] for row in rows]
        out["arms"][arm] = {
            "hit_05": sum(item["hit_05"] for item in values),
            "hit_075": sum(item["hit_075"] for item in values),
            "hit_09": sum(item["hit_09"] for item in values),
            "top1_iou_mean": mean([item["top"]["iou"] for item in values]),
            "oracle_iou_mean": mean([item["max_iou_at_02"] for item in values]),
            "matched_score_iou05_mean": mean([item["max_score_iou_05"] for item in values]),
            "matched_score_iou075_mean": mean([item["max_score_iou_075"] for item in values]),
            "top_is_iou05": sum(item["top_is_iou05"] for item in values),
        }
    for key in ("hit_05", "hit_075", "hit_09"):
        out["paired"][key] = {
            "recovered_by_full": sum(row["full"][key] and not row["sgc_only"][key] for row in rows),
            "lost_by_full": sum(row["sgc_only"][key] and not row["full"][key] for row in rows),
            "both": sum(row["sgc_only"][key] and row["full"][key] for row in rows),
        }
    for name in ("top", "oracle_at_02", "matched_score_at_02"):
        common = [row for row in rows if row["sgc_only"][name] is not None
                  and row["full"][name] is not None]
        out["paired"][name] = {"common_images": len(common)}
        for metric in ("iou", "score", "center_error_gt_norm", "size_log_error"):
            deltas = [row["full"][name][metric] - row["sgc_only"][name][metric]
                      for row in common]
            out["paired"][name][f"{metric}_delta_mean"] = mean(deltas)
            out["paired"][name][f"{metric}_delta_median"] = float(np.median(deltas)) if deltas else None
            if metric == "iou":
                out["paired"][name]["iou_improved_gt_002"] = sum(x > .02 for x in deltas)
                out["paired"][name]["iou_degraded_lt_minus002"] = sum(x < -.02 for x in deltas)
    margin_rows = [row for row in rows if row["sgc_only"]["hit_05"] and row["full"]["hit_05"]]
    out["paired"]["within_image_ranking"] = {"common_detectable_images": len(margin_rows)}
    for arm in ("sgc_only", "full"):
        margins = [row[arm]["max_score_iou_05"] - row[arm]["max_score_nonmatch_iou05"]
                   for row in margin_rows]
        out["paired"]["within_image_ranking"][arm] = {
            "positive_outscores_negative": sum(x > 0 for x in margins),
            "margin_mean": mean(margins), "margin_median": float(np.median(margins)) if margins else None}
    changes = [(row["full"]["max_score_iou_05"] - row["full"]["max_score_nonmatch_iou05"])
               - (row["sgc_only"]["max_score_iou_05"] - row["sgc_only"]["max_score_nonmatch_iou05"])
               for row in margin_rows]
    out["paired"]["within_image_ranking"]["full_minus_sgc_margin_mean"] = mean(changes)
    return out


def analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "paired_prediction_summary_v2.json"
    if destination.exists():
        raise FileExistsError(destination)
    arrays = {"sgc_only": load_arrays(OUT / "sgc_only_predictions.npz"),
              "full": load_arrays(EXISTING)}
    ids = arrays["sgc_only"]["image_ids"]
    if len(ids) != 1820 or not np.array_equal(ids, arrays["full"]["image_ids"]):
        raise RuntimeError("Prediction images do not match in order")
    data = json.loads(ANNOTATION.read_text(encoding="utf-8"))
    anns = {int(a["image_id"]): a for a in data["annotations"] if not a.get("iscrowd", 0)}
    grouped = {"small": [], "medium": [], "empty": []}
    for i, image_id in enumerate(ids.tolist()):
        if image_id not in anns:
            grouped["empty"].append({"image_id": image_id,
                "sgc_only": float(arrays["sgc_only"]["scores"][i].max()),
                "full": float(arrays["full"]["scores"][i].max())})
            continue
        x, y, width, height = anns[image_id]["bbox"]
        gt = np.array([x, y, x + width, y + height], dtype=np.float32)
        scale = "small" if width * height < 1024 else "medium"
        row = {"image_id": image_id, "gt": gt.tolist()}
        for arm in arrays:
            row[arm] = row_stats(arrays[arm]["boxes"][i], arrays[arm]["scores"][i], gt)
        grouped[scale].append(row)
    result = {"schema": "r2_sqer_contribution_prediction_audit_v1", "status": "PASS",
        "arms": {arm: {"checkpoint": str(ROOT / RUNS["sgc_only"][1]) if arm == "sgc_only"
                      else str(ROOT / "outputs/M_LAYOUT_RECOVERY_20E_TESTDEV/R2_OTE2_LAYOUT/seed0/best_stg1.pth")}
                 for arm in arrays},
        "image_count": len(ids), "groups": {scale: summarize_group(grouped[scale])
                                          for scale in ("small", "medium")},
        "empty": {"images": len(grouped["empty"]), "top_score_ge_02": {}, "top_score_ge_05": {}},
        "caveats": ["Two independently trained best EMAs (epoch 11 versus 16); score scales may differ.",
                    "GT-conditioned oracle and matched boxes are diagnostics, not actual detector assignments.",
                    "Score threshold 0.2 is diagnostic, not COCO AP.",
                    "The original test split has been used repeatedly for development; single seed."]}
    for arm in arrays:
        result["empty"]["top_score_ge_02"][arm] = sum(r[arm] >= .2 for r in grouped["empty"])
        result["empty"]["top_score_ge_05"][arm] = sum(r[arm] >= .5 for r in grouped["empty"])
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "paired_prediction_rows_v2.json").write_text(json.dumps(grouped, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def validate():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    destination = OUT / "sgc_only_coco_reproduction.json"
    if destination.exists():
        raise FileExistsError(destination)
    rows = [json.loads(line) for line in (RUN / "log.txt").read_text(encoding="utf-8").splitlines()]
    expected = max(rows, key=lambda row: row["test_coco_eval_bbox"][0])["test_coco_eval_bbox"]
    data = load_arrays(OUT / "sgc_only_predictions.npz")
    detections = []
    for image_id, boxes, scores, labels in zip(data["image_ids"], data["boxes"],
                                                data["scores"], data["labels"]):
        for box, score, label in zip(boxes, scores, labels):
            detections.append({"image_id": int(image_id), "category_id": int(label),
                "score": float(score), "bbox": [float(box[0]), float(box[1]),
                float(box[2] - box[0]), float(box[3] - box[1])]})
    gt = COCO(str(ANNOTATION))
    dt = gt.loadRes(detections)
    evaluator = COCOeval(gt, dt, "bbox")
    evaluator.params.imgIds = data["image_ids"].tolist()
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = evaluator.stats.tolist()
    error = max(abs(float(a) - float(b)) for a, b in zip(metrics, expected))
    if error > 5e-4:
        raise RuntimeError(f"COCO reconstruction error {error}")
    report = {"status": "PASS", "max_abs_error": error, "tolerance": 5e-4,
              "metrics": metrics, "expected": expected,
              "checkpoint_sha256": hashlib.sha256((RUN / "best_stg1.pth").read_bytes()).hexdigest()}
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("predict", "analyze", "validate"))
    args = parser.parse_args()
    {"predict": predict, "analyze": analyze, "validate": validate}[args.mode]()
