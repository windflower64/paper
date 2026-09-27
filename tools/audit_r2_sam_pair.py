"""Paired instance-level predictions for trained R2 with and without SAM guidance."""

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

OUT = ROOT / "reports/M_R2_SAM_FACTORIAL_20E_TESTDEV/instance_audit"
RUNS = {
    "with_sam": ("experiments/phase_m/m_layout_recovery_r2_main.yml",
                 "outputs/M_LAYOUT_RECOVERY_20E_TESTDEV/R2_OTE2_LAYOUT/seed0/best_stg1.pth"),
    "without_sam": ("experiments/phase_m/m_r2_no_s_main.yml",
                    "outputs/M_R2_SAM_FACTORIAL_20E_TESTDEV/no_s_seed0/best_stg1.pth"),
}


def predict(arm):
    shared.OUT = OUT
    shared.RUNS = RUNS
    shared.predict(arm)


def analyze():
    target = OUT / "paired_instance_summary.json"
    if target.exists():
        raise FileExistsError(target)
    predictions = {}
    for arm in RUNS:
        with np.load(OUT / f"{arm}_predictions.npz") as archive:
            predictions[arm] = {key: archive[key] for key in archive.files}
    ids = predictions["with_sam"]["image_ids"]
    if len(ids) != 1820 or not np.array_equal(ids, predictions["without_sam"]["image_ids"]):
        raise RuntimeError("Image IDs do not match")
    annotation_path = ROOT / "data/antiuav6k_common/annotations/instances_visible_common_test.json"
    dataset = json.loads(annotation_path.read_text(encoding="utf-8"))
    anns = {int(item["image_id"]): item for item in dataset["annotations"] if not item.get("iscrowd", 0)}
    if len(anns) != len([item for item in dataset["annotations"] if not item.get("iscrowd", 0)]):
        raise RuntimeError("More than one target per image")
    grouped = {"small": [], "medium": [], "empty": []}
    for index, image_id in enumerate(ids.tolist()):
        if image_id not in anns:
            grouped["empty"].append({"image_id": image_id, "top_score": {
                arm: float(predictions[arm]["scores"][index].max()) for arm in RUNS}})
            continue
        ann = anns[image_id]
        x, y, width, height = ann["bbox"]
        gt = np.array([x, y, x + width, y + height], dtype=np.float32)
        scale = "small" if width * height < 1024 else "medium"
        stats = {arm: shared.image_stats(predictions[arm]["boxes"][index],
                                        predictions[arm]["scores"][index], gt)
                 for arm in RUNS}
        grouped[scale].append({"image_id": image_id, "gt": gt.tolist(), "models": stats})
    summary = {"schema": "r2_sam_pair_instance_audit_v1", "status": "PASS",
               "prediction_sources": {}, "annotation_source": str(annotation_path),
               "groups": {}, "paired": {}, "caveats": [
                   "Best EMA weights are independently trained and share best epoch 16 in this run.",
                   "Score threshold 0.2 and IoU thresholds 0.5/0.75/0.9 are diagnostic choices, not COCO AP.",
                   "Region or spatial-feature preservation is not measured by these prediction counts.",
                   "Single seed; original test is repeatedly used as development validation."]}
    for arm in RUNS:
        checkpoint = ROOT / RUNS[arm][1]
        summary["prediction_sources"][arm] = {
            "checkpoint": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "metadata": str(OUT / f"{arm}_metadata.json")}
    for scale in ("small", "medium"):
        rows = grouped[scale]
        item = {"images": len(rows), "thresholds": {}, "localization": {}}
        for threshold in (0.5, 0.75, 0.9):
            key = f"{threshold:.2f}"
            hit = f"hit02_{key}"
            item["thresholds"][key] = {
                "with_sam_hits": sum(row["models"]["with_sam"][hit] for row in rows),
                "without_sam_hits": sum(row["models"]["without_sam"][hit] for row in rows),
                "recovered_by_sam": sum(row["models"]["with_sam"][hit] and not row["models"]["without_sam"][hit] for row in rows),
                "lost_by_sam": sum(row["models"]["without_sam"][hit] and not row["models"]["with_sam"][hit] for row in rows),
            }
        changes = np.array([row["models"]["with_sam"]["best_iou_at_02"]
                            - row["models"]["without_sam"]["best_iou_at_02"] for row in rows])
        item["localization"] = {
            "best_iou_mean_delta": float(changes.mean()),
            "best_iou_median_delta": float(np.median(changes)),
            "improved_more_than_002": int((changes > .02).sum()),
            "degraded_more_than_002": int((changes < -.02).sum()),
        }
        summary["groups"][scale] = item
    empty = grouped["empty"]
    summary["groups"]["empty"] = {"images": len(empty), "false_positive_image_counts": {
        str(threshold): {arm: sum(row["top_score"][arm] >= threshold for row in empty)
                         for arm in RUNS} for threshold in (.2, .5)}}
    OUT.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "paired_instance_rows.json").write_text(json.dumps(grouped, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    output = OUT / "coco_reproduction.json"
    if output.exists():
        raise FileExistsError(output)
    annotation = ROOT / "data/antiuav6k_common/annotations/instances_visible_common_test.json"
    gt = COCO(str(annotation))
    result = {}
    for arm, (config, checkpoint) in RUNS.items():
        checkpoint = ROOT / checkpoint
        run = checkpoint.parent
        rows = [json.loads(line) for line in (run / "log.txt").read_text(encoding="utf-8").splitlines()]
        expected = max(rows, key=lambda row: row["test_coco_eval_bbox"][0])["test_coco_eval_bbox"]
        with np.load(OUT / f"{arm}_predictions.npz") as archive:
            ids, boxes, scores, labels = (archive[key] for key in ("image_ids", "boxes", "scores", "labels"))
        detections = []
        for image_id, image_boxes, image_scores, image_labels in zip(ids, boxes, scores, labels):
            for box, score, label in zip(image_boxes, image_scores, image_labels):
                detections.append({"image_id": int(image_id), "category_id": int(label),
                                   "score": float(score), "bbox": [float(box[0]), float(box[1]),
                                                                  float(box[2] - box[0]), float(box[3] - box[1])]})
        dt = gt.loadRes(detections)
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.imgIds = ids.tolist()
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        metrics = evaluator.stats.tolist()
        differences = [abs(float(a) - float(b)) for a, b in zip(metrics, expected)]
        error = max(differences)
        if error > 5e-4:
            raise RuntimeError(f"{arm} COCO reproduction mismatch: {differences}")
        result[arm] = {"status": "PASS", "max_abs_error": error,
                       "per_metric_abs_error": differences, "tolerance": 5e-4,
                       "metrics": metrics, "expected": expected}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({arm: item["max_abs_error"] for arm, item in result.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("predict", "analyze", "validate"))
    parser.add_argument("--arm", choices=tuple(RUNS))
    args = parser.parse_args()
    if args.mode == "predict":
        if args.arm is None:
            parser.error("predict requires --arm")
        predict(args.arm)
    elif args.mode == "analyze":
        analyze()
    else:
        validate()
