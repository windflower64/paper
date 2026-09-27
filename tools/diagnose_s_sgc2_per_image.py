"""Paired per-image audit for SGC2 against C and constant-SAM SGC1."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
OLD = ROOT / "reports/107_sgc1_per_image_diagnosis"
OUT = ROOT / "reports/109_sgc2_teacher_retirement"
sys.path.insert(0, str(REPO))

from tools import diagnose_s_sgc1_per_image as shared


def predict(arm):
    assert arm in ("sam", "box")
    alias = "sgc2" if arm == "sam" else "sgc2_box"
    shared.OUT = OUT
    shared.RUNS = {
        alias: (
            f"experiments/phase_s/s_sgc2_{arm}_decay9_14_b8a4_20e.yml",
            f"outputs/S_SGC2_{arm.upper()}_DECAY9_14_B8A4_20E_TESTDEV/seed0/best_stg1.pth",
        )
    }
    shared.predict(alias)


def load(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def analyze():
    predictions = {
        "c": load(OLD / "c_predictions.npz"),
        "sgc1_sam": load(OLD / "sam_predictions.npz"),
        "sgc2": load(OUT / "sgc2_predictions.npz"),
        "sgc2_box": load(OUT / "sgc2_box_predictions.npz"),
    }
    ids = predictions["c"]["image_ids"]
    assert all(np.array_equal(ids, item["image_ids"]) for item in predictions.values())
    annotation_path = ROOT / "data/antiuav6k_common/annotations/instances_visible_common_test.json"
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in coco["images"]}
    annotations = {image_id: [] for image_id in images}
    for annotation in coco["annotations"]:
        if not annotation.get("iscrowd", 0):
            annotations[annotation["image_id"]].append(annotation)

    rows = []
    for index, image_id in enumerate(ids.tolist()):
        assert len(annotations[image_id]) <= 1
        annotation = annotations[image_id]
        area = 0 if not annotation else annotation[0]["bbox"][2] * annotation[0]["bbox"][3]
        group = "empty" if not annotation else "small" if area < 32**2 else "medium"
        row = {
            "image_id": image_id,
            "file_name": images[image_id]["file_name"],
            "group": group,
        }
        if annotation:
            x, y, width, height = annotation[0]["bbox"]
            gt = [x, y, x + width, y + height]
            row["models"] = {
                arm: shared.image_stats(
                    item["boxes"][index], item["scores"][index], gt
                )
                for arm, item in predictions.items()
            }
        else:
            row["models"] = {
                arm: {"top_score": float(item["scores"][index].max())}
                for arm, item in predictions.items()
            }
        rows.append(row)

    summary = {"images": len(rows), "groups": {}, "paired": {}}
    for group in ("small", "medium", "empty"):
        subset = [row for row in rows if row["group"] == group]
        summary["groups"][group] = {"images": len(subset)}
        if group == "empty":
            for arm in predictions:
                scores = [row["models"][arm]["top_score"] for row in subset]
                summary["groups"][group][arm] = {
                    "fp02": sum(score >= 0.2 for score in scores),
                    "fp05": sum(score >= 0.5 for score in scores),
                    "top_score_mean": float(np.mean(scores)),
                    "top_score_p95": float(np.quantile(scores, 0.95)),
                }

    for baseline in ("c", "sgc1_sam", "sgc2_box"):
        comparison = {}
        for group in ("small", "medium"):
            subset = [row for row in rows if row["group"] == group]
            item = {}
            for threshold in (0.5, 0.75, 0.9):
                key = f"{threshold:.2f}"
                hit_key = f"hit02_{key}"
                both = [
                    row
                    for row in subset
                    if row["models"][baseline][hit_key]
                    and row["models"]["sgc2"][hit_key]
                ]
                item[key] = {
                    "recovered": sum(
                        not row["models"][baseline][hit_key]
                        and row["models"]["sgc2"][hit_key]
                        for row in subset
                    ),
                    "lost": sum(
                        row["models"][baseline][hit_key]
                        and not row["models"]["sgc2"][hit_key]
                        for row in subset
                    ),
                    "unchanged_hit": len(both),
                    "both_hit_score_delta_mean": float(
                        np.mean(
                            [
                                row["models"]["sgc2"][f"matched_score_{key}"]
                                - row["models"][baseline][f"matched_score_{key}"]
                                for row in both
                            ]
                        )
                    )
                    if both
                    else None,
                }
            deltas = [
                row["models"]["sgc2"]["best_iou_at_02"]
                - row["models"][baseline]["best_iou_at_02"]
                for row in subset
            ]
            item["localization"] = {
                "improved_gt_002": sum(delta > 0.02 for delta in deltas),
                "degraded_lt_minus002": sum(delta < -0.02 for delta in deltas),
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
            }
            comparison[group] = item
        summary["paired"][f"sgc2_vs_{baseline}"] = comparison

    (OUT / "per_image_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("predict", "analyze"))
    parser.add_argument("--arm", choices=("sam", "box"), default="sam")
    arguments = parser.parse_args()
    predict(arguments.arm) if arguments.mode == "predict" else analyze()
