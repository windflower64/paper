"""Frozen A5 image-level router pilot; threshold selected only on train data.

This is an exploratory upper-level routing probe, not a new detector or a
publication-ready result. The router sees only the M-off RGB top score.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from faster_coco_eval import COCO, COCOeval_faster


ROOT = Path(__file__).resolve().parents[2]
GRID = (0.5, 0.6, 0.7, 0.8)


def load_pair(folder: Path) -> tuple[dict, dict]:
    results = []
    for mode in ("off", "trained"):
        with np.load(folder / f"A5_{mode}_predictions.npz") as archive:
            results.append({key: archive[key] for key in archive.files})
    off, trained = results
    if not np.array_equal(off["image_ids"], trained["image_ids"]):
        raise RuntimeError("Image order mismatch")
    if len(np.unique(off["image_ids"])) != len(off["image_ids"]):
        raise RuntimeError("Duplicate image IDs")
    return off, trained


def coco_ap(coco_gt: COCO, off: dict, trained: dict, threshold: float | str) -> dict:
    if threshold == "off":
        choose_trained = np.zeros(len(off["image_ids"]), dtype=bool)
    elif threshold == "trained":
        choose_trained = np.ones(len(off["image_ids"]), dtype=bool)
    else:
        choose_trained = off["scores"].max(axis=1) < threshold
    detections = []
    for index, image_id in enumerate(off["image_ids"].tolist()):
        source = trained if choose_trained[index] else off
        scores = source["scores"][index]
        selected = np.argsort(scores)[-100:][::-1]
        for item in selected:
            x1, y1, x2, y2 = source["boxes"][index, item].tolist()
            detections.append({
                "image_id": int(image_id),
                "category_id": int(source["labels"][index, item]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(scores[item]),
            })
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(detections)
        evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox", print_function=lambda *_: None)
        evaluator.params.imgIds = off["image_ids"].tolist()
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "threshold": threshold,
        "images_using_M": int(choose_trained.sum()),
        "images": len(choose_trained),
        "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "APS": float(evaluator.stats[3]),
        "APM": float(evaluator.stats[4]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    train_off, train_on = load_pair(args.train_dir.resolve())
    test_off, test_on = load_pair(args.test_dir.resolve())
    if len(train_off["image_ids"]) != 3200 or len(test_off["image_ids"]) != 1820:
        raise RuntimeError("Unexpected train/testdev image counts")
    with contextlib.redirect_stdout(io.StringIO()):
        train_gt = COCO(str(ROOT / "data/antiuav6k_common/annotations/instances_visible_common_train.json"))
        test_gt = COCO(str(ROOT / "data/antiuav6k_common/annotations/instances_visible_common_test.json"))
    train = [coco_ap(train_gt, train_off, train_on, item) for item in ("off", "trained", *GRID)]
    candidates = [item for item in train if isinstance(item["threshold"], float)]
    selected = max(candidates, key=lambda item: item["AP"])["threshold"]
    test = [coco_ap(test_gt, test_off, test_on, item) for item in ("off", "trained", selected)]
    result = {
        "schema": "m_confidence_router_pilot_v1",
        "selection_rule": "Use M when frozen M-off RGB top1 score is below threshold",
        "threshold_grid": GRID,
        "selected_threshold_from_train_only": selected,
        "train": train,
        "testdev": test,
        "caveat": "A5 detector was trained on the train split and selected using testdev; this is exploratory, not independent evidence. Image-level route may miss per-target opportunities.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
