"""Fixed-EMA M-write intervention linked to SAM/no-SAM region and detection changes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig
from tools.diagnose_s_sgc1_per_image import image_stats

OUT = ROOT / "reports/M_R2_SAM_FACTORIAL_20E_TESTDEV/m_mechanism"
SOURCE = ROOT / "reports/M_R2_SAM_FACTORIAL_20E_TESTDEV"
REGION = SOURCE / "region_audit"
ANNOTATION = ROOT / "data/antiuav6k_common/annotations/instances_visible_common_test.json"
ARMS = {
    "with_sam": (REPO / "experiments/phase_m/m_layout_recovery_r2_main.yml",
                 ROOT / "outputs/M_LAYOUT_RECOVERY_20E_TESTDEV/R2_OTE2_LAYOUT/seed0/best_stg1.pth"),
    "without_sam": (REPO / "experiments/phase_m/m_r2_no_s_main.yml",
                    ROOT / "outputs/M_R2_SAM_FACTORIAL_20E_TESTDEV/no_s_seed0/best_stg1.pth"),
}


def load_npz(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def predict(arm):
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{arm}_m_off_predictions.npz"
    if destination.exists():
        raise FileExistsError(destination)
    config, checkpoint = ARMS[arm]
    cfg = YAMLConfig(str(config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model.cuda().eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["ema"]["module"], strict=True)
    loader = cfg.val_dataloader
    if len(loader.dataset) != 1820 or loader.dataset.sam_mask_root is not None:
        raise RuntimeError("Expected original testdev without inference-time SAM")
    ids, boxes, scores, labels = [], [], [], []
    hook = model.mote_fusion.register_forward_hook(
        lambda module, inputs, result: (inputs[0], result[1]))
    torch.set_num_threads(4)
    try:
        with torch.inference_mode():
            for step, (samples, targets) in enumerate(loader):
                sizes = torch.stack([target["orig_size"] for target in targets]).cuda()
                results = cfg.postprocessor(model(samples.cuda()), sizes)
                for target, result in zip(targets, results):
                    ids.append(int(target["image_id"]))
                    boxes.append(result["boxes"].cpu().float().numpy())
                    scores.append(result["scores"].cpu().float().numpy())
                    labels.append(result["labels"].cpu().numpy())
                if step % 50 == 0:
                    print(arm, step, len(ids), flush=True)
    finally:
        hook.remove()
    if len(set(ids)) != 1820 or np.stack(boxes).shape != (1820, 300, 4):
        raise RuntimeError("Incomplete or malformed predictions")
    np.savez_compressed(destination, image_ids=np.array(ids, dtype=np.int64),
                        boxes=np.stack(boxes), scores=np.stack(scores), labels=np.stack(labels))


def coco_metrics(predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(str(ANNOTATION))
    detections = []
    for image_id, boxes, scores, labels in zip(predictions["image_ids"], predictions["boxes"],
                                                predictions["scores"], predictions["labels"]):
        for box, score, label in zip(boxes, scores, labels):
            detections.append({"image_id": int(image_id), "category_id": int(label),
                "score": float(score), "bbox": [float(box[0]), float(box[1]),
                float(box[2] - box[0]), float(box[3] - box[1])]})
    evaluator = COCOeval(gt, gt.loadRes(detections), "bbox")
    evaluator.params.imgIds = predictions["image_ids"].tolist()
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator.stats.tolist()


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def mean(values):
    return float(np.mean(values)) if values else None


def analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "mechanism_link_summary.json"
    if destination.exists():
        raise FileExistsError(destination)
    predictions = {}
    for arm in ARMS:
        predictions[arm] = {
            "on": load_npz(SOURCE / "instance_audit" / f"{arm}_predictions.npz"),
            "off": load_npz(OUT / f"{arm}_m_off_predictions.npz")}
    ids = predictions["with_sam"]["on"]["image_ids"]
    if len(ids) != 1820 or any(not np.array_equal(ids, data[mode]["image_ids"])
                               for data in predictions.values() for mode in ("on", "off")):
        raise RuntimeError("Prediction IDs/order do not match")
    expected = {
        "with_sam": json.loads((ROOT / "reports/M_LAYOUT_RECOVERY_20E_TESTDEV/r2_best_ema_m_ablation.json").read_text(encoding="utf-8")),
        "without_sam": json.loads((SOURCE / "r2_no_s_best_ema_m_ablation.json").read_text(encoding="utf-8")),
    }
    checks = {}
    for arm in ARMS:
        observed = coco_metrics(predictions[arm]["off"])
        reference = expected[arm]["m_bypassed_metrics"]
        error = max(abs(float(x) - float(y)) for x, y in zip(observed, reference))
        if error > 5e-4:
            raise RuntimeError(f"{arm} M-off COCO mismatch: {error}")
        checks[arm] = {"off_coco_max_abs_error": error, "off_metrics": observed,
                       "reference": reference, "tolerance": 5e-4}
    anns = {int(item["image_id"]): item for item in json.loads(ANNOTATION.read_text(
        encoding="utf-8"))["annotations"] if not item.get("iscrowd", 0)}
    regions = {arm: {row["image_id"]: row for row in json.loads((REGION / f"{arm}_regions.json").read_text(
        encoding="utf-8"))["rows"]} for arm in ARMS}
    if set(regions["with_sam"]) != set(regions["without_sam"]):
        raise RuntimeError("Region eligibility mismatch")
    groups = {"small": [], "medium": []}
    for i, image_id in enumerate(ids.tolist()):
        if image_id not in regions["with_sam"]:
            continue
        ann = anns[image_id]
        x, y, w, h = ann["bbox"]
        gt = np.array([x, y, x + w, y + h], dtype=np.float32)
        scale = "small" if w * h < 1024 else "medium"
        record = {"image_id": image_id, "scale": scale, "arms": {}}
        for arm in ARMS:
            region = regions[arm][image_id]
            on = image_stats(predictions[arm]["on"]["boxes"][i],
                             predictions[arm]["on"]["scores"][i], gt)
            off = image_stats(predictions[arm]["off"]["boxes"][i],
                              predictions[arm]["off"]["scores"][i], gt)
            record["arms"][arm] = {
                "m_iou_gain": on["best_iou_at_02"] - off["best_iou_at_02"],
                "m_score_gain_iou075": on["matched_score_0.75"] - off["matched_score_0.75"],
                "m_recovered_075": bool(on["hit02_0.75"] and not off["hit02_0.75"]),
                "m_lost_075": bool(off["hit02_0.75"] and not on["hit02_0.75"]),
                "m_hit_075_on": bool(on["hit02_0.75"]),
                "m_hit_075_off": bool(off["hit02_0.75"]),
            }
            for metric in ("cosine_distance", "normalized_distance"):
                record["arms"][arm][f"fusion_change_{metric}"] = (
                    region[f"post_s16_{metric}"] - region[f"pre_s16_{metric}"])
                record["arms"][arm][f"post_s16_{metric}"] = region[f"post_s16_{metric}"]
        groups[scale].append(record)
    result = {"schema": "r2_sam_m_mechanism_link_v1", "status": "PASS", "images": len(ids),
              "region_eligible": sum(map(len, groups.values())), "coco_checks": checks, "groups": {},
              "caveats": ["M-off is same-weight inference intervention, not independently trained no-M.",
                          "GT-box 9 inside/8 near-background points are a coarse region proxy.",
                          "Independent model feature spaces differ; single seed and reused testdev.",
                          "Score threshold 0.2 is diagnostic; correlations are exploratory."]}
    for scale, rows in groups.items():
        item = {"n": len(rows), "arms": {}, "sam_conditional": {}}
        for arm in ARMS:
            arm_rows = [row["arms"][arm] for row in rows]
            arm_result = {"m_recovered_075": sum(row["m_recovered_075"] for row in arm_rows),
                          "m_lost_075": sum(row["m_lost_075"] for row in arm_rows),
                          "m_hit_075_on": sum(row["m_hit_075_on"] for row in arm_rows),
                          "m_hit_075_off": sum(row["m_hit_075_off"] for row in arm_rows),
                          "m_iou_gain_mean": mean([row["m_iou_gain"] for row in arm_rows]),
                          "m_score_gain_iou075_mean": mean([row["m_score_gain_iou075"] for row in arm_rows])}
            for metric in ("cosine_distance", "normalized_distance"):
                key = f"fusion_change_{metric}"
                x = [row[key] for row in arm_rows]
                arm_result[key + "_mean"] = mean(x)
                arm_result[key + "_corr_m_iou_gain"] = correlation(
                    x, [row["m_iou_gain"] for row in arm_rows])
                arm_result[key + "_corr_m_score_gain_iou075"] = correlation(
                    x, [row["m_score_gain_iou075"] for row in arm_rows])
                for cohort, predicate in (("recovered", "m_recovered_075"), ("lost", "m_lost_075")):
                    selected = [row[key] for row in arm_rows if row[predicate]]
                    arm_result[key + f"_{cohort}_mean"] = mean(selected)
            item["arms"][arm] = arm_result
        for metric in ("cosine_distance", "normalized_distance"):
            key = f"fusion_change_{metric}"
            x = [row["arms"]["with_sam"][key] - row["arms"]["without_sam"][key]
                 for row in rows]
            y_iou = [row["arms"]["with_sam"]["m_iou_gain"]
                     - row["arms"]["without_sam"]["m_iou_gain"] for row in rows]
            y_score = [row["arms"]["with_sam"]["m_score_gain_iou075"]
                       - row["arms"]["without_sam"]["m_score_gain_iou075"] for row in rows]
            item["sam_conditional"][key] = {
                "mean": mean(x), "corr_conditional_m_iou_gain": correlation(x, y_iou),
                "corr_conditional_m_score_gain_iou075": correlation(x, y_score),
                "top_quintile_m_iou_gain_mean": mean([v for v, z in zip(y_iou, x)
                                                     if z >= np.quantile(x, .8)]),
                "bottom_quintile_m_iou_gain_mean": mean([v for v, z in zip(y_iou, x)
                                                        if z <= np.quantile(x, .2)]),
            }
        result["groups"][scale] = item
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "mechanism_link_rows.json").write_text(json.dumps(groups, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("predict", "analyze"))
    parser.add_argument("--arm", choices=tuple(ARMS))
    args = parser.parse_args()
    if args.mode == "predict":
        if args.arm is None:
            parser.error("predict requires --arm")
        predict(args.arm)
    else:
        analyze()
