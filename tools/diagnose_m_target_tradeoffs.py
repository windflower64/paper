"""Same-checkpoint M on/off analysis, without assuming query-slot identity."""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from torchvision.ops import box_iou, box_convert

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from src.core import YAMLConfig
from src.solver.rgb_preservation import checkpoint_weights
from src.data import CocoEvaluator

OUT = ROOT / "reports/97_rgb_preservation/m_target_tradeoffs"
CHECKPOINT = ROOT / "outputs/C_PLUS_M_SD22_B8A4_20E_TESTDEV/seed0/best_stg1.pth"


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def target_metrics(prediction, gt):
    scores, order = prediction["scores"].sort(descending=True)
    boxes = prediction["boxes"][order]
    iou = box_iou(boxes, gt.reshape(1, 4)).flatten()
    result = {"top1_iou": float(iou[0]), "top1_score": float(scores[0]),
              "best_iou_300": float(iou.max()), "best_iou_100": float(iou[:100].max()),
              "best_iou_10": float(iou[:10].max())}
    for threshold in (.5, .75):
        tag = str(int(threshold * 100))
        valid = torch.where(iou >= threshold)[0]
        result[f"correct_score_{tag}"] = float(scores[valid[0]]) if len(valid) else None
        result[f"correct_rank_{tag}"] = int(valid[0]) + 1 if len(valid) else None
    return result


def false_predictions(prediction, gt, score_threshold):
    order = prediction["scores"].argsort(descending=True)[:100]
    order = order[prediction["scores"][order] >= score_threshold]
    if not len(gt):
        return {"predictions": len(order), "background": len(order), "localization": 0, "duplicate": 0}
    ious = box_iou(prediction["boxes"][order], gt)
    seen = set()
    counts = {"predictions": len(order), "background": 0, "localization": 0, "duplicate": 0}
    for row in ious:
        best = float(row.max())
        if best < .1:
            counts["background"] += 1
        elif best < .5:
            counts["localization"] += 1
        else:
            eligible = [i for i in row.argsort(descending=True).tolist() if float(row[i]) >= .5 and i not in seen]
            if eligible:
                seen.add(eligible[0])
            else:
                counts["duplicate"] += 1
    return counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = YAMLConfig(str(REPO / "experiments/phase_m/c_plus_m_sd22_b8a4_20e_testdev_local.yml"))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model.cuda().eval()
    model.load_state_dict(checkpoint_weights(CHECKPOINT), strict=True)
    loader = cfg.val_dataloader
    coco = loader.dataset.coco
    evaluators = {mode: CocoEvaluator(coco, ["bbox"]) for mode in ("on", "off")}
    conditioner = model.sd2_conditioner
    cache = {}
    rows, image_rows = [], []
    with torch.inference_mode():
        for batch, (samples, targets) in enumerate(loader):
            samples = samples.cuda()
            sizes = torch.stack([t["orig_size"] for t in targets]).cuda()
            predictions = {}
            for mode in ("on", "off"):
                model.sd2_conditioner = conditioner if mode == "on" else None
                outputs = model(samples)
                processed = cfg.postprocessor(outputs, sizes)
                predictions[mode] = [{k: v.cpu() for k, v in p.items()} for p in processed]
                evaluators[mode].update({int(t["image_id"]): p for t, p in zip(targets, predictions[mode])})
            model.sd2_conditioner = conditioner
            for index, target in enumerate(targets):
                image_id = int(target["image_id"])
                annotations = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])) if not a.get("iscrowd", 0)]
                gt = torch.tensor([a["bbox"] for a in annotations], dtype=torch.float32).reshape(-1, 4)
                gt = box_convert(gt, "xywh", "xyxy")
                cache[image_id] = {mode: predictions[mode][index] for mode in predictions}
                image_record = {"image_id": image_id, "target_count": len(annotations)}
                for mode in predictions:
                    p = predictions[mode][index]
                    image_record[mode] = {"max_score": float(p["scores"].max()),
                        "fp": {str(t): false_predictions(p, gt, t) for t in (.05, .25, .5)}}
                image_rows.append(image_record)
                for j, annotation in enumerate(annotations):
                    area = annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3])
                    row = {"image_id": image_id, "annotation_id": annotation["id"], "area": area,
                           "scale": "small" if area < 1024 else "medium" if area < 9216 else "large",
                           "single_target_image": len(annotations) == 1}
                    for mode in predictions:
                        row[mode] = target_metrics(predictions[mode][index], gt[j])
                    rows.append(row)
            if batch % 25 == 0:
                print(f"processed {len(image_rows)}/{len(loader.dataset)} images", flush=True)
    metrics = {}
    for mode, evaluator in evaluators.items():
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        metrics[mode] = evaluator.coco_eval["bbox"].stats.tolist()
    summaries = {}
    for scale in ("all", "small", "medium", "large"):
        selected = [r for r in rows if scale == "all" or r["scale"] == scale]
        if not selected:
            continue
        summary = {"targets": len(selected)}
        for mode in ("on", "off"):
            summary[mode] = {key: mean([r[mode][key] for r in selected]) for key in
                             ("best_iou_300", "best_iou_100", "best_iou_10")}
            single = [r for r in selected if r["single_target_image"]]
            summary[mode]["single_target_top1_iou"] = mean([r[mode]["top1_iou"] for r in single])
            for tag in ("50", "75"):
                summary[mode][f"gt_covered_iou{tag}_top300"] = sum(r[mode][f"correct_rank_{tag}"] is not None for r in selected)
                summary[mode][f"gt_covered_iou{tag}_top100"] = sum(r[mode][f"correct_rank_{tag}"] is not None and r[mode][f"correct_rank_{tag}"] <= 100 for r in selected)
        summary["changes"] = {key: {"up_005": sum(r["on"][key] - r["off"][key] >= .05 for r in selected),
                                   "down_005": sum(r["on"][key] - r["off"][key] <= -.05 for r in selected)}
                              for key in ("best_iou_300", "best_iou_100", "best_iou_10")}
        for tag in ("50", "75"):
            both = [r for r in selected if all(r[m][f"correct_rank_{tag}"] is not None for m in ("on", "off"))]
            summary[f"both_cover_{tag}"] = {"count": len(both), "on_minus_off_correct_score": mean([
                r["on"][f"correct_score_{tag}"]-r["off"][f"correct_score_{tag}"] for r in both]),
                "on_minus_off_correct_rank": mean([r["on"][f"correct_rank_{tag}"]-r["off"][f"correct_rank_{tag}"] for r in both])}
        summaries[scale] = summary
    false_summaries = {}
    for group in ("all", "empty", "single_small", "single_medium"):
        ids = {r["image_id"] for r in rows if r["single_target_image"] and r["scale"] == group.removeprefix("single_")}
        selected = [r for r in image_rows if group == "all" or (group == "empty" and r["target_count"] == 0) or r["image_id"] in ids]
        false_summaries[group] = {"images": len(selected)}
        for mode in ("on", "off"):
            false_summaries[group][mode] = {str(t): {key: sum(r[mode]["fp"][str(t)][key] for r in selected)
                for key in ("predictions", "background", "localization", "duplicate")} for t in (.05, .25, .5)}
    summary = {"checkpoint": str(CHECKPOINT), "sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
               "images": len(image_rows), "targets": len(rows), "multi_target_images": sum(r["target_count"] > 1 for r in image_rows),
               "coco_metrics": metrics, "by_scale": summaries, "false_predictions": false_summaries,
               "boundary": "Fixed-weight diagnostic. No query-slot pairing, no GT reranking, no new model. Threshold FP counts are not COCO AP decomposition."}
    for mode in ("on", "off"):
        old = json.loads((ROOT / f"reports/80_cdm_joint/C_PLUS_M_SD22/best_epoch13_{'enabled' if mode == 'on' else 'disabled'}.json").read_text(encoding="utf-8-sig"))
        # Historical schema is retained verbatim for independent comparison.
        summary[f"historical_{mode}"] = old
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "targets.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    (OUT / "images.json").write_text(json.dumps(image_rows, ensure_ascii=False), encoding="utf-8")
    torch.save(cache, OUT / "predictions.pt")
    print("ANALYSIS_COMPLETE", OUT, flush=True)


if __name__ == "__main__":
    main()
