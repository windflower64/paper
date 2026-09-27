"""Measure paired target-vs-near-background feature contrast around M-OTE2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig

OUT = ROOT / "reports/M_R2_SAM_FACTORIAL_20E_TESTDEV/region_audit"
ARMS = {
    "with_sam": (REPO / "experiments/phase_m/m_layout_recovery_r2_main.yml",
                 ROOT / "outputs/M_LAYOUT_RECOVERY_20E_TESTDEV/R2_OTE2_LAYOUT/seed0/best_stg1.pth"),
    "without_sam": (REPO / "experiments/phase_m/m_r2_no_s_main.yml",
                    ROOT / "outputs/M_R2_SAM_FACTORIAL_20E_TESTDEV/no_s_seed0/best_stg1.pth"),
}
INTERIOR = [(x, y) for y in (-.25, 0., .25) for x in (-.25, 0., .25)]
RING = [(-.75, -.75), (0., -.75), (.75, -.75),
        (-.75, 0.), (.75, 0.),
        (-.75, .75), (0., .75), (.75, .75)]


def grid_for_batch(targets, annotations, images, device):
    grids, valid = [], []
    for target in targets:
        image_id = int(target["image_id"])
        ann = annotations.get(image_id)
        if ann is None:
            grids.append([[0., 0.]] * 17)
            valid.append(False)
            continue
        x, y, width, height = map(float, ann["bbox"])
        size = images[image_id]
        cx, cy = x + width / 2, y + height / 2
        inside = (cx - .75 * width >= 0 and cx + .75 * width <= size["width"]
                  and cy - .75 * height >= 0 and cy + .75 * height <= size["height"])
        valid.append(inside)
        points = [[2 * (cx + dx * width) / size["width"] - 1,
                   2 * (cy + dy * height) / size["height"] - 1]
                  for dx, dy in INTERIOR + RING]
        grids.append(points)
    return torch.tensor(grids, dtype=torch.float32, device=device).unsqueeze(2), valid


def contrast(feature, grid):
    sampled = F.grid_sample(feature.float(), grid, mode="bilinear",
                            padding_mode="border", align_corners=False)
    sampled = sampled.squeeze(-1)
    foreground = sampled[:, :, :9].mean(-1)
    near_background = sampled[:, :, 9:].mean(-1)
    cosine_distance = 1 - F.cosine_similarity(foreground, near_background, dim=1)
    image_rms = feature.float().square().mean(dim=(1, 2, 3)).add(1e-8).sqrt()
    normalized_distance = (foreground - near_background).square().mean(dim=1).sqrt() / image_rms
    return cosine_distance.cpu().numpy(), normalized_distance.cpu().numpy()


def predict(arm):
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{arm}_regions.json"
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
        raise RuntimeError("Expected full testdev without SAM input")
    annotation_file = ROOT / "data/antiuav6k_common/annotations/instances_visible_common_test.json"
    dataset = json.loads(annotation_file.read_text(encoding="utf-8"))
    annotations = {int(item["image_id"]): item for item in dataset["annotations"]
                   if not item.get("iscrowd", 0)}
    images = {int(item["id"]): item for item in dataset["images"]}
    captured = {}
    handles = [
        model.backbone.register_forward_hook(
            lambda module, inputs, result: captured.__setitem__("rgb_s8", result[0].detach())),
        model.mote_fusion.register_forward_hook(
            lambda module, inputs, result: captured.update(
                pre_s16=inputs[0].detach(), post_s16=result[0].detach())),
    ]
    rows = []
    torch.set_num_threads(4)
    try:
        with torch.inference_mode():
            for step, (samples, targets) in enumerate(loader):
                model(samples.cuda())
                grid, valid = grid_for_batch(targets, annotations, images, samples.device)
                if not grid.is_cuda:
                    grid = grid.cuda()
                batch_rows = []
                for index, target in enumerate(targets):
                    if valid[index]:
                        image_id = int(target["image_id"])
                        ann = annotations[image_id]
                        batch_rows.append((index, {"image_id": image_id,
                                                   "scale": "small" if ann["area"] < 1024 else "medium"}))
                for name in ("rgb_s8", "pre_s16", "post_s16"):
                    cosine, distance = contrast(captured.pop(name), grid)
                    for index, row in batch_rows:
                        row[f"{name}_cosine_distance"] = float(cosine[index])
                        row[f"{name}_normalized_distance"] = float(distance[index])
                rows.extend(row for _, row in batch_rows)
                if step % 50 == 0:
                    print(arm, step, len(rows), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if len({row["image_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate image rows")
    result = {"arm": arm, "checkpoint": str(checkpoint),
              "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "definition": "9 bilinear samples inside GT box; 8 samples just outside box; contrast is between region mean vectors",
              "limitations": "GT box interior is not a segmentation mask; contrast does not itself prove detection use or preserved shape",
              "eligible": len(rows), "rows": rows}
    destination.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


def analyze():
    output = OUT / "paired_region_summary.json"
    if output.exists():
        raise FileExistsError(output)
    sources = {arm: json.loads((OUT / f"{arm}_regions.json").read_text(encoding="utf-8"))
               for arm in ARMS}
    indexed = {arm: {row["image_id"]: row for row in source["rows"]}
               for arm, source in sources.items()}
    if set(indexed["with_sam"]) != set(indexed["without_sam"]):
        raise RuntimeError("Different eligible images between arms")
    result = {"schema": "r2_sam_region_separation_v1", "status": "PASS",
              "eligible_images": len(indexed["with_sam"]), "groups": {},
              "limitations": ["GT-box region contrast is a coarse proxy, not true boundary or SAM mask quality.",
                              "Models were independently trained; this is paired observational feature analysis.",
                              "Raw cosine or normalized distances across distinct representation spaces need cautious interpretation."]}
    for scale in ("all", "small", "medium"):
        ids = [image_id for image_id, row in indexed["with_sam"].items()
               if scale == "all" or row["scale"] == scale]
        item = {"n": len(ids), "arms": {}, "sam_minus_no_s": {}}
        for arm in ARMS:
            item["arms"][arm] = {}
            for stage in ("rgb_s8", "pre_s16", "post_s16"):
                for measure in ("cosine_distance", "normalized_distance"):
                    key = f"{stage}_{measure}"
                    values = np.array([indexed[arm][image_id][key] for image_id in ids])
                    item["arms"][arm][key] = {"mean": float(values.mean()),
                                               "median": float(np.median(values))}
            for measure in ("cosine_distance", "normalized_distance"):
                change = np.array([indexed[arm][image_id][f"post_s16_{measure}"]
                                   - indexed[arm][image_id][f"pre_s16_{measure}"] for image_id in ids])
                item["arms"][arm][f"fusion_change_{measure}"] = {
                    "mean": float(change.mean()), "median": float(np.median(change)),
                    "positive_count": int((change > 0).sum())}
        for measure in ("cosine_distance", "normalized_distance"):
            for stage in ("rgb_s8", "pre_s16", "post_s16"):
                key = f"{stage}_{measure}"
                deltas = np.array([indexed["with_sam"][image_id][key]
                                   - indexed["without_sam"][image_id][key] for image_id in ids])
                item["sam_minus_no_s"][key] = {"mean": float(deltas.mean()),
                                                "median": float(np.median(deltas)),
                                                "positive_count": int((deltas > 0).sum())}
            changes = np.array([
                (indexed["with_sam"][image_id][f"post_s16_{measure}"]
                 - indexed["with_sam"][image_id][f"pre_s16_{measure}"])
                - (indexed["without_sam"][image_id][f"post_s16_{measure}"]
                   - indexed["without_sam"][image_id][f"pre_s16_{measure}"])
                for image_id in ids])
            item["sam_minus_no_s"][f"fusion_change_{measure}"] = {
                "mean": float(changes.mean()), "median": float(np.median(changes)),
                "positive_count": int((changes > 0).sum())}
        result["groups"][scale] = item
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def link_detection():
    output = OUT / "region_detection_link.json"
    if output.exists():
        raise FileExistsError(output)
    region = {arm: json.loads((OUT / f"{arm}_regions.json").read_text(encoding="utf-8"))
              for arm in ARMS}
    indexed = {arm: {row["image_id"]: row for row in payload["rows"]}
               for arm, payload in region.items()}
    instance_path = ROOT / "reports/M_R2_SAM_FACTORIAL_20E_TESTDEV/instance_audit/paired_instance_rows.json"
    instances = json.loads(instance_path.read_text(encoding="utf-8"))
    result = {"schema": "r2_sam_region_detection_link_v1", "status": "PASS", "groups": {},
              "caveat": "Exploratory observational correlations; feature spaces differ and GT-box contrast is not a segmentation metric."}
    for scale in ("small", "medium"):
        records = []
        for item in instances[scale]:
            image_id = item["image_id"]
            if image_id not in indexed["with_sam"]:
                continue
            feature_a = indexed["with_sam"][image_id]
            feature_b = indexed["without_sam"][image_id]
            detection_a = item["models"]["with_sam"]
            detection_b = item["models"]["without_sam"]
            records.append({
                "iou_delta": detection_a["best_iou_at_02"] - detection_b["best_iou_at_02"],
                "s8_delta": feature_a["rgb_s8_cosine_distance"] - feature_b["rgb_s8_cosine_distance"],
                "post_s16_delta": feature_a["post_s16_cosine_distance"] - feature_b["post_s16_cosine_distance"],
                "fusion_delta": (feature_a["post_s16_cosine_distance"] - feature_a["pre_s16_cosine_distance"])
                                - (feature_b["post_s16_cosine_distance"] - feature_b["pre_s16_cosine_distance"]),
                "recovered_075": bool(detection_a["hit02_0.75"] and not detection_b["hit02_0.75"]),
                "lost_075": bool(detection_b["hit02_0.75"] and not detection_a["hit02_0.75"]),
            })
        summary = {"n": len(records), "correlation_with_iou_delta": {}, "cohorts": {}}
        y = np.array([row["iou_delta"] for row in records])
        for key in ("s8_delta", "post_s16_delta", "fusion_delta"):
            x = np.array([row[key] for row in records])
            summary["correlation_with_iou_delta"][key] = float(np.corrcoef(x, y)[0, 1])
        cohorts = {
            "recovered_075": [row for row in records if row["recovered_075"]],
            "lost_075": [row for row in records if row["lost_075"]],
            "iou_gain_gt_002": [row for row in records if row["iou_delta"] > .02],
            "iou_loss_lt_minus002": [row for row in records if row["iou_delta"] < -.02],
        }
        for label, selected in cohorts.items():
            summary["cohorts"][label] = {"n": len(selected), "mean": {
                key: float(np.mean([row[key] for row in selected])) if selected else None
                for key in ("iou_delta", "s8_delta", "post_s16_delta", "fusion_delta")}}
        result["groups"][scale] = summary
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("predict", "analyze", "link"))
    parser.add_argument("--arm", choices=tuple(ARMS))
    args = parser.parse_args()
    if args.mode == "predict":
        if args.arm is None:
            parser.error("predict requires --arm")
        predict(args.arm)
    elif args.mode == "analyze":
        analyze()
    else:
        link_detection()
