"""Read-only diagnosis of S-QER candidate rank and ROI geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from preflight_sqer1 import build_config, load_tuning_state, move_targets, seed_all


def roi_from_box(box: torch.Tensor, reader) -> tuple[float, float, float, float]:
    cx, cy, width, height = [float(value) for value in box]
    width = max(width * reader.roi_expand, reader.min_roi_width)
    height = max(height * reader.roi_expand, reader.min_roi_height)
    return (
        max(0.0, cx - width / 2),
        max(0.0, cy - height / 2),
        min(1.0, cx + width / 2),
        min(1.0, cy + height / 2),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tuning-init", action="store_true")
    parser.add_argument("--checkpoint-state", choices=("model", "ema"), default="ema")
    parser.add_argument("--teacher-signal", action="store_true")
    parser.add_argument("--max-images", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(4)
    seed_all(0)
    repo = args.repo.resolve()
    cfg = build_config(repo, args.config.resolve())
    from src.zoo.dfine.sam_query_evidence_reader import _box_occupancy, _region_distributions
    model = cfg.model
    if args.tuning_init:
        saved = {"last_epoch": None}
        load_tuning_state(model, args.checkpoint.resolve())
    else:
        saved = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
        source = saved["model"] if args.checkpoint_state == "model" else saved["ema"]["module"]
        incompatible = model.load_state_dict(source, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = list(incompatible.missing_keys)
        if unexpected or any(not key.startswith("sqer.") for key in missing):
            raise RuntimeError(f"checkpoint incompatibility: missing={missing[:12]}, unexpected={unexpected[:12]}")
    model = model.cuda().eval()
    model.sqer_enabled = False
    matcher = cfg.criterion.matcher.cuda()
    loader = cfg.train_dataloader

    total = 0
    eligible = 0
    matched = 0
    matched_small = 0
    ranks = []
    ranks_small = []
    center_hit = 0
    center_hit_small = 0
    gt_coverages = []
    covered_at = {k: 0 for k in (64, 96, 128, 192, 256, 300)}
    covered_small_at = {k: 0 for k in covered_at}
    eligible_at = {k: 0 for k in covered_at}
    teacher_roi_checked = 0
    teacher_region_valid = 0
    teacher_distribution_tv = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for samples, targets in loader:
            if total >= args.max_images:
                break
            count = min(len(targets), args.max_images - total)
            samples = samples[:count].cuda()
            targets = move_targets(targets[:count], torch.device("cuda"))
            outputs = model(samples)
            indices = matcher(outputs, targets)["indices"]
            score = outputs["pred_logits"].detach().float().sigmoid().amax(-1)
            order = score.argsort(dim=1, descending=True)
            inv_rank = order.argsort(dim=1)
            for image_index, (src_ids, tgt_ids) in enumerate(indices):
                target = targets[image_index]
                teacher_ok = (
                    len(target["boxes"]) == 1
                    and float(torch.as_tensor(target.get("sam_quality", 0)).reshape(-1)[0]) > 0
                    and target.get("masks") is not None
                )
                for src, tgt in zip(src_ids.tolist(), tgt_ids.tolist()):
                    matched += 1
                    box = target["boxes"][tgt].detach().float().cpu()
                    size = math.sqrt(max(float(box[2] * box[3]) * 512 * 640, 0.0))
                    small = size < 32
                    matched_small += int(small)
                    rank = int(inv_rank[image_index, src]) + 1
                    ranks.append(rank)
                    if small:
                        ranks_small.append(rank)
                    if teacher_ok:
                        eligible += 1
                    if args.teacher_signal and teacher_ok and rank <= 64:
                        teacher_roi_checked += 1
                        roi_grid = model.sqer._roi_grid(
                            outputs["pred_boxes"][image_index:image_index + 1],
                            torch.tensor([[src]], device="cuda"),
                        )[0, 0].unsqueeze(0)
                        mask = target["masks"][0].float()
                        sam_occupancy = F.grid_sample(
                            mask[None, None], roi_grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False,
                        )[0, 0]
                        box_mask = _box_occupancy(
                            target["boxes"][tgt].reshape(1, 1, 4),
                            int(mask.shape[-2]), int(mask.shape[-1]),
                        )[0, 0]
                        box_occupancy = F.grid_sample(
                            box_mask[None, None], roi_grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False,
                        )[0, 0]
                        sam_distribution, sam_valid, _ = _region_distributions(sam_occupancy)
                        box_distribution, box_valid, _ = _region_distributions(box_occupancy)
                        if bool(sam_valid[0] and box_valid[0]):
                            teacher_region_valid += 1
                            teacher_distribution_tv.append(float(
                                (sam_distribution - box_distribution).abs().sum(-1).mean() / 2
                            ))
                    for topk in covered_at:
                        if rank <= topk:
                            covered_at[topk] += 1
                            covered_small_at[topk] += int(small)
                            eligible_at[topk] += int(teacher_ok)
                    prediction = outputs["pred_boxes"][image_index, src].detach().float().cpu()
                    x1, y1, x2, y2 = roi_from_box(prediction, model.sqer)
                    gx1, gy1 = float(box[0] - box[2] / 2), float(box[1] - box[3] / 2)
                    gx2, gy2 = float(box[0] + box[2] / 2), float(box[1] + box[3] / 2)
                    hit = x1 <= float(box[0]) <= x2 and y1 <= float(box[1]) <= y2
                    center_hit += int(hit)
                    center_hit_small += int(hit and small)
                    intersection = max(0.0, min(x2, gx2) - max(x1, gx1)) * max(0.0, min(y2, gy2) - max(y1, gy1))
                    gt_coverages.append(intersection / max((gx2 - gx1) * (gy2 - gy1), 1e-12))
            total += count
            if total % 256 == 0:
                print(f"diagnosed {total} images", flush=True)

    ranks.sort()
    ranks_small.sort()
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": saved.get("last_epoch"),
        "checkpoint_state": "tuning_init" if args.tuning_init else args.checkpoint_state,
        "images": total,
        "matched": matched,
        "matched_small": matched_small,
        "teacher_eligible_matched": eligible,
        "rank_quantiles": {str(q): ranks[min(len(ranks) - 1, int(q * (len(ranks) - 1)))] for q in (0.5, 0.9, 0.95, 0.99)},
        "rank_small_quantiles": {str(q): ranks_small[min(len(ranks_small) - 1, int(q * (len(ranks_small) - 1)))] for q in (0.5, 0.9, 0.95, 0.99)},
        "topk": {
            str(k): {
                "all": covered_at[k] / max(matched, 1),
                "small": covered_small_at[k] / max(matched_small, 1),
                "teacher_eligible": eligible_at[k] / max(eligible, 1),
            }
            for k in covered_at
        },
        "matched_roi_gt_center_coverage": center_hit / max(matched, 1),
        "matched_roi_small_gt_center_coverage": center_hit_small / max(matched_small, 1),
        "matched_roi_gt_area_coverage_mean": sum(gt_coverages) / max(len(gt_coverages), 1),
        "teacher_signal": {
            "checked_top64_eligible": teacher_roi_checked,
            "valid_both_regions": teacher_region_valid,
            "valid_fraction": teacher_region_valid / max(teacher_roi_checked, 1),
            "sam_box_region_distribution_tv_mean": sum(teacher_distribution_tv) / max(len(teacher_distribution_tv), 1),
        } if args.teacher_signal else None,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
