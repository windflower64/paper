"""Read-only IR candidate and RGB write-region diagnostics for M-OTE2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def region_mask(boxes, height, width, margin_cells=0.0):
    boxes = torch.as_tensor(boxes).float().reshape(-1, 4)
    ys = (torch.arange(height).float() + 0.5) / height
    xs = (torch.arange(width).float() + 0.5) / width
    xx = xs[None, :]
    yy = ys[:, None]
    mask = torch.zeros(height, width, dtype=torch.bool)
    for cx, cy, bw, bh in boxes:
        mask |= (
            ((xx - cx).abs() <= bw * 0.5 + margin_cells / width)
            & ((yy - cy).abs() <= bh * 0.5 + margin_cells / height)
        )
    return mask


def normalized_cxcywh(boxes, height, width, default_format="xyxy"):
    fmt = str(getattr(boxes, "format", default_format)).lower()
    values = torch.as_tensor(boxes).float().reshape(-1, 4).clone()
    if values.numel() == 0:
        return values
    if "xyxy" in fmt:
        x1, y1, x2, y2 = values.unbind(-1)
        values = torch.stack(
            ((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1),
            dim=-1,
        )
    elif "cxcywh" not in fmt:
        raise ValueError(f"Unknown box format: {fmt}")
    if values.abs().max() > 1.5:
        values[:, (0, 2)] /= width
        values[:, (1, 3)] /= height
    return values


def mean(values):
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo, config, checkpoint, output = (
        args.repo.resolve(), args.config.resolve(),
        args.checkpoint.resolve(), args.output.resolve(),
    )
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from src.core import YAMLConfig
    from src.misc import dist_utils
    from src.solver import TASKS

    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)
    cfg = YAMLConfig(
        str(config), resume=str(checkpoint),
        output_dir=str(output.parent / f"candidate_runtime_{output.stem}"),
    )
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    module = solver.ema.module if solver.ema else solver.model
    if module.mote_fusion is None:
        raise RuntimeError("Expected an M-OTE2 checkpoint")
    if hasattr(module, "set_training_epoch"):
        module.set_training_epoch(int(solver.last_epoch))
    module.eval()

    captured = {}

    def capture_write(_module, inputs, output_value):
        before = inputs[0].detach().float()
        delta = output_value[0].detach().float() - before
        captured["delta_energy"] = delta.square().mean(dim=1).cpu()
        captured["base_energy"] = before.square().mean(dim=1).cpu()

    handle = module.mote_fusion.register_forward_hook(capture_write)
    gt_count = 0
    gt_hit_strict = 0
    gt_hit_near = 0
    candidate_count = 0
    candidate_inside = 0
    positive_images = 0
    negative_images = 0
    positive_confidence = []
    negative_confidence = []
    center_objectness = []
    background_objectness = []
    gate_means = []
    update_ratios = []
    rgb_target_update = []
    rgb_background_update = []

    with torch.no_grad():
        for samples, targets in solver.val_dataloader:
            module(samples.to(solver.device))
            logits = module.mote_fusion.last_objectness_logits.float().cpu()
            probability = logits.sigmoid()
            batch, _, h8, w8 = probability.shape
            maxima = F.max_pool2d(probability, kernel_size=3, stride=1, padding=1)
            peak = probability * (probability >= maxima).float()
            scores, indices = peak.flatten(1).topk(
                min(module.mote_fusion.num_candidates, h8 * w8), dim=1
            )
            x = ((indices % w8).float() + 0.5) / w8
            y = (torch.div(indices, w8, rounding_mode="floor").float() + 0.5) / h8
            gate_means.append(float(module.mote_fusion.last_gate_mean))
            update_ratios.append(float(module.mote_fusion.last_update_ratio))

            for index, target in enumerate(targets):
                ir_boxes = normalized_cxcywh(
                    target["infrared_boxes"], samples.shape[-2], samples.shape[-1]
                )
                score = float(scores[index].max())
                if len(ir_boxes):
                    positive_images += 1
                    positive_confidence.append(score)
                    for cx, cy, bw, bh in ir_boxes:
                        dx = (x[index] - cx).abs()
                        dy = (y[index] - cy).abs()
                        strict = (dx <= bw * 0.5) & (dy <= bh * 0.5)
                        near = (dx <= bw * 0.5 + 1.0 / w8) & (
                            dy <= bh * 0.5 + 1.0 / h8
                        )
                        gt_count += 1
                        gt_hit_strict += int(strict.any())
                        gt_hit_near += int(near.any())
                    inside = region_mask(ir_boxes, h8, w8)
                    nearest_x = ((ir_boxes[:, 0] * w8).long()).clamp(0, w8 - 1)
                    nearest_y = ((ir_boxes[:, 1] * h8).long()).clamp(0, h8 - 1)
                    center_objectness.extend(
                        probability[index, 0, nearest_y, nearest_x].tolist()
                    )
                    if (~inside).any():
                        background_objectness.append(
                            float(probability[index, 0][~inside].mean())
                        )
                    candidate_count += len(indices[index])
                    candidate_inside += int(
                        inside[(y[index] * h8).long().clamp(0, h8 - 1),
                               (x[index] * w8).long().clamp(0, w8 - 1)].sum()
                    )
                else:
                    negative_images += 1
                    negative_confidence.append(score)

                visible_boxes = normalized_cxcywh(
                    target["boxes"], samples.shape[-2], samples.shape[-1]
                )
                h16, w16 = captured["delta_energy"].shape[-2:]
                target_region = region_mask(visible_boxes, h16, w16, margin_cells=1.0)
                if target_region.any():
                    delta = captured["delta_energy"][index]
                    base = captured["base_energy"][index]
                    rgb_target_update.append(
                        float((delta[target_region].mean() / base[target_region].mean().clamp_min(1e-9)).sqrt())
                    )
                    if (~target_region).any():
                        rgb_background_update.append(
                            float((delta[~target_region].mean() / base[~target_region].mean().clamp_min(1e-9)).sqrt())
                        )

    handle.remove()
    result = {
        "schema": "mote2_candidate_recall_v2",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": int(solver.last_epoch),
        "images": positive_images + negative_images,
        "positive_images": positive_images,
        "negative_images": negative_images,
        "ir_gt_count": gt_count,
        "topk_ir_box_recall_strict": gt_hit_strict / gt_count if gt_count else None,
        "topk_ir_box_recall_one_cell_margin": gt_hit_near / gt_count if gt_count else None,
        "candidate_fraction_inside_ir_box": candidate_inside / candidate_count if candidate_count else None,
        "mean_top_candidate_confidence_positive": mean(positive_confidence),
        "mean_top_candidate_confidence_negative": mean(negative_confidence),
        "mean_objectness_at_ir_gt_center": mean(center_objectness),
        "mean_objectness_outside_ir_gt": mean(background_objectness),
        "mean_gate": mean(gate_means),
        "mean_write_rms_ratio": mean(update_ratios),
        "rgb_target_region_write_rms_ratio": mean(rgb_target_update),
        "rgb_background_region_write_rms_ratio": mean(rgb_background_update),
        "gt_use": "offline diagnostic only; no GT passed into detector forward",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    dist_utils.cleanup()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
