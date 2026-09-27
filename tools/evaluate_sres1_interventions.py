#!/usr/bin/env python3
"""Evaluate custom object-size AP and an optional S8 importance predictor."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def binary_auc(y, score):
    y, score = np.asarray(y, dtype=np.int64), np.asarray(score)
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    np_, nn_ = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - np_ * (np_ + 1) / 2) / max(np_ * nn_, 1))


def average_precision(y, score):
    y = np.asarray(y, dtype=np.int64)[np.argsort(-np.asarray(score))]
    n = int(y.sum())
    return float((np.cumsum(y) / np.arange(1, len(y) + 1) * y).sum() / max(n, 1))


def make_mask(targets, height, width, device, image_height, image_width):
    masks = torch.zeros((len(targets), 1, height, width), dtype=torch.bool, device=device)
    centers = []
    for bi, target in enumerate(targets):
        image_centers = []
        boxes = target["boxes"]
        # Training targets are normalized cxcywh after ConvertBoxes, whereas
        # the validation pipeline keeps resized absolute xyxy boxes.  Handle
        # both explicitly so the diagnostic mask matches the loss mask.
        absolute_xyxy = bool(boxes.numel() and boxes.max() > 2)
        image_h, image_w = image_height, image_width
        for box in boxes:
            if absolute_xyxy:
                bx1, by1, bx2, by2 = box
                fx1, fy1 = bx1 / image_w * width, by1 / image_h * height
                fx2, fy2 = bx2 / image_w * width, by2 / image_h * height
                center_x, center_y = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            else:
                cx, cy, bw, bh = box
                fx1, fy1 = (cx - bw / 2) * width, (cy - bh / 2) * height
                fx2, fy2 = (cx + bw / 2) * width, (cy + bh / 2) * height
                center_x, center_y = cx * width, cy * height
            x1 = max(0, min(width - 1, int(torch.floor(fx1).item())))
            y1 = max(0, min(height - 1, int(torch.floor(fy1).item())))
            x2 = max(x1 + 1, min(width, int(torch.ceil(fx2).item())))
            y2 = max(y1 + 1, min(height, int(torch.ceil(fy2).item())))
            masks[bi, 0, y1:y2, x1:x2] = True
            image_centers.append((min(width - 1, int(center_x)), min(height - 1, int(center_y))))
        centers.append(image_centers)
    return masks, centers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--weight-source", choices=("ema", "model"), default="ema")
    p.add_argument(
        "--sres1-gate-mode",
        choices=("learned", "one", "zero", "shuffled"),
        default="learned",
        help="Evaluation-only intervention on the S-RES1 spatial gate.",
    )
    p.add_argument(
        "--sres1-scale-multiplier",
        type=float,
        default=1.0,
        help="Multiply the learned S-RES1 residual scale at evaluation time.",
    )
    p.add_argument(
        "--lad-mode",
        choices=("learned", "uniform", "shuffled"),
        default="learned",
        help="Evaluation-only intervention on LAD four-phase weights.",
    )
    p.add_argument(
        "--bpc-gate-mode",
        choices=("learned", "zero", "shifted", "constant", "one"),
        default="learned",
        help="Evaluation-only intervention on the BPC boundary carrier gate.",
    )
    p.add_argument(
        "--bpc-scale-multiplier",
        type=float,
        default=1.0,
        help="Multiply the learned bounded BPC residual scale at evaluation time.",
    )
    p.add_argument(
        "--bpc-output-gain",
        type=float,
        default=1.0,
        help="Evaluation-only gain applied after the trained bounded BPC scale.",
    )
    p.add_argument(
        "--bpc-gate-logit-bias",
        type=float,
        default=0.0,
        help="Evaluation-only bias added to learned BPC boundary logits.",
    )
    p.add_argument(
        "--bpc-linear-residual",
        action="store_true",
        help="Remove the final tanh compression from the BPC carrier.",
    )
    p.add_argument("--spatial-budget-pct", type=int, choices=(5, 10, 20, 30))
    p.add_argument(
        "--spatial-budget-mode",
        choices=("learned", "random"),
        help="Keep only this type of S8 locations before S8->S16.",
    )
    a = p.parse_args()
    sys.path.insert(0, str(a.repo))
    from src.core import YAMLConfig
    from faster_coco_eval import COCOeval_faster

    cfg = YAMLConfig(str(a.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model.cuda().eval()
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    weights = state["ema"]["module"] if a.weight_source == "ema" and "ema" in state else state["model"]
    model.load_state_dict(weights, strict=True)
    detail_residual = getattr(model.backbone, "srtod_detail_residual", None)
    if detail_residual is not None:
        detail_residual.gate_mode = a.sres1_gate_mode
        with torch.no_grad():
            detail_residual.scale.mul_(a.sres1_scale_multiplier)
    bpc_branch = getattr(model.backbone, "bpc_branch", None)
    if bpc_branch is not None:
        bpc_branch.gate_mode = a.bpc_gate_mode
        if a.bpc_output_gain < 0:
            p.error("--bpc-output-gain must be non-negative")
        bpc_branch.intervention_output_gain = a.bpc_output_gain
        bpc_branch.intervention_gate_logit_bias = a.bpc_gate_logit_bias
        bpc_branch.intervention_linear_residual = a.bpc_linear_residual
        if a.bpc_scale_multiplier < 0:
            p.error("--bpc-scale-multiplier must be non-negative")
        with torch.no_grad():
            learned_scale = bpc_branch.max_scale * torch.tanh(bpc_branch.scale_logit)
            target_scale = learned_scale * a.bpc_scale_multiplier
            if target_scale >= bpc_branch.max_scale:
                p.error(
                    "--bpc-scale-multiplier exceeds the branch's bounded max_scale"
                )
            target_ratio = target_scale / bpc_branch.max_scale
            bpc_branch.scale_logit.copy_(torch.atanh(target_ratio))
    elif a.bpc_gate_mode != "learned":
        p.error("--bpc-gate-mode requires a checkpoint/config with an active BPC branch")
    elif a.bpc_scale_multiplier != 1.0:
        p.error("--bpc-scale-multiplier requires a checkpoint/config with an active BPC branch")
    if (a.spatial_budget_pct is None) != (a.spatial_budget_mode is None):
        p.error("--spatial-budget-pct and --spatial-budget-mode must be provided together")

    budget_hook = None
    if a.spatial_budget_pct is not None:
        aux_stage = getattr(model.backbone, "spatial_aux_stage", -1)
        if aux_stage < 0 or getattr(model.backbone, "spatial_aux_head", None) is None:
            p.error("spatial-budget evaluation requires an S-AUX model/config")
        next_stage = aux_stage + 1
        if next_stage >= len(model.backbone.stages):
            p.error("S-AUX stage has no following downsample to intervene on")
        budget_rng = torch.Generator(device="cpu").manual_seed(20260808)

        def apply_spatial_budget(module, inputs):
            feature = inputs[0]
            score = model.backbone.spatial_importance_logits.detach().sigmoid()
            if score.shape[-2:] != feature.shape[-2:]:
                score = torch.nn.functional.interpolate(
                    score, size=feature.shape[-2:], mode="bilinear", align_corners=False
                )
            flat_score = score[:, 0].flatten(1)
            keep_n = max(1, round(flat_score.shape[1] * a.spatial_budget_pct / 100))
            keep = torch.zeros_like(flat_score, dtype=torch.bool)
            for batch_idx in range(flat_score.shape[0]):
                if a.spatial_budget_mode == "learned":
                    indices = torch.topk(flat_score[batch_idx], keep_n).indices
                else:
                    indices = torch.randperm(
                        flat_score.shape[1], generator=budget_rng
                    )[:keep_n].to(flat_score.device)
                keep[batch_idx, indices] = True
            keep = keep.reshape(feature.shape[0], 1, *feature.shape[-2:])
            return (feature * keep.to(feature.dtype),) + tuple(inputs[1:])

        budget_hook = model.backbone.stages[next_stage].register_forward_pre_hook(
            apply_spatial_budget
        )
    loader, post = cfg.val_dataloader, cfg.postprocessor
    coco_gt = loader.dataset.coco
    cat_ids = sorted(coco_gt.getCatIds())
    detections, pixel_scores, pixel_labels = [], [], []
    map_kind = None
    region = {k: [] for k in ("target", "ring", "background", "hard_background", "random_background")}
    coverage = {str(k): {"target_cells": [], "centers": []} for k in (5, 10, 20, 30)}
    # Hard background is selected independently from the LAD phase weights:
    # it is the highest-energy non-target area in the feature entering LAD.
    # Random background uses an equal number of cells and a fixed RNG seed.
    lad_modules = [m for m in model.modules() if hasattr(m, "last_selectivity_map")]
    for module in lad_modules:
        module.phase_weight_mode = a.lad_mode
    lad_input_energy = {}
    hooks = []
    for module in lad_modules:
        def save_input_energy(mod, inputs, key=id(module)):
            lad_input_energy[key] = inputs[0].detach().float().square().mean(dim=1, keepdim=True).sqrt()
        hooks.append(module.register_forward_pre_hook(save_input_energy))
    rng = torch.Generator(device="cpu").manual_seed(20260808)
    lad_entropy, lad_candidate_variance, lad_weight_deviation = [], [], []

    with torch.inference_mode():
        for samples, targets in loader:
            samples = samples.cuda()
            targets = [{k: v.cuda() if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            out = model(samples)
            sizes = torch.stack([t["orig_size"] for t in targets])
            results = post(out, sizes)
            for target, result in zip(targets, results):
                boxes = result["boxes"].detach().cpu()
                boxes[:, 2:] -= boxes[:, :2]
                for box, score, label in zip(boxes.tolist(), result["scores"].tolist(), result["labels"].tolist()):
                    detections.append({"image_id": int(target["image_id"]), "category_id": int(cat_ids[int(label)]), "bbox": box, "score": float(score)})

            if "spatial_importance_logits" in out:
                prob = out["spatial_importance_logits"].sigmoid()
                map_kind = "supervised_spatial_importance_probability"
                active_lad = [module for module in lad_modules if module.last_selectivity_map is not None]
                if active_lad:
                    lad_entropy.append(float(active_lad[0].last_phase_entropy))
                    lad_candidate_variance.append(float(active_lad[0].last_candidate_variance))
                    lad_weight_deviation.append(float(active_lad[0].last_weight_deviation_from_uniform))
            else:
                active_lad = [module for module in lad_modules if module.last_selectivity_map is not None]
                lad_maps = [module.last_selectivity_map for module in active_lad]
                if not lad_maps:
                    continue
                # S-ST experiments contain exactly one LAD. This is the mean
                # maximum four-phase sampling probability, not a foreground
                # probability; use it only for ranking/region comparisons.
                prob = lad_maps[0]
                map_kind = "lad_mean_max_phase_selectivity"
                lad_entropy.append(float(active_lad[0].last_phase_entropy))
                lad_candidate_variance.append(float(active_lad[0].last_candidate_variance))
                lad_weight_deviation.append(float(active_lad[0].last_weight_deviation_from_uniform))
            mask, centers = make_mask(
                targets, prob.shape[-2], prob.shape[-1], prob.device,
                samples.shape[-2], samples.shape[-1]
            )
            dilated = torch.nn.functional.max_pool2d(mask.float(), 5, 1, 2).bool()
            ring = dilated & ~mask
            background = ~dilated
            pixel_scores.extend(prob.flatten().cpu().tolist())
            pixel_labels.extend(mask.flatten().cpu().tolist())
            for bi in range(len(targets)):
                for name, area in (("target", mask[bi]), ("ring", ring[bi]), ("background", background[bi])):
                    if area.any(): region[name].append(float(prob[bi][area].mean()))
                if active_lad:
                    energy = lad_input_energy[id(active_lad[0])][bi : bi + 1]
                    energy = torch.nn.functional.adaptive_avg_pool2d(energy, prob.shape[-2:])[0, 0]
                    bg = background[bi, 0]
                    bg_indices = bg.flatten().nonzero(as_tuple=False).flatten()
                    sample_n = min(int(mask[bi, 0].sum().item()), int(bg_indices.numel()))
                    if sample_n:
                        bg_energy = energy.flatten()[bg_indices]
                        hard_indices = bg_indices[torch.topk(bg_energy, sample_n).indices]
                        random_order = torch.randperm(bg_indices.numel(), generator=rng)[:sample_n].to(bg_indices.device)
                        random_indices = bg_indices[random_order]
                        flat_prob = prob[bi, 0].flatten()
                        region["hard_background"].append(float(flat_prob[hard_indices].mean()))
                        region["random_background"].append(float(flat_prob[random_indices].mean()))
                flat = prob[bi, 0].flatten()
                for pct in (5, 10, 20, 30):
                    k = max(1, round(flat.numel() * pct / 100))
                    chosen = torch.zeros_like(flat, dtype=torch.bool)
                    chosen[torch.topk(flat, k).indices] = True
                    chosen = chosen.reshape(prob.shape[-2:])
                    target_cells = mask[bi, 0]
                    coverage[str(pct)]["target_cells"].append(float((chosen & target_cells).sum() / target_cells.sum().clamp(min=1)))
                    cc = centers[bi]
                    coverage[str(pct)]["centers"].append(float(np.mean([bool(chosen[y, x]) for x, y in cc])) if cc else 0.0)

    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")
    labels = ["all", "lt8", "8to16", "16to32", "32to48", "ge48"]
    edges = [(0, 1e10), (0, 8**2), (8**2, 16**2), (16**2, 32**2), (32**2, 48**2), (48**2, 1e10)]
    evaluator.params.areaRng, evaluator.params.areaRngLbl = [list(x) for x in edges], labels
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate(); evaluator.accumulate()
    precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]
    custom = {}
    for ai, label in enumerate(labels):
        all_p = precision[:, :, :, ai, -1]; p50 = precision[0, :, :, ai, -1]
        i75 = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.75))); p75 = precision[i75, :, :, ai, -1]
        rr = recall[:, :, ai, -1]
        mean_valid = lambda x: float(x[x > -1].mean()) if np.any(x > -1) else None
        custom[label] = {"AP50_95": mean_valid(all_p), "AP50": mean_valid(p50), "AP75": mean_valid(p75), "AR100": mean_valid(rr)}

    result = {
        "checkpoint": str(a.checkpoint),
        "weight_source": a.weight_source,
        "lad_mode": a.lad_mode,
        "bpc_gate_mode": a.bpc_gate_mode,
        "bpc_scale_multiplier": a.bpc_scale_multiplier,
        "bpc_output_gain": a.bpc_output_gain,
        "bpc_gate_logit_bias": a.bpc_gate_logit_bias,
        "bpc_linear_residual": a.bpc_linear_residual,
        "bpc_effective_scale": (
            float(bpc_branch.max_scale * torch.tanh(bpc_branch.scale_logit).detach())
            if bpc_branch is not None
            else None
        ),
        "spatial_budget_pct": a.spatial_budget_pct,
        "spatial_budget_mode": a.spatial_budget_mode,
        "custom_size_metrics": custom,
    }
    if pixel_scores:
        result["importance"] = {
            "map_kind": map_kind,
            "pixel_roc_auc": binary_auc(pixel_labels, pixel_scores),
            "pixel_average_precision": average_precision(pixel_labels, pixel_scores),
            "region_mean": {k: (float(np.mean(v)) if v else None) for k, v in region.items()},
            "region_definition": {
                "target": "GT box cells",
                "ring": "5x5 dilation around GT cells excluding target",
                "background": "all cells outside the dilated target region",
                "hard_background": "highest LAD-input RMS-energy background cells, count matched to target cells",
                "random_background": "fixed-seed random background cells, count matched to target cells",
            },
            "top_percent_coverage": {k: {q: float(np.mean(v)) for q, v in d.items()} for k, d in coverage.items()},
        }
        if lad_entropy:
            result["importance"]["lad_internal"] = {
                "normalized_phase_entropy": float(np.mean(lad_entropy)),
                "candidate_phase_variance": float(np.mean(lad_candidate_variance)),
                "mean_abs_weight_deviation_from_uniform": float(np.mean(lad_weight_deviation)),
            }
    for hook in hooks:
        hook.remove()
    if budget_hook is not None:
        budget_hook.remove()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    scale_suffix = str(a.bpc_scale_multiplier).replace(".", "p")
    suffix = (
        f"{a.weight_source}_{a.lad_mode}_bpc-{a.bpc_gate_mode}"
        f"_scale-{scale_suffix}x"
        f"_gain-{str(a.bpc_output_gain).replace('.', 'p')}x"
        f"_bias-{str(a.bpc_gate_logit_bias).replace('.', 'p')}"
        f"_linear-{int(a.bpc_linear_residual)}"
    )
    if a.spatial_budget_pct is not None:
        suffix += f"_budget{a.spatial_budget_pct}_{a.spatial_budget_mode}"
    (a.output_dir / f"custom_metrics_and_importance_{suffix}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
