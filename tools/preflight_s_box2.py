#!/usr/bin/env python3
"""S-BOX2 hard-gate preflight on a real SAM batch; no checkpoints are written."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "experiments/phase_s/s_box2_extreme_point_jointinit_b32_60e_local.yml",
    )
    parser.add_argument(
        "--a00-config",
        type=Path,
        default=ROOT / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument(
        "--tuning",
        type=Path,
        default=WORKSPACE / "weights/dfine_n_coco.pth",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE
        / "reports/23_sam_box_alignment/S_BOX2_PREFLIGHT_B16/report.json",
    )
    return parser.parse_args()


def checkpoint_weights(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("ema", {}).get("module") is not None:
        return state["ema"]["module"], "ema.module"
    if "model" in state:
        return state["model"], "model"
    return state, "raw"


def load_matched(model, weights):
    current = model.state_dict()
    matched = {
        key: value
        for key, value in weights.items()
        if key in current and current[key].shape == value.shape
    }
    missing = sorted(set(current).difference(matched))
    model.load_state_dict(matched, strict=False)
    return missing


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def translate_no_wrap(x, shift_x):
    result = torch.zeros_like(x)
    width = x.shape[-1]
    src_x1, src_x2 = max(0, -shift_x), min(width, width - shift_x)
    dst_x1, dst_x2 = max(0, shift_x), min(width, width + shift_x)
    if src_x2 > src_x1:
        result[..., dst_x1:dst_x2] = x[..., src_x1:src_x2]
    return result


def map_loss(criterion, logits, target, neighborhood, valid):
    focal_map = torchvision.ops.sigmoid_focal_loss(
        logits,
        target,
        alpha=criterion.sbox_focal_alpha,
        gamma=criterion.sbox_focal_gamma,
        reduction="none",
    )
    positive_mass = target.sum((-2, -1)).clamp_min(1.0)
    negative_weight = neighborhood * (1.0 - target)
    negative_mass = negative_weight.sum((-2, -1)).clamp_min(1.0)
    focal = (focal_map * target).sum((-2, -1)) / positive_mass
    focal = focal + (focal_map * negative_weight).sum((-2, -1)) / negative_mass
    probability = logits.sigmoid() * neighborhood
    intersection = (probability * target).sum((-2, -1))
    denominator = probability.sum((-2, -1)) + target.sum((-2, -1))
    dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    per_side = focal + criterion.sbox_dice_weight * dice
    return (per_side * valid).sum() / valid.sum().clamp_min(1.0)


def cosine_and_ratio(left, right):
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    cosine = F.cosine_similarity(left_flat, right_flat, dim=1, eps=1e-12)
    ratio = right_flat.norm(dim=1) / left_flat.norm(dim=1).clamp_min(1e-12)
    return {
        "cosine_mean": float(cosine.mean()),
        "cosine_min": float(cosine.min()),
        "raw_aux_over_box_norm_mean": float(ratio.mean()),
        "raw_aux_over_box_norm_max": float(ratio.max()),
    }


def tensor_max_error(left, right):
    if left.dtype == torch.bool or not (left.is_floating_point() or left.is_complex()):
        return float((left != right).any())
    return float((left - right).abs().max())


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("S-BOX2 preflight requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # Construct A00 first. S-BOX preserves the shared RNG stream explicitly.
    a00_cfg = YAMLConfig(str(args.a00_config))
    a00_cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    torch.manual_seed(args.seed)
    a00 = a00_cfg.model
    sbox_cfg = YAMLConfig(str(args.config))
    sbox_cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    sbox_cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = args.batch_size
    sbox_cfg.yaml_cfg["train_dataloader"]["num_workers"] = 0
    torch.manual_seed(args.seed)
    model = sbox_cfg.model
    criterion = sbox_cfg.criterion

    a00_state = a00.state_dict()
    model_state = model.state_dict()
    shared_keys = sorted(set(a00_state).intersection(model_state))
    shared_initial_error = max(
        tensor_max_error(a00_state[key], model_state[key])
        for key in shared_keys
    )
    sbox_only_keys = sorted(set(model_state).difference(a00_state))

    weights, weight_source = checkpoint_weights(args.tuning)
    a00_missing = load_matched(a00, weights)
    sbox_missing = load_matched(model, weights)
    shared_loaded_error = max(
        tensor_max_error(a00.state_dict()[key], model.state_dict()[key])
        for key in shared_keys
    )

    loader = sbox_cfg.train_dataloader
    samples, targets = next(iter(loader))
    samples = samples.to(device)
    targets = move_targets(targets, device)
    a00 = a00.to(device).eval()
    model = model.to(device).eval()
    criterion = criterion.to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        torch.manual_seed(args.seed)
        a00_outputs = a00(samples)
        torch.manual_seed(args.seed)
        sbox_eval_outputs = model(samples)
    eval_box_error = float(
        (a00_outputs["pred_boxes"] - sbox_eval_outputs["pred_boxes"]).abs().max()
    )
    eval_logit_error = float(
        (a00_outputs["pred_logits"] - sbox_eval_outputs["pred_logits"]).abs().max()
    )
    eval_omits_sbox = "sbox_extreme_logits" not in sbox_eval_outputs
    del a00, a00_outputs, sbox_eval_outputs
    torch.cuda.empty_cache()

    model.train()
    cache = {}

    def capture_s8(_module, _inputs, output):
        cache["s8"] = output
        output.retain_grad()

    hook = model.backbone.stages[1].register_forward_hook(capture_s8)
    try:
        model.set_training_epoch(0)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(samples, targets=targets)
        logits = outputs["sbox_extreme_logits"]
        components = criterion._sbox_loss_components(logits, targets)
        initial_loss = float(components["combined"])
        valid = components["valid"].detach()
        target_mass = components["target"].sum((-2, -1)).detach()
        s8_feature = cache["s8"].detach()

        # Epoch 0-4 trains the side head only. A cached S8 batch is sufficient
        # to verify direction capacity without changing shared detector weights.
        head_optimizer = torch.optim.AdamW(
            model.backbone.sbox_head.parameters(), lr=1e-3, weight_decay=0.0
        )
        target = components["target"].detach()
        neighborhood = components["neighborhood"].detach()
        for _ in range(args.head_warmup_steps):
            head_optimizer.zero_grad(set_to_none=True)
            warm_logits = model.backbone.sbox_head(s8_feature)
            warm_loss = map_loss(
                criterion, warm_logits.float(), target, neighborhood, valid
            )
            warm_loss.backward()
            head_optimizer.step()
        with torch.no_grad():
            warm_logits = model.backbone.sbox_head(s8_feature).float()
            correct_loss = map_loss(
                criterion, warm_logits, target, neighborhood, valid
            )
            shifted_target = translate_no_wrap(target, 4)
            shifted_neighborhood = translate_no_wrap(neighborhood, 4)
            shifted_loss = map_loss(
                criterion,
                warm_logits,
                shifted_target,
                shifted_neighborhood,
                valid,
            )

        model.zero_grad(set_to_none=True)
        model.set_training_epoch(10)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(samples, targets=targets)
        losses = criterion(outputs, targets, epoch=10)
        s8 = cache["s8"]
        box_keys = [
            key for key in losses if key.startswith(("loss_bbox", "loss_giou"))
        ]
        box_loss = sum(losses[key] for key in box_keys)
        aux_unweighted = criterion._sbox_loss_components(
            outputs["sbox_extreme_logits"], targets
        )["combined"]
        box_gradient = torch.autograd.grad(box_loss, s8, retain_graph=True)[0]
        aux_gradient = torch.autograd.grad(aux_unweighted, s8, retain_graph=True)[0]
        compatibility = cosine_and_ratio(box_gradient, aux_gradient)
        target_ratio = 0.05 if compatibility["cosine_mean"] < 0 else 0.10
        recommended_weight = min(
            1.0,
            target_ratio
            / max(compatibility["raw_aux_over_box_norm_mean"], 1e-12),
        )
        recommended_weight = float(f"{recommended_weight:.6g}")

        optimizer = sbox_cfg.optimizer
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        sbox_parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if "sbox_head" in name
        }
        optimizer_covers_sbox = all(
            id(parameter) in optimizer_parameter_ids
            for parameter in sbox_parameters.values()
        )

        # The compatibility probe above deliberately retains the graph twice.
        # Release it before measuring the real training step; otherwise the
        # reported peak includes both graphs and can exceed physical VRAM on
        # Windows because CUDA allocations may spill into shared memory.
        del outputs, losses, box_loss, aux_unweighted
        del box_gradient, aux_gradient, s8
        cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

        # Real full-batch forward/backward at the frozen recommended influence.
        criterion.sbox_aux_weight = recommended_weight
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(samples, targets=targets)
        weighted_losses = criterion(outputs, targets, epoch=10)
        total_loss = sum(weighted_losses.values())
        total_loss.backward()
        direction_gradient_norms = []
        final_weight = model.backbone.sbox_head.block[-1].weight.grad
        for direction in range(4):
            direction_gradient_norms.append(float(final_weight[direction].norm()))
        peak_memory_mb = torch.cuda.max_memory_allocated() / 2**20
        all_finite = bool(
            torch.isfinite(total_loss)
            and all(
                torch.isfinite(parameter.grad).all()
                for parameter in sbox_parameters.values()
                if parameter.grad is not None
            )
        )
    finally:
        hook.remove()

    valid_fraction_by_side = [
        float((valid[:, direction] > 0).float().mean()) for direction in range(4)
    ]
    target_mass_mean_by_side = [
        float(target_mass[:, direction].mean()) for direction in range(4)
    ]
    gate = {
        "shared_initialization_exact": shared_initial_error == 0.0,
        "shared_tuning_state_exact": shared_loaded_error == 0.0,
        "eval_output_exact": eval_box_error == 0.0 and eval_logit_error == 0.0,
        "sbox_absent_in_eval": eval_omits_sbox,
        "all_four_directions_have_targets": min(target_mass_mean_by_side) > 0,
        "valid_side_coverage": min(valid_fraction_by_side) >= 0.50,
        "head_learns_correct_over_shifted": float(correct_loss) < 0.90 * float(shifted_loss),
        "optimizer_covers_sbox": optimizer_covers_sbox,
        "all_four_directions_have_gradients": min(direction_gradient_norms) > 0,
        "finite_training_batch": all_finite,
        "training_batch_memory_safe": peak_memory_mb < 15_500,
        "recommended_gradient_ratio_safe": (
            recommended_weight
            * compatibility["raw_aux_over_box_norm_mean"]
            <= target_ratio * 1.01
        ),
    }
    gate["pass"] = all(gate.values())
    report = {
        "protocol": {
            "config": str(args.config),
            "tuning": str(args.tuning),
            "tuning_weight_source": weight_source,
            "batch_size": args.batch_size,
            "head_warmup_steps": args.head_warmup_steps,
            "updates_to_shared_detector": 0,
        },
        "identity": {
            "shared_key_count": len(shared_keys),
            "sbox_only_keys": sbox_only_keys,
            "shared_initial_max_abs_error": shared_initial_error,
            "shared_loaded_max_abs_error": shared_loaded_error,
            "eval_box_max_abs_error": eval_box_error,
            "eval_logit_max_abs_error": eval_logit_error,
            "a00_missing_after_tuning": a00_missing,
            "sbox_missing_after_tuning": sbox_missing,
        },
        "targets_and_head": {
            "direction_order": ["left", "top", "right", "bottom"],
            "initial_loss": initial_loss,
            "valid_fraction_by_side": valid_fraction_by_side,
            "target_mass_mean_by_side": target_mass_mean_by_side,
            "correct_loss_after_head_warmup": float(correct_loss),
            "shifted_loss_after_head_warmup": float(shifted_loss),
            "shifted_over_correct": float(shifted_loss / correct_loss.clamp_min(1e-12)),
            "sbox_parameter_count": sum(p.numel() for p in sbox_parameters.values()),
            "optimizer_covers_sbox": optimizer_covers_sbox,
            "direction_gradient_norms": direction_gradient_norms,
        },
        "gradient_compatibility_at_epoch10": {
            **compatibility,
            "box_loss_keys": box_keys,
            "target_weighted_ratio": target_ratio,
            "recommended_sbox_aux_weight": recommended_weight,
            "weighted_aux_over_box_norm_mean": recommended_weight
            * compatibility["raw_aux_over_box_norm_mean"],
        },
        "training_batch": {
            "batch_size": args.batch_size,
            "total_loss": float(total_loss),
            "loss_sbox": float(weighted_losses["loss_sbox"]),
            "peak_cuda_memory_mb": peak_memory_mb,
            "all_finite": all_finite,
        },
        "entry_gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
