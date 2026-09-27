#!/usr/bin/env python3
"""Diagnose whether SAM regions are specifically enriched for localization gradients."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


CLASS_PREFIXES = ("loss_vfl",)
LOCALIZATION_PREFIXES = ("loss_bbox", "loss_giou", "loss_fgl", "loss_ddf")
REGIONS = ("interior", "boundary", "outer_ring", "background", "shifted_boundary")
SIZE_GROUPS = ("lt16", "16to32", "32to48", "ge48")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a00-config",
        type=Path,
        default=ROOT / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument(
        "--bpc-config",
        type=Path,
        default=ROOT
        / "experiments/phase_s/s_bpc1_sam_boundary_polyphase_s8_s16_local.yml",
    )
    parser.add_argument(
        "--a00-checkpoint",
        type=Path,
        default=ROOT.parent
        / "outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth",
    )
    parser.add_argument(
        "--bpc-checkpoint",
        type=Path,
        default=ROOT.parent
        / "runs/20_spatial_importance/S_BPC1_SAM_BOUNDARY_POLYPHASE/seed0/best_stg1.pth",
    )
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--boundary-radius", type=int, default=4)
    parser.add_argument("--outer-radius", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent
        / "reports/20_spatial_importance/S_LTA_D0_SAM_TASK_ALIGNMENT/gradient_regions.json",
    )
    return parser.parse_args()


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def load_component(config_path, checkpoint_path, device, expected_bpc_stage):
    cfg = YAMLConfig(str(config_path))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    # YAML includes are cached as mutable dictionaries in this repository.
    # Loading the BPC config before A00 otherwise leaks bpc_stage=2 into A00.
    cfg.yaml_cfg["HGNetv2"]["bpc_stage"] = int(expected_bpc_stage)
    cfg.yaml_cfg["DFINECriterion"]["bpc_boundary_aux_weight"] = (
        1.0 if expected_bpc_stage >= 0 else 0.0
    )
    model = cfg.model.to(device).train()
    criterion = cfg.criterion.to(device).train()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = state["ema"]["module"] if "ema" in state else state["model"]
    model.load_state_dict(weights, strict=True)
    cache = {}

    def capture_s8(_module, inputs):
        cache["s8"] = inputs[0]

    hook = model.backbone.stages[2].register_forward_pre_hook(capture_s8)
    return model, criterion, cache, hook


def morphology_regions(target, size, boundary_radius, outer_radius, dtype):
    masks = target.get("masks")
    if masks is None or masks.shape[0] == 0 or not bool(masks.any()):
        return None
    masks = masks.float().unsqueeze(1)
    union = masks.amax(dim=0, keepdim=True)

    def dilate(x, radius):
        return F.max_pool2d(x, 2 * radius + 1, stride=1, padding=radius)

    def erode(x, radius):
        return 1.0 - dilate(1.0 - x, radius)

    inner = erode(union, boundary_radius)
    near = dilate(union, boundary_radius)
    far = dilate(union, outer_radius)
    regions = {
        "interior": inner,
        "boundary": (near - inner).clamp(0.0, 1.0),
        "outer_ring": (far - near).clamp(0.0, 1.0),
        "background": (1.0 - far).clamp(0.0, 1.0),
    }
    for name in tuple(regions):
        regions[name] = F.interpolate(regions[name], size=size, mode="area").to(dtype)
    regions["shifted_boundary"] = torch.roll(
        regions["boundary"], shifts=(size[0] // 2, size[1] // 2), dims=(-2, -1)
    )
    return {name: value[0] for name, value in regions.items()}


def weighted_mean(value, weight):
    denominator = weight.sum().clamp_min(1e-12)
    return float((value * weight).sum() / denominator)


def object_size_group(target, image_height, image_width):
    box = target["boxes"][0]
    if float(box.max()) <= 2.0:
        width = float(box[2]) * image_width
        height = float(box[3]) * image_height
    else:
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
    scale = max(width * height, 0.0) ** 0.5
    if scale < 16:
        return "lt16"
    if scale < 32:
        return "16to32"
    if scale < 48:
        return "32to48"
    return "ge48"


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def bootstrap_mean_ci(values, rng, samples=2000):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return [None, None]
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main():
    args = parse_args()
    if args.max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if not 0 < args.boundary_radius < args.outer_radius:
        raise ValueError("require 0 < boundary_radius < outer_radius")
    if not torch.cuda.is_available():
        raise RuntimeError("S-LTA-D0 requires CUDA")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    loader_cfg = YAMLConfig(str(args.bpc_config))
    loader = loader_cfg.train_dataloader
    components = {
        "A00": load_component(args.a00_config, args.a00_checkpoint, device, -1),
        "BPC1": load_component(args.bpc_config, args.bpc_checkpoint, device, 2),
    }
    raw = {
        model_name: {
            loss_name: {region: [] for region in REGIONS}
            for loss_name in ("classification", "localization")
        }
        for model_name in components
    }
    ratios = {
        model_name: {
            loss_name: {
                "boundary_over_background": [],
                "interior_over_background": [],
                "outer_ring_over_background": [],
                "boundary_over_shifted": [],
                "boundary_over_outer_ring": [],
            }
            for loss_name in ("classification", "localization")
        }
        for model_name in components
    }
    size_ratios = {
        model_name: {
            loss_name: {
                size_group: {
                    "boundary_over_background": [],
                    "boundary_over_shifted": [],
                    "boundary_over_outer_ring": [],
                }
                for size_group in SIZE_GROUPS
            }
            for loss_name in ("classification", "localization")
        }
        for model_name in components
    }
    recorded_loss_keys = {}
    processed_batches = 0
    valid_images = 0

    try:
        for batch_index, (samples, targets) in enumerate(loader):
            if batch_index >= args.max_batches:
                break
            samples = samples.to(device)
            targets = move_targets(targets, device)
            valid_in_batch = sum(
                int(target.get("masks") is not None and target["masks"].shape[0] > 0 and bool(target["masks"].any()))
                for target in targets
            )
            valid_images += valid_in_batch
            if valid_in_batch == 0:
                continue

            for model_offset, (model_name, component) in enumerate(components.items()):
                model, criterion, cache, _hook = component
                torch.manual_seed(args.seed + batch_index)
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = model(samples, targets=targets)
                with torch.autocast("cuda", enabled=False):
                    losses = criterion(outputs, targets)
                class_keys = [key for key in losses if key.startswith(CLASS_PREFIXES)]
                loc_keys = [key for key in losses if key.startswith(LOCALIZATION_PREFIXES)]
                if not class_keys or not loc_keys:
                    raise RuntimeError(
                        f"missing diagnostic losses for {model_name}: "
                        f"classification={class_keys}, localization={loc_keys}"
                    )
                recorded_loss_keys[model_name] = {
                    "classification": class_keys,
                    "localization": loc_keys,
                }
                s8 = cache["s8"]
                class_loss = sum(losses[key] for key in class_keys)
                loc_loss = sum(losses[key] for key in loc_keys)
                class_grad = torch.autograd.grad(class_loss, s8, retain_graph=True)[0]
                loc_grad = torch.autograd.grad(loc_loss, s8, retain_graph=False)[0]
                gradient_maps = {
                    "classification": class_grad.float().square().mean(dim=1, keepdim=True).sqrt(),
                    "localization": loc_grad.float().square().mean(dim=1, keepdim=True).sqrt(),
                }

                for image_index, target in enumerate(targets):
                    regions = morphology_regions(
                        target,
                        s8.shape[-2:],
                        args.boundary_radius,
                        args.outer_radius,
                        gradient_maps["classification"].dtype,
                    )
                    if regions is None:
                        continue
                    size_group = object_size_group(
                        target, samples.shape[-2], samples.shape[-1]
                    )
                    for loss_name, gradient_map in gradient_maps.items():
                        image_map = gradient_map[image_index]
                        region_means = {
                            region: weighted_mean(image_map, weight)
                            for region, weight in regions.items()
                        }
                        for region, value in region_means.items():
                            raw[model_name][loss_name][region].append(value)
                        background = max(region_means["background"], 1e-12)
                        shifted = max(region_means["shifted_boundary"], 1e-12)
                        outer_ring = max(region_means["outer_ring"], 1e-12)
                        ratios[model_name][loss_name]["boundary_over_background"].append(
                            region_means["boundary"] / background
                        )
                        ratios[model_name][loss_name]["interior_over_background"].append(
                            region_means["interior"] / background
                        )
                        ratios[model_name][loss_name]["outer_ring_over_background"].append(
                            region_means["outer_ring"] / background
                        )
                        ratios[model_name][loss_name]["boundary_over_shifted"].append(
                            region_means["boundary"] / shifted
                        )
                        ratios[model_name][loss_name]["boundary_over_outer_ring"].append(
                            region_means["boundary"] / outer_ring
                        )
                        size_ratios[model_name][loss_name][size_group][
                            "boundary_over_background"
                        ].append(region_means["boundary"] / background)
                        size_ratios[model_name][loss_name][size_group][
                            "boundary_over_shifted"
                        ].append(region_means["boundary"] / shifted)
                        size_ratios[model_name][loss_name][size_group][
                            "boundary_over_outer_ring"
                        ].append(region_means["boundary"] / outer_ring)

                del outputs, losses, class_loss, loc_loss, class_grad, loc_grad, gradient_maps
            processed_batches += 1
            del samples, targets
    finally:
        for model, criterion, cache, hook in components.values():
            hook.remove()

    rng = np.random.default_rng(args.seed)
    summary = {}
    for model_name in components:
        summary[model_name] = {}
        for loss_name in ("classification", "localization"):
            summary[model_name][loss_name] = {
                "region_gradient_rms": {
                    region: summarize(raw[model_name][loss_name][region]) for region in REGIONS
                },
                "enrichment_ratios": {
                    name: {
                        **summarize(values),
                        "bootstrap_mean_95ci": bootstrap_mean_ci(values, rng),
                    }
                    for name, values in ratios[model_name][loss_name].items()
                },
                "size_group_enrichment": {
                    size_group: {
                        name: summarize(values)
                        for name, values in size_ratios[model_name][loss_name][
                            size_group
                        ].items()
                    }
                    for size_group in SIZE_GROUPS
                },
            }
        loc_boundary = ratios[model_name]["localization"]["boundary_over_background"]
        cls_boundary = ratios[model_name]["classification"]["boundary_over_background"]
        loc_shift = ratios[model_name]["localization"]["boundary_over_shifted"]
        cls_shift = ratios[model_name]["classification"]["boundary_over_shifted"]
        loc_outer = ratios[model_name]["localization"]["boundary_over_outer_ring"]
        cls_outer = ratios[model_name]["classification"]["boundary_over_outer_ring"]
        summary[model_name]["task_specificity"] = {
            "boundary_background_localization_over_classification": summarize(
                [loc / max(cls, 1e-12) for loc, cls in zip(loc_boundary, cls_boundary)]
            ),
            "boundary_shifted_localization_over_classification": summarize(
                [loc / max(cls, 1e-12) for loc, cls in zip(loc_shift, cls_shift)]
            ),
            "boundary_outer_ring_localization_over_classification": summarize(
                [loc / max(cls, 1e-12) for loc, cls in zip(loc_outer, cls_outer)]
            ),
        }

    report = {
        "protocol": {
            "a00_config": str(args.a00_config),
            "bpc_config_and_mask_loader": str(args.bpc_config),
            "a00_checkpoint": str(args.a00_checkpoint),
            "bpc_checkpoint": str(args.bpc_checkpoint),
            "seed": args.seed,
            "requested_max_batches": args.max_batches,
            "processed_batches": processed_batches,
            "valid_sam_images": valid_images,
            "boundary_radius_input_pixels": args.boundary_radius,
            "outer_radius_input_pixels": args.outer_radius,
            "gradient_map": "channel RMS of d(loss_group)/d(S8)",
            "classification_prefixes": CLASS_PREFIXES,
            "localization_prefixes": LOCALIZATION_PREFIXES,
            "size_groups_sqrt_area_input_pixels": SIZE_GROUPS,
            "recorded_loss_keys": recorded_loss_keys,
            "model_mode": "train for faithful detector losses; no optimizer/backward update",
        },
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
