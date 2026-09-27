#!/usr/bin/env python3
"""零训练诊断：SAM轮廓是否富集定位任务真正需要的高频信息。

本脚本不更新模型。它在真实S8->S16接口截取S8特征，将S8特征和检测损失梯度
同时做一级Haar分解，并用 |频率系数 * 对应梯度系数| 作为一阶任务价值。
比较区域包括正确SAM轮廓、框轮廓、仅框轮廓、错位SAM轮廓和背景。

这个诊断用于阻止重复S-RES1/BPC1：只有正确SAM轮廓在定位高频价值上具有独占优势，
才允许进入SAM引导的频率蒸馏；它不会因为单纯高频能量较大而放行。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


CLASS_PREFIXES = ("loss_vfl",)
LOCALIZATION_PREFIXES = ("loss_bbox", "loss_giou", "loss_fgl", "loss_ddf")
REGIONS = (
    "sam_boundary",
    "box_boundary",
    "box_only_boundary",
    "sam_interior",
    "shifted_sam_boundary",
    "background",
)
FIELDS = (
    "high_energy",
    "high_grad_localization",
    "high_value_localization",
    "high_value_classification",
    "low_value_localization",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a00-config",
        type=Path,
        default=ROOT / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument(
        "--sam-config",
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
        "--rep-config",
        type=Path,
        default=ROOT
        / "experiments/phase_s/s_rep1_1_restart30_spar_w025_lr02_local.yml",
    )
    parser.add_argument(
        "--rep-checkpoint",
        type=Path,
        default=ROOT.parent
        / "runs/21_public_reproduction/S_REP1_1_RESTART30_SPAR_W025_LR02/seed0/best_stg1.pth",
    )
    parser.add_argument("--max-batches", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--boundary-radius", type=int, default=4)
    parser.add_argument("--outer-radius", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent
        / "reports/22_sam_frequency_detail/S_FREQ0_SAM_FREQUENCY_VALUE/report.json",
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


def checkpoint_weights(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("ema", {}).get("module") is not None:
        return state["ema"]["module"], "ema.module"
    if "model" in state:
        return state["model"], "model"
    return state, "raw"


def load_component(config_path, checkpoint_path, device, is_rep):
    cfg = YAMLConfig(str(config_path))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    # 本仓库的YAML include缓存会复用可变字典。SAM数据加载配置含BPC1，
    # 若不在每次构造前显式复位，会把bpc_stage=2泄漏进A00/REP1.1。
    cfg.yaml_cfg["HGNetv2"]["bpc_stage"] = -1
    cfg.yaml_cfg["DFINECriterion"]["bpc_boundary_aux_weight"] = 0.0
    if is_rep:
        # 保留REP1.1训练期SPAR参数以便严格加载，但关闭其损失；本诊断只测检测梯度。
        cfg.yaml_cfg["DFINECriterion"]["spar_aux_weight"] = 0.0
    model = cfg.model.to(device).train()
    criterion = cfg.criterion.to(device).train()
    weights, source = checkpoint_weights(checkpoint_path)
    model.load_state_dict(weights, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    cache = {}

    def capture_s8(_module, inputs):
        # 模型参数冻结，但输入图需要梯度，才能得到检测损失对S8的梯度。
        cache["s8"] = inputs[0]
        cache["s8"].retain_grad()

    hook = model.backbone.stages[2].register_forward_pre_hook(capture_s8)
    return model, criterion, cache, hook, source


def pad_even(x):
    pad_h = x.shape[-2] % 2
    pad_w = x.shape[-1] % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x


def haar_one_level(x):
    """正交一级Haar，返回LL和三个方向高频，空间尺寸减半。"""
    x = pad_even(x)
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, torch.stack((lh, hl, hh), dim=2)


def dilate(x, radius):
    return F.max_pool2d(x, 2 * radius + 1, stride=1, padding=radius)


def erode(x, radius):
    return 1.0 - dilate(1.0 - x, radius)


def box_mask_from_target(target, height, width, device):
    result = torch.zeros((1, 1, height, width), device=device)
    boxes = target["boxes"]
    for box in boxes:
        if float(box.max()) <= 2.0:
            cx, cy, bw, bh = box
            x1 = (cx - bw / 2) * width
            x2 = (cx + bw / 2) * width
            y1 = (cy - bh / 2) * height
            y2 = (cy + bh / 2) * height
        else:
            x1, y1, x2, y2 = box
        ix1 = max(0, min(width - 1, int(torch.floor(x1).item())))
        iy1 = max(0, min(height - 1, int(torch.floor(y1).item())))
        ix2 = max(ix1 + 1, min(width, int(torch.ceil(x2).item())))
        iy2 = max(iy1 + 1, min(height, int(torch.ceil(y2).item())))
        result[..., iy1:iy2, ix1:ix2] = 1.0
    return result


def build_regions(target, output_size, boundary_radius, outer_radius):
    masks = target.get("masks")
    if masks is None or masks.shape[0] == 0 or not bool(masks.any()):
        return None
    union = masks.float().unsqueeze(1).amax(dim=0, keepdim=True)
    height, width = union.shape[-2:]
    box = box_mask_from_target(target, height, width, union.device)
    sam_inner = erode(union, boundary_radius)
    sam_near = dilate(union, boundary_radius)
    sam_boundary = (sam_near - sam_inner).clamp(0.0, 1.0)
    box_boundary = (dilate(box, boundary_radius) - erode(box, boundary_radius)).clamp(
        0.0, 1.0
    )
    far = dilate(box.maximum(union), outer_radius)
    regions = {
        "sam_boundary": sam_boundary,
        "box_boundary": box_boundary,
        "box_only_boundary": box_boundary * (1.0 - sam_near),
        "sam_interior": sam_inner,
        "shifted_sam_boundary": torch.roll(
            sam_boundary, shifts=(height // 2, width // 2), dims=(-2, -1)
        ),
        "background": 1.0 - far,
    }
    return {
        name: F.interpolate(value, size=output_size, mode="area")[0]
        for name, value in regions.items()
    }


def weighted_mean(value, weight):
    denominator = weight.sum()
    if float(denominator) <= 1e-8:
        return None
    return float((value * weight).sum() / denominator)


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


def safe_ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 1e-20:
        return None
    return numerator / denominator


def main():
    args = parse_args()
    if args.max_batches <= 0 or args.batch_size < 1:
        raise ValueError("max-batches and batch-size must be positive")
    if not 0 < args.boundary_radius < args.outer_radius:
        raise ValueError("require 0 < boundary-radius < outer-radius")
    if not torch.cuda.is_available():
        raise RuntimeError("S-FREQ0 requires CUDA")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    loader_cfg = YAMLConfig(str(args.sam_config))
    loader_cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = args.batch_size
    loader = loader_cfg.train_dataloader
    components = {
        "A00": load_component(
            args.a00_config, args.a00_checkpoint, device, is_rep=False
        ),
        "REP1.1": load_component(
            args.rep_config, args.rep_checkpoint, device, is_rep=True
        ),
    }

    raw = {
        model_name: {field: defaultdict(list) for field in FIELDS}
        for model_name in components
    }
    ratios = {
        model_name: defaultdict(list)
        for model_name in components
    }
    recorded_loss_keys = {}
    valid_images = 0
    processed_batches = 0

    try:
        for batch_index, (samples, targets) in enumerate(loader):
            if batch_index >= args.max_batches:
                break
            samples = samples.to(device).requires_grad_(True)
            targets = move_targets(targets, device)
            valid_in_batch = sum(
                int(
                    target.get("masks") is not None
                    and target["masks"].shape[0] > 0
                    and bool(target["masks"].any())
                )
                for target in targets
            )
            valid_images += valid_in_batch
            if valid_in_batch == 0:
                continue

            for model_offset, (model_name, component) in enumerate(components.items()):
                model, criterion, cache, _hook, _source = component
                model.zero_grad(set_to_none=True)
                torch.manual_seed(args.seed + batch_index)
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = model(samples, targets=targets)
                with torch.autocast("cuda", enabled=False):
                    losses = criterion(outputs, targets)
                class_keys = [key for key in losses if key.startswith(CLASS_PREFIXES)]
                loc_keys = [key for key in losses if key.startswith(LOCALIZATION_PREFIXES)]
                if not class_keys or not loc_keys:
                    raise RuntimeError(
                        f"missing losses for {model_name}: class={class_keys}, loc={loc_keys}"
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

                ll, high = haar_one_level(s8.float())
                ll_loc_grad, high_loc_grad = haar_one_level(loc_grad.float())
                _, high_cls_grad = haar_one_level(class_grad.float())
                fields = {
                    "high_energy": high.square().mean(dim=(1, 2)).sqrt().unsqueeze(1),
                    "high_grad_localization": high_loc_grad.square()
                    .mean(dim=(1, 2))
                    .sqrt()
                    .unsqueeze(1),
                    "high_value_localization": (high * high_loc_grad)
                    .abs()
                    .mean(dim=(1, 2))
                    .unsqueeze(1),
                    "high_value_classification": (high * high_cls_grad)
                    .abs()
                    .mean(dim=(1, 2))
                    .unsqueeze(1),
                    "low_value_localization": (ll * ll_loc_grad)
                    .abs()
                    .mean(dim=1, keepdim=True),
                }

                for image_index, target in enumerate(targets):
                    regions = build_regions(
                        target,
                        high.shape[-2:],
                        args.boundary_radius,
                        args.outer_radius,
                    )
                    if regions is None:
                        continue
                    values = {}
                    for field_name, field in fields.items():
                        values[field_name] = {}
                        for region_name, region in regions.items():
                            value = weighted_mean(field[image_index], region)
                            values[field_name][region_name] = value
                            if value is not None:
                                raw[model_name][field_name][region_name].append(value)

                    loc = values["high_value_localization"]
                    cls = values["high_value_classification"]
                    for denominator in (
                        "box_boundary",
                        "box_only_boundary",
                        "shifted_sam_boundary",
                        "background",
                    ):
                        ratio = safe_ratio(loc["sam_boundary"], loc[denominator])
                        if ratio is not None:
                            ratios[model_name][
                                f"localization_sam_over_{denominator}"
                            ].append(ratio)
                    task_ratio = safe_ratio(
                        safe_ratio(loc["sam_boundary"], loc["shifted_sam_boundary"]),
                        safe_ratio(cls["sam_boundary"], cls["shifted_sam_boundary"]),
                    )
                    if task_ratio is not None:
                        ratios[model_name][
                            "sam_shift_localization_over_classification"
                        ].append(task_ratio)
                    freq_ratio = safe_ratio(
                        loc["sam_boundary"],
                        values["low_value_localization"]["sam_boundary"],
                    )
                    if freq_ratio is not None:
                        ratios[model_name]["sam_high_over_low_localization"].append(
                            freq_ratio
                        )

                del outputs, losses, class_loss, loc_loss, class_grad, loc_grad
                del ll, high, ll_loc_grad, high_loc_grad, high_cls_grad, fields
            processed_batches += 1
            del samples, targets

    finally:
        for model, criterion, cache, hook, source in components.values():
            hook.remove()

    rng = np.random.default_rng(args.seed)
    summary = {}
    for model_name in components:
        summary[model_name] = {
            "region_values": {
                field: {
                    region: summarize(raw[model_name][field][region])
                    for region in REGIONS
                }
                for field in FIELDS
            },
            "ratios": {
                name: {
                    **summarize(values),
                    "bootstrap_mean_95ci": bootstrap_mean_ci(values, rng),
                }
                for name, values in sorted(ratios[model_name].items())
            },
        }

    a00_ratios = summary["A00"]["ratios"]
    gate = {
        "sam_beats_shifted_for_localization_high_value": bool(
            a00_ratios.get("localization_sam_over_shifted_sam_boundary", {}).get(
                "mean", 0.0
            )
            > 1.25
            and a00_ratios.get(
                "localization_sam_over_shifted_sam_boundary", {}
            ).get("bootstrap_mean_95ci", [0.0])[0]
            > 1.0
        ),
        "sam_beats_background_for_localization_high_value": bool(
            a00_ratios.get("localization_sam_over_background", {}).get("mean", 0.0)
            > 1.25
            and a00_ratios.get("localization_sam_over_background", {}).get(
                "bootstrap_mean_95ci", [0.0]
            )[0]
            > 1.0
        ),
        "sam_adds_beyond_box_boundary": bool(
            a00_ratios.get("localization_sam_over_box_boundary", {}).get("mean", 0.0)
            > 1.05
        ),
    }
    gate["pass"] = all(gate.values())

    report = {
        "protocol": {
            "diagnostic": "S-FREQ0 SAM定位高频价值诊断",
            "a00_config": str(args.a00_config),
            "sam_data_config": str(args.sam_config),
            "rep_config": str(args.rep_config),
            "a00_checkpoint": str(args.a00_checkpoint),
            "rep_checkpoint": str(args.rep_checkpoint),
            "checkpoint_sources": {
                name: component[-1] for name, component in components.items()
            },
            "batch_size": args.batch_size,
            "requested_max_batches": args.max_batches,
            "processed_batches": processed_batches,
            "valid_sam_images": valid_images,
            "boundary_radius_input_pixels": args.boundary_radius,
            "outer_radius_input_pixels": args.outer_radius,
            "frequency_value": "mean(abs(Haar(feature) * Haar(detection_loss_gradient)))",
            "recorded_loss_keys": recorded_loss_keys,
            "updates": 0,
        },
        "summary": summary,
        "entry_gate": gate,
        "interpretation_boundary": (
            "通过仅允许进入频率蒸馏最小实现，不证明模块会提高AP；"
            "失败则关闭SAM频率路线，不搜索Haar方向、层数或门控权重。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
