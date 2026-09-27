#!/usr/bin/env python3
"""审计FreeKD官方频率提示能否直接迁移到D-FINE真实S8特征。

严格复刻官方 ``freekd.py`` 的一个容易忽略的细节：DWTForward(J=3)返回三个
分解层级，官方代码将它们依次命名为cH/cV/cD，并在每个层级上对三个方向求和。
本脚本保持这一行为，不把三个提示误解释为三个方向。

脚本只读取冻结A00特征和FreeKD官方提示权重，不更新任何参数。
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
from pytorch_wavelets import DWTForward


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


REGIONS = (
    "sam_boundary",
    "box_boundary",
    "box_only_boundary",
    "shifted_sam_boundary",
    "background",
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
        "--freekd-checkpoint",
        type=Path,
        default=ROOT.parent
        / "_third_party/FreeKD/checkpoints/FreeKD_retinanet_r101_prompt.pth",
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
        / "reports/22_sam_frequency_detail/S_FREQ1_FREEKD_PROMPT_TRANSFER/report.json",
    )
    return parser.parse_args()


def checkpoint_weights(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("ema", {}).get("module") is not None:
        return state["ema"]["module"], "ema.module"
    if "model" in state:
        return state["model"], "model"
    return state, "raw"


def load_a00(config_path, checkpoint_path, device):
    cfg = YAMLConfig(str(config_path))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    # 防止先建立SAM数据配置后，其BPC字段通过可变include缓存泄漏进A00。
    cfg.yaml_cfg["HGNetv2"]["bpc_stage"] = -1
    cfg.yaml_cfg["DFINECriterion"]["bpc_boundary_aux_weight"] = 0.0
    model = cfg.model.to(device).eval()
    weights, source = checkpoint_weights(checkpoint_path)
    model.load_state_dict(weights, strict=True)
    cache = {}

    def capture_s8(_module, inputs):
        cache["s8"] = inputs[0].detach().float()

    hook = model.backbone.stages[2].register_forward_pre_hook(capture_s8)
    return model, cache, hook, source


def load_prompt_tokens(checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    tokens = []
    for module_index in range(5):
        key = f"mask_modules.{module_index}.mask_token"
        if key not in state_dict:
            raise KeyError(f"missing official FreeKD prompt tensor: {key}")
        value = state_dict[key].float()
        if tuple(value.shape) != (4, 6, 256):
            raise ValueError(f"unexpected {key} shape: {tuple(value.shape)}")
        tokens.append(value)
    return torch.stack(tokens, dim=0)


def dilate(x, radius):
    return F.max_pool2d(x, 2 * radius + 1, stride=1, padding=radius)


def erode(x, radius):
    return 1.0 - dilate(1.0 - x, radius)


def box_mask_from_target(target, height, width, device):
    result = torch.zeros((1, 1, height, width), device=device)
    for box in target["boxes"]:
        if float(box.max()) <= 2.0:
            cx, cy, bw, bh = box
            x1, x2 = (cx - bw / 2) * width, (cx + bw / 2) * width
            y1, y2 = (cy - bh / 2) * height, (cy + bh / 2) * height
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
    sam_near = dilate(union, boundary_radius)
    sam_boundary = (sam_near - erode(union, boundary_radius)).clamp(0.0, 1.0)
    box_boundary = (dilate(box, boundary_radius) - erode(box, boundary_radius)).clamp(
        0.0, 1.0
    )
    far = dilate(box.maximum(union), outer_radius)
    regions = {
        "sam_boundary": sam_boundary,
        "box_boundary": box_boundary,
        "box_only_boundary": box_boundary * (1.0 - sam_near),
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
    if numerator is None or denominator is None or denominator <= 1e-12:
        return None
    return numerator / denominator


def official_prompt_response(feature, tokens):
    """返回5个官方模块在三个DWT层级上的提示权重与加权高频能量。"""
    transform = DWTForward(J=3, mode="zero", wave="haar").to(feature.device)
    _low, high_levels = transform(feature)
    output = {}
    for module_index in range(tokens.shape[0]):
        per_level = {}
        for level_index, coefficients in enumerate(high_levels):
            # 官方代码对该层级的三个方向求和，再与对应频率提示做点积。
            frequency_slice = coefficients.sum(dim=2)
            prompt = tokens[module_index, level_index + 1].to(feature.device)
            attention = torch.einsum("tc,bchw->bthw", prompt, frequency_slice).sigmoid()
            weight = attention.mean(dim=1, keepdim=True)
            energy = coefficients.square().mean(dim=(1, 2)).sqrt().unsqueeze(1)
            per_level[level_index] = {
                "prompt_weight": weight,
                "weighted_high_energy": weight * energy,
                "token_spatial_std": attention.flatten(2).std(dim=2).mean(dim=1),
                "token_pair_std": attention.std(dim=1).mean(dim=(-2, -1)),
            }
        output[module_index] = per_level
    return output


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("official prompt transfer audit requires CUDA")
    if args.max_batches <= 0 or args.batch_size <= 0:
        raise ValueError("max-batches and batch-size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    loader_cfg = YAMLConfig(str(args.sam_config))
    loader_cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = args.batch_size
    loader = loader_cfg.train_dataloader
    model, cache, hook, source = load_a00(
        args.a00_config, args.a00_checkpoint, device
    )
    tokens = load_prompt_tokens(args.freekd_checkpoint)

    region_values = defaultdict(lambda: defaultdict(list))
    ratios = defaultdict(list)
    diversity = defaultdict(list)
    valid_images = 0
    processed_batches = 0

    try:
        with torch.inference_mode():
            for batch_index, (samples, targets) in enumerate(loader):
                if batch_index >= args.max_batches:
                    break
                samples = samples.to(device)
                targets = [
                    {
                        key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in target.items()
                    }
                    for target in targets
                ]
                with torch.autocast("cuda", dtype=torch.float16):
                    model.backbone(samples)
                responses = official_prompt_response(cache["s8"], tokens)

                for module_index, levels in responses.items():
                    for level_index, fields in levels.items():
                        key = f"module{module_index}_level{level_index + 1}"
                        diversity[f"{key}/token_spatial_std"].extend(
                            fields["token_spatial_std"].cpu().tolist()
                        )
                        diversity[f"{key}/token_pair_std"].extend(
                            fields["token_pair_std"].cpu().tolist()
                        )
                        for image_index, target in enumerate(targets):
                            regions = build_regions(
                                target,
                                fields["prompt_weight"].shape[-2:],
                                args.boundary_radius,
                                args.outer_radius,
                            )
                            if regions is None:
                                continue
                            values = {}
                            for field_name in ("prompt_weight", "weighted_high_energy"):
                                values[field_name] = {}
                                for region_name, region in regions.items():
                                    value = weighted_mean(
                                        fields[field_name][image_index], region
                                    )
                                    values[field_name][region_name] = value
                                    if value is not None:
                                        region_values[f"{key}/{field_name}"][region_name].append(
                                            value
                                        )
                            prompt_values = values["prompt_weight"]
                            for denominator in (
                                "box_boundary",
                                "box_only_boundary",
                                "shifted_sam_boundary",
                                "background",
                            ):
                                ratio = safe_ratio(
                                    prompt_values["sam_boundary"],
                                    prompt_values[denominator],
                                )
                                if ratio is not None:
                                    ratios[f"{key}/sam_over_{denominator}"].append(ratio)

                valid_images += sum(
                    int(
                        target.get("masks") is not None
                        and target["masks"].shape[0] > 0
                        and bool(target["masks"].any())
                    )
                    for target in targets
                )
                processed_batches += 1
    finally:
        hook.remove()

    rng = np.random.default_rng(args.seed)
    ratio_summary = {
        name: {
            **summarize(values),
            "bootstrap_mean_95ci": bootstrap_mean_ci(values, rng),
        }
        for name, values in sorted(ratios.items())
    }
    candidates = []
    for module_index in range(5):
        for level_index in range(1, 4):
            key = f"module{module_index}_level{level_index}"
            shift = ratio_summary.get(f"{key}/sam_over_shifted_sam_boundary", {})
            box_only = ratio_summary.get(f"{key}/sam_over_box_only_boundary", {})
            spatial = summarize(diversity[f"{key}/token_spatial_std"])
            passed = bool(
                shift.get("mean", 0.0) > 1.05
                and shift.get("bootstrap_mean_95ci", [0.0])[0] > 1.0
                and box_only.get("mean", 0.0) > 1.02
                and box_only.get("bootstrap_mean_95ci", [0.0])[0] > 1.0
                and spatial.get("mean", 0.0) > 0.01
            )
            candidates.append(
                {
                    "key": key,
                    "sam_over_shifted": shift,
                    "sam_over_box_only": box_only,
                    "token_spatial_std": spatial,
                    "pass": passed,
                }
            )

    report = {
        "protocol": {
            "diagnostic": "S-FREQ1 FreeKD官方频率提示迁移审计",
            "a00_config": str(args.a00_config),
            "sam_data_config": str(args.sam_config),
            "a00_checkpoint": str(args.a00_checkpoint),
            "a00_weight_source": source,
            "freekd_checkpoint": str(args.freekd_checkpoint),
            "official_prompt_shape": list(tokens.shape),
            "official_code_semantics": (
                "J=3三个分解层级分别进入mask_token[1:4]；每层先对三个方向求和"
            ),
            "batch_size": args.batch_size,
            "requested_max_batches": args.max_batches,
            "processed_batches": processed_batches,
            "valid_sam_images": valid_images,
            "updates": 0,
        },
        "region_values": {
            name: {
                region: summarize(values)
                for region, values in grouped.items()
            }
            for name, grouped in sorted(region_values.items())
        },
        "ratios": ratio_summary,
        "diversity": {
            name: summarize(values) for name, values in sorted(diversity.items())
        },
        "candidate_gates": candidates,
        "entry_gate": {
            "pass": any(item["pass"] for item in candidates),
            "rule": (
                "至少一个官方模块/层级同时满足SAM相对错位>1.05且CI下界>1、"
                "相对仅框轮廓>1.02且CI下界>1、提示空间标准差>0.01"
            ),
        },
        "interpretation_boundary": (
            "通过只说明官方COCO频率提示可作为初始化；失败时保留FreeKD频率蒸馏公式，"
            "但不得直接迁移官方提示参数。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "protocol": report["protocol"],
                "candidate_gates": candidates,
                "entry_gate": report["entry_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
