#!/usr/bin/env python3
"""S-AFF0：零训练检查 SAM 空间关系是否值得蒸馏到 D-FINE。

本脚本只读取 A00、SAM2 特征和已人工核验的 SAM 掩码，不更新任何参数。
它回答两个问题：
1. SAM 的局部关系突变是否与检测定位梯度对齐，并优于框边缘和近邻错位掩码；
2. 这种关系与 D-FINE 的一致性是否从真实 S8 到 S16 明显下降。
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
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


LOCALIZATION_PREFIXES = ("loss_bbox", "loss_giou", "loss_fgl", "loss_ddf")
REGIONS = ("sam_boundary", "box_boundary", "shifted_sam_boundary", "sam_body")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a00-config",
        type=Path,
        default=ROOT / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument(
        "--a00-checkpoint",
        type=Path,
        default=WORKSPACE / "outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth",
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=WORKSPACE
        / "reports/20_spatial_importance/S_DIAG3_TRUE_CONTOUR/masks_val",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=WORKSPACE / "data/antiuav6k_common/images/val",
    )
    parser.add_argument(
        "--sam2-repo", type=Path, default=WORKSPACE / "_third_party/sam2"
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=WORKSPACE / "weights/sam2/sam2.1_hiera_b+.pt",
    )
    parser.add_argument(
        "--sam2-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml"
    )
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument(
        "--selection",
        choices=("manual_audit", "heldout_accepted"),
        default="manual_audit",
    )
    parser.add_argument("--max-box-area", type=float, default=None)
    parser.add_argument("--boundary-radius", type=int, default=4)
    parser.add_argument("--roi-radius", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE
        / "reports/23_sam_spatial_relation/S_AFF0_ZERO_TRAIN/report.json",
    )
    return parser.parse_args()


def checkpoint_weights(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("ema", {}).get("module") is not None:
        return state["ema"]["module"], "ema.module"
    if "model" in state:
        return state["model"], "model"
    return state, "raw"


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def dilate(x, radius):
    return F.max_pool2d(x, 2 * radius + 1, stride=1, padding=radius)


def erode(x, radius):
    return 1.0 - dilate(1.0 - x, radius)


def translate_no_wrap(x, dx, dy=0):
    result = torch.zeros_like(x)
    height, width = x.shape[-2:]
    src_x1, src_x2 = max(0, -dx), min(width, width - dx)
    src_y1, src_y2 = max(0, -dy), min(height, height - dy)
    dst_x1, dst_x2 = max(0, dx), min(width, width + dx)
    dst_y1, dst_y2 = max(0, dy), min(height, height + dy)
    if src_x2 > src_x1 and src_y2 > src_y1:
        result[..., dst_y1:dst_y2, dst_x1:dst_x2] = x[
            ..., src_y1:src_y2, src_x1:src_x2
        ]
    return result


def build_regions(record, mask_path, output_size, boundary_radius, roi_radius, device):
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 127
    union = torch.from_numpy(mask.copy()).to(device=device, dtype=torch.float32)[
        None, None
    ]
    height, width = mask.shape
    x1, y1, x2, y2 = [float(value) for value in record["bbox_xyxy"]]
    ix1, iy1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    ix2, iy2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
    box = torch.zeros_like(union)
    box[..., iy1:iy2, ix1:ix2] = 1.0

    sam_boundary = (dilate(union, boundary_radius) - erode(union, boundary_radius)).clamp(0, 1)
    box_boundary = (dilate(box, boundary_radius) - erode(box, boundary_radius)).clamp(0, 1)
    object_width = max(1.0, x2 - x1)
    shift = int(round(max(16.0, 1.5 * object_width)))
    center_x = 0.5 * (x1 + x2)
    dx = shift if center_x < width / 2 else -shift
    shifted = translate_no_wrap(sam_boundary, dx=dx)
    roi = dilate(torch.maximum(union, box), roi_radius)

    raw = {
        "sam_boundary": sam_boundary,
        "box_boundary": box_boundary,
        "shifted_sam_boundary": shifted,
        "sam_body": union,
        "roi": roi,
    }
    return {
        name: F.interpolate(value, size=output_size, mode="area")
        for name, value in raw.items()
    }


def local_relation_contrast(feature):
    """每个位置与四邻域的平均 1-cos，相当于稀疏版空间亲和关系。"""
    feature = F.normalize(feature.float(), dim=1, eps=1e-6)
    total = torch.zeros_like(feature[:, :1])
    count = torch.zeros_like(total)

    vertical = 1.0 - (feature[:, :, 1:] * feature[:, :, :-1]).sum(1, keepdim=True)
    total[:, :, 1:] += vertical
    total[:, :, :-1] += vertical
    count[:, :, 1:] += 1
    count[:, :, :-1] += 1

    horizontal = 1.0 - (feature[:, :, :, 1:] * feature[:, :, :, :-1]).sum(1, keepdim=True)
    total[:, :, :, 1:] += horizontal
    total[:, :, :, :-1] += horizontal
    count[:, :, :, 1:] += 1
    count[:, :, :, :-1] += 1
    return total / count.clamp_min(1.0)


def weighted_mean(value, weight):
    denominator = weight.sum()
    if float(denominator) <= 1e-8:
        return None
    return float((value * weight).sum() / denominator)


def weighted_corr(x, y, weight):
    denominator = weight.sum()
    if float(denominator) <= 1e-8:
        return None
    mx = (x * weight).sum() / denominator
    my = (y * weight).sum() / denominator
    xc, yc = x - mx, y - my
    covariance = (xc * yc * weight).sum()
    variance = ((xc.square() * weight).sum() * (yc.square() * weight).sum()).sqrt()
    if float(variance) <= 1e-12:
        return None
    return float(covariance / variance)


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


def bootstrap_mean_ci(values, rng, samples=3000):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return [None, None]
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_summary(left, right, rng):
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left_array) & np.isfinite(right_array)
    differences = left_array[valid] - right_array[valid]
    ratios = left_array[valid] / np.maximum(right_array[valid], 1e-12)
    return {
        "count": int(valid.sum()),
        "left": summarize(left_array[valid]),
        "right": summarize(right_array[valid]),
        "difference": {
            **summarize(differences),
            "bootstrap_mean_95ci": bootstrap_mean_ci(differences, rng),
            "positive_fraction": float((differences > 0).mean()) if differences.size else None,
        },
        "ratio": summarize(ratios),
    }


def audit_records(audit_root, maximum, selection, max_box_area):
    audited_ids = {
        int(path.stem) for path in (audit_root / "audit_overlays").glob("*.webp")
    }
    records = json.loads((audit_root / "records.json").read_text(encoding="utf-8"))
    def box_area(record):
        x1, y1, x2, y2 = record["bbox_xyxy"]
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    if selection == "manual_audit":
        # audit_overlays 中的 200 张已经由用户人工核验；人工结论覆盖旧自动阈值。
        candidates = [
            record for record in records if int(record["image_id"]) in audited_ids
        ]
    else:
        candidates = [
            record
            for record in records
            if int(record["image_id"]) not in audited_ids
            and bool(record.get("accepted"))
        ]
    if max_box_area is not None:
        candidates = [record for record in candidates if box_area(record) <= max_box_area]
    selected = {int(record["image_id"]): record for record in candidates}
    available_count = len(selected)
    automatic_accepted = sum(bool(record.get("accepted")) for record in selected.values())
    return (
        dict(list(sorted(selected.items()))[:maximum]),
        len(audited_ids),
        automatic_accepted,
        available_count,
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("S-AFF0 需要 CUDA")
    if args.max_samples < 1:
        raise ValueError("max-samples 必须大于 0")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    selected, audited_count, automatic_accepted, available_count = audit_records(
        args.audit_root, args.max_samples, args.selection, args.max_box_area
    )
    if not selected:
        raise RuntimeError("没有找到通过人工核验的样本")

    cfg = YAMLConfig(str(args.a00_config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = 1
    loader = cfg.val_dataloader
    model = cfg.model.to(device).train()
    criterion = cfg.criterion.to(device).train()
    weights, checkpoint_source = checkpoint_weights(args.a00_checkpoint)
    model.load_state_dict(weights, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    feature_cache = {}

    def capture_s8(_module, inputs):
        feature_cache["s8"] = inputs[0]
        feature_cache["s8"].retain_grad()

    def capture_s16(_module, _inputs, output):
        feature_cache["s16"] = output
        feature_cache["s16"].retain_grad()

    pre_hook = model.backbone.stages[2].register_forward_pre_hook(capture_s8)
    post_hook = model.backbone.stages[2].register_forward_hook(capture_s16)

    sys.path.insert(0, str(args.sam2_repo))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam_model = build_sam2(
        args.sam2_config, str(args.sam2_checkpoint), device="cuda"
    ).eval()
    sam_predictor = SAM2ImagePredictor(sam_model)

    raw = defaultdict(list)
    samples_report = []
    processed_ids = set()
    peak_memory = 0.0

    try:
        for samples, targets in loader:
            image_id = int(targets[0]["image_id"].item())
            if image_id not in selected:
                continue
            record = selected[image_id]
            image_path = args.image_root / record["file_name"]
            mask_path = Path(record["mask_path"])
            if not mask_path.exists():
                mask_path = args.audit_root / "masks" / f"{image_id:06d}.png"

            image = np.asarray(Image.open(image_path).convert("RGB")).copy()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                sam_predictor.set_image(image)
                # SAM2 的第二个高分辨率特征是编码器 /8 层，空间细节与语义最平衡。
                sam_feature = sam_predictor._features["high_res_feats"][-1].float()

            samples = samples.to(device).requires_grad_(True)
            targets = move_targets(targets, device)
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(samples, targets=targets)
            with torch.autocast("cuda", enabled=False):
                losses = criterion(outputs, targets)
            location_keys = [
                key for key in losses if key.startswith(LOCALIZATION_PREFIXES)
            ]
            if not location_keys:
                raise RuntimeError("没有找到定位损失")
            location_loss = sum(losses[key] for key in location_keys)
            s8, s16 = feature_cache["s8"], feature_cache["s16"]
            grad_s8, grad_s16 = torch.autograd.grad(
                location_loss, (s8, s16), retain_graph=False, allow_unused=False
            )

            sam_s8 = F.interpolate(sam_feature, size=s8.shape[-2:], mode="bilinear", align_corners=False)
            sam_s16 = F.interpolate(sam_feature, size=s16.shape[-2:], mode="bilinear", align_corners=False)
            relation_sam_s8 = local_relation_contrast(sam_s8)
            relation_sam_s16 = local_relation_contrast(sam_s16)
            relation_dfine_s8 = local_relation_contrast(s8)
            relation_dfine_s16 = local_relation_contrast(s16)
            # 主比较必须在同一 S8 网格进行，否则 S16 的低分辨率平滑会虚增相关性。
            s16_on_s8 = F.interpolate(
                s16.float(), size=s8.shape[-2:], mode="bilinear", align_corners=False
            )
            relation_dfine_s16_on_s8 = local_relation_contrast(s16_on_s8)
            gradient_s8 = grad_s8.float().square().mean(1, keepdim=True).sqrt()
            joint_value = relation_sam_s8 * gradient_s8

            regions_s8 = build_regions(
                record,
                mask_path,
                s8.shape[-2:],
                args.boundary_radius,
                args.roi_radius,
                device,
            )
            regions_s16 = build_regions(
                record,
                mask_path,
                s16.shape[-2:],
                args.boundary_radius,
                args.roi_radius,
                device,
            )

            sample_result = {"image_id": image_id, "file_name": record["file_name"]}
            for region in REGIONS:
                joint = weighted_mean(joint_value, regions_s8[region])
                teacher = weighted_mean(relation_sam_s8, regions_s8[region])
                gradient = weighted_mean(gradient_s8, regions_s8[region])
                sample_result[f"joint_{region}"] = joint
                sample_result[f"teacher_{region}"] = teacher
                sample_result[f"gradient_{region}"] = gradient
                raw[f"joint_{region}"].append(joint)
                raw[f"teacher_{region}"].append(teacher)
                raw[f"gradient_{region}"].append(gradient)

            corr_s8 = weighted_corr(
                relation_sam_s8, relation_dfine_s8, regions_s8["roi"]
            )
            corr_s16 = weighted_corr(
                relation_sam_s8, relation_dfine_s16_on_s8, regions_s8["roi"]
            )
            corr_s16_native = weighted_corr(
                relation_sam_s16, relation_dfine_s16, regions_s16["roi"]
            )
            sample_result["relation_corr_s8"] = corr_s8
            sample_result["relation_corr_s16"] = corr_s16
            sample_result["relation_corr_s16_native_resolution"] = corr_s16_native
            raw["relation_corr_s8"].append(corr_s8)
            raw["relation_corr_s16"].append(corr_s16)
            raw["relation_corr_s16_native_resolution"].append(corr_s16_native)
            samples_report.append(sample_result)
            processed_ids.add(image_id)
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 2**20)

            sam_predictor.reset_predictor()
            del sam_feature, sam_s8, sam_s16, outputs, losses, location_loss
            del s8, s16, grad_s8, grad_s16, samples, targets
            if len(processed_ids) >= len(selected):
                break
    finally:
        pre_hook.remove()
        post_hook.remove()

    rng = np.random.default_rng(args.seed)
    correct_shift = paired_summary(
        raw["joint_sam_boundary"], raw["joint_shifted_sam_boundary"], rng
    )
    correct_box = paired_summary(
        raw["joint_sam_boundary"], raw["joint_box_boundary"], rng
    )
    s8_s16 = paired_summary(raw["relation_corr_s8"], raw["relation_corr_s16"], rng)

    gate = {
        "correct_sam_beats_near_shift": bool(
            correct_shift["ratio"]["mean"] > 1.10
            and correct_shift["difference"]["bootstrap_mean_95ci"][0] > 0
        ),
        "correct_sam_adds_beyond_box": bool(
            correct_box["ratio"]["mean"] > 1.02
            and correct_box["difference"]["positive_fraction"] > 0.55
        ),
        "sam_relation_declines_s8_to_s16": bool(
            s8_s16["difference"]["mean"] > 0.02
            and s8_s16["difference"]["bootstrap_mean_95ci"][0] > 0
        ),
    }
    gate["pass"] = all(gate.values())

    report = {
        "protocol": {
            "diagnostic": "S-AFF0 SAM空间关系零训练诊断",
            "updates": 0,
            "a00_config": str(args.a00_config),
            "a00_checkpoint": str(args.a00_checkpoint),
            "checkpoint_source": checkpoint_source,
            "sam2_checkpoint": str(args.sam2_checkpoint),
            "sam_teacher_feature": "SAM2.1 Hiera B+ high_res_feats[-1] (/8, 64 channels)",
            "relation": "四邻域平均(1-cos)，是SAMFeat全局亲和矩阵的边缘聚焦稀疏版本",
            "scale_comparison": "将S16双线性还原到S8网格后计算关系，两层均与同一张SAM-S8关系图比较",
            "audited_overlay_count": audited_count,
            "selection": args.selection,
            "max_box_area": args.max_box_area,
            "available_after_selection": available_count,
            "selected_count": len(selected),
            "automatic_accepted_within_selection": automatic_accepted,
            "processed_count": len(processed_ids),
            "processed_image_ids": sorted(processed_ids),
            "boundary_radius_input_pixels": args.boundary_radius,
            "near_shift": "沿水平方向平移max(16px, 1.5倍框宽)，不循环回绕",
            "roi_radius_input_pixels": args.roi_radius,
            "localization_loss_keys": location_keys if processed_ids else [],
            "peak_cuda_memory_mb": peak_memory,
        },
        "summary": {key: summarize(values) for key, values in sorted(raw.items())},
        "paired_tests": {
            "correct_sam_boundary_vs_near_shift": correct_shift,
            "correct_sam_boundary_vs_box_boundary": correct_box,
            "sam_dfine_relation_agreement_s8_vs_s16": s8_s16,
        },
        "entry_gate": gate,
        "interpretation_boundary": (
            "通过只允许实现一个SAMFeat式关系蒸馏最小版本，不证明训练一定增益；"
            "失败则不进入训练，也不搜索权重、层数或训练轮数。"
        ),
        "samples": samples_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **report["protocol"], "entry_gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
