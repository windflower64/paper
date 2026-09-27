#!/usr/bin/env python3
"""S-HRBR5-D0：高分辨率细节因果验证的训练前检查。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.core import YAMLConfig
from train_s_hrbr1_refinebox import (
    BackboneFeatureTap,
    RefineBoxHead,
    load_frozen_detector,
    matched_training_boxes,
    move_targets,
    regression_loss,
)


MODES = ("full", "lowpass", "shifted_detail")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT.parent
        / "outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent
        / "reports/24_high_resolution_box_refinement"
        / "S_HRBR5_D0_DETAIL_CAUSALITY/preflight.json",
    )
    return parser.parse_args()


def max_state_error(left: dict, right: dict) -> float:
    if left.keys() != right.keys():
        raise RuntimeError("三组校准头的参数键不一致")
    maximum = 0.0
    for key in left:
        if left[key].dtype == torch.bool:
            error = 0.0 if torch.equal(left[key], right[key]) else 1.0
        else:
            error = float((left[key] - right[key]).abs().max())
        maximum = max(maximum, error)
    return maximum


def build_head(channels: tuple[int, ...], mode: str, seed: int) -> RefineBoxHead:
    torch.manual_seed(seed)
    return RefineBoxHead(
        channels,
        d_model=64,
        roi_size=7,
        refine_steps=3,
        feature_mode=mode,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("S-HRBR5-D0 预检需要 CUDA")

    torch.manual_seed(args.seed)
    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = args.batch_size
    detector, weight_source = load_frozen_detector(cfg, args.checkpoint)
    matcher = cfg.criterion.cuda().eval().matcher
    tap = BackboneFeatureTap(detector.backbone)
    channels = tuple(
        detector.backbone._out_channels[index] for index in (0, 1, 2, 3)
    )

    cpu_heads = {mode: build_head(channels, mode, args.seed) for mode in MODES}
    parameter_counts = {
        mode: sum(parameter.numel() for parameter in head.parameters())
        for mode, head in cpu_heads.items()
    }
    initialization_errors = {
        mode: max_state_error(
            cpu_heads["full"].state_dict(), cpu_heads[mode].state_dict()
        )
        for mode in MODES
    }
    if len(set(parameter_counts.values())) != 1 or max(initialization_errors.values()) != 0:
        raise RuntimeError(
            f"三组不是严格同构同初始化：count={parameter_counts}, "
            f"error={initialization_errors}"
        )
    base_state = cpu_heads["full"].state_dict()
    del cpu_heads

    samples, targets = next(iter(cfg.train_dataloader))
    actual_batch = int(samples.shape[0])
    if actual_batch != args.batch_size:
        raise RuntimeError(
            f"真实训练批次不是 batch={args.batch_size}，而是 {actual_batch}"
        )
    samples = samples.cuda(non_blocking=True)
    targets = move_targets(targets, "cuda")
    tap.clear()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        outputs = detector(samples)
        features = tuple(feature.detach().float() for feature in tap.features())
    predicted, truth, batch_ids = matched_training_boxes(outputs, targets, matcher)
    if predicted is None:
        raise RuntimeError("真实 batch 中没有匹配到正样本")
    predicted = predicted.float().detach()
    truth = truth.float().detach()
    batch_ids = batch_ids.detach()

    # 检查 FULL 显式模式与未指定 feature_mode 的原始 HRBR1 路径完全相同。
    legacy = RefineBoxHead(channels, 64, 7, 3).cuda()
    explicit_full = RefineBoxHead(
        channels, 64, 7, 3, feature_mode="full"
    ).cuda()
    legacy.load_state_dict(base_state, strict=True)
    explicit_full.load_state_dict(base_state, strict=True)
    legacy.force_identity = False
    explicit_full.force_identity = False
    sample_features = tuple(feature[:1] for feature in features)
    sample_mask = batch_ids == 0
    sample_boxes = predicted[sample_mask][:32]
    sample_batch_ids = torch.zeros(
        len(sample_boxes), dtype=torch.long, device=sample_boxes.device
    )
    with torch.no_grad():
        legacy_output = legacy(sample_features, sample_boxes, sample_batch_ids)[-1]
        full_output = explicit_full(sample_features, sample_boxes, sample_batch_ids)[-1]
    full_legacy_error = float((legacy_output - full_output).abs().max())
    if full_legacy_error != 0.0:
        raise RuntimeError(f"FULL 没有保持原 HRBR1 路径：max_error={full_legacy_error}")
    del legacy, explicit_full, legacy_output, full_output
    torch.cuda.empty_cache()

    # 在同一个真实 P2 上验证分解可逆，以及空间错位只改变位置、不改变能量。
    probe = RefineBoxHead(channels, 64, 7, 3, feature_mode="full").cuda()
    probe.load_state_dict(base_state, strict=True)
    with torch.no_grad():
        p2 = probe.fpn(sample_features)[0]
        lowpass = F.interpolate(
            F.avg_pool2d(p2, kernel_size=2, stride=2),
            size=p2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        detail = p2 - lowpass
        shifted = torch.roll(
            detail,
            shifts=(max(1, p2.shape[-2] // 3), max(1, p2.shape[-1] // 5)),
            dims=(-2, -1),
        )
        reconstruction_error = float((lowpass + detail - p2).abs().max())
        detail_energy = float(detail.square().sum().sqrt())
        shifted_energy = float(shifted.square().sum().sqrt())
        energy_relative_error = abs(shifted_energy - detail_energy) / max(
            detail_energy, 1e-12
        )
    if reconstruction_error > 1e-6 or energy_relative_error > 1e-6:
        raise RuntimeError(
            "P2 细节控制不满足重构/等能量要求："
            f"reconstruction={reconstruction_error}, energy={energy_relative_error}"
        )
    del probe, p2, lowpass, detail, shifted
    torch.cuda.empty_cache()

    mode_results = {}
    for mode in MODES:
        head = RefineBoxHead(
            channels, 64, 7, 3, feature_mode=mode
        ).cuda()
        head.load_state_dict(base_state, strict=True)

        # 未训练时必须保持输入框不变，三种特征处理不得影响检测器基线。
        head.force_identity = True
        with torch.no_grad():
            identity_output = head(
                sample_features, sample_boxes, sample_batch_ids
            )[-1]
        identity_error = float((identity_output - sample_boxes).abs().max())
        if identity_error != 0.0:
            raise RuntimeError(f"{mode} 恒等预检失败：max_error={identity_error}")

        # 用真实 batch=16 做完整三步回归和反向传播。
        head.force_identity = False
        head.train()
        head.zero_grad(set_to_none=True)
        sequence = head(features, predicted, batch_ids)
        losses = [regression_loss(boxes, truth)[0] for boxes in sequence]
        total_loss = torch.stack(losses).sum()
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"{mode} 出现非有限损失：{float(total_loss)}")
        total_loss.backward()
        gradients = [
            parameter.grad.detach().float().square().sum()
            for parameter in head.parameters()
            if parameter.grad is not None
        ]
        gradient_norm = float(torch.stack(gradients).sum().sqrt())
        if not torch.isfinite(torch.tensor(gradient_norm)) or gradient_norm <= 0:
            raise FloatingPointError(f"{mode} 梯度异常：{gradient_norm}")
        mode_results[mode] = {
            "loss": float(total_loss.detach()),
            "gradient_norm": gradient_norm,
            "identity_box_max_error": identity_error,
            "finite": True,
        }
        del head, sequence, losses, total_loss, gradients, identity_output
        torch.cuda.empty_cache()

    report = {
        "experiment": "S-HRBR5-D0-DETAIL-CAUSALITY-PREFLIGHT",
        "batch_size": actual_batch,
        "matched_boxes": int(len(predicted)),
        "feature_shapes": [list(feature.shape) for feature in features],
        "checkpoint_weight_source": weight_source,
        "parameter_counts": parameter_counts,
        "same_initialization_max_errors": initialization_errors,
        "full_legacy_output_max_error": full_legacy_error,
        "p2_detail_controls": {
            "lowpass_plus_detail_reconstruction_max_error": reconstruction_error,
            "detail_l2": detail_energy,
            "shifted_detail_l2": shifted_energy,
            "shifted_energy_relative_error": energy_relative_error,
        },
        "modes": mode_results,
        "pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
