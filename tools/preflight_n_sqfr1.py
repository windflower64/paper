#!/usr/bin/env python3
"""N-SQFR1真实batch、零点等价、权重继承和两步梯度预检。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "experiments/phase_n/n_sqfr1_c_gq1_b8a4_20e_testdev_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT.parent / "weights/m_sd2_joint_coco_thermal_identity_init.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "reports/95_n_sqfr1/N_SQFR1/preflight.json",
    )
    parser.add_argument("--expected-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def checkpoint_weights(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("ema", {}).get("module")
    source_name = "ema.module"
    if not isinstance(source, dict):
        source = checkpoint.get("model")
        source_name = "model"
    if not isinstance(source, dict):
        raise RuntimeError(f"权重文件中没有模型参数：{path}")
    return source, source_name


def load_compatible(model, source):
    own = model.state_dict()
    matched = {
        key: value
        for key, value in source.items()
        if key in own and own[key].shape == value.shape
    }
    incompatible = model.load_state_dict(matched, strict=False)
    return matched, list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def move_targets(targets, device):
    return [
        {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def grad_norm(parameters):
    terms = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.0


def detection_loss(cfg, criterion, outputs, targets, step):
    with torch.autocast("cuda", enabled=False):
        losses = criterion(
            outputs,
            targets,
            epoch=0,
            step=step,
            global_step=step,
            epoch_step=len(cfg.train_dataloader),
        )
        total = sum(losses.values())
    return total, losses


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("N-SQFR1正式预检需要CUDA")
    if args.expected_batch != 8:
        raise RuntimeError("首轮正式协议固定物理batch=8")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    source, source_name = checkpoint_weights(args.checkpoint)
    matched, missing, unexpected = load_compatible(model, source)

    sqfr_prefix = "decoder.sqfr_refiner."
    sqfr_missing = [key for key in missing if key.startswith(sqfr_prefix)]
    allowed_transfer = {
        "decoder.anchors",
        "decoder.valid_mask",
        "decoder.denoising_class_embed.weight",
        "decoder.enc_score_head.weight",
        "decoder.enc_score_head.bias",
        "decoder.dec_score_head.0.weight",
        "decoder.dec_score_head.0.bias",
        "decoder.dec_score_head.1.weight",
        "decoder.dec_score_head.1.bias",
        "decoder.dec_score_head.2.weight",
        "decoder.dec_score_head.2.bias",
    }
    other_missing = [
        key
        for key in missing
        if not key.startswith(sqfr_prefix)
        and ".pat_sf." not in key
        and key not in allowed_transfer
    ]
    if not sqfr_missing or other_missing or unexpected:
        raise RuntimeError(
            "权重隔离失败："
            f"sqfr_missing={sqfr_missing}, other_missing={other_missing}, "
            f"unexpected={unexpected}"
        )
    if model.backbone.return_idx != [1, 2, 3]:
        raise RuntimeError(f"S8入口错误：{model.backbone.return_idx}")
    if len(model.encoder.in_channels) != 2:
        raise RuntimeError("N-SQFR1不得把S8送入完整HybridEncoder")
    if model.decoder.hrqs_enabled or not model.decoder.sqfr_enabled:
        raise RuntimeError("N-SQFR1必须关闭HRQS并启用SQFR")

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device)
    criterion = cfg.criterion.to(device)
    optimizer = cfg.optimizer
    refiner = model.decoder.sqfr_refiner
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if not set(map(id, refiner.parameters())).issubset(optimizer_ids):
        raise RuntimeError("优化器没有覆盖全部N-SQFR1参数")

    samples, targets = next(iter(cfg.train_dataloader))
    if samples.shape[0] != args.expected_batch:
        raise RuntimeError(
            f"真实batch错误：预期{args.expected_batch}，实际{samples.shape[0]}"
        )
    samples = samples.to(device, non_blocking=True)
    targets = move_targets(targets, device)

    # The zero-initialized branch must be identical to the structurally
    # disabled detector before any update.
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        refiner.feature_mode = "full"
        initial_full = model(samples[:1])
        refiner.feature_mode = "zero"
        initial_zero = model(samples[:1])
        model.decoder.sqfr_refiner = None
        initial_disabled = model(samples[:1])
        model.decoder.sqfr_refiner = refiner
    initial_box_error = float(
        (initial_full["pred_boxes"] - initial_disabled["pred_boxes"]).abs().max()
    )
    initial_zero_error = float(
        (initial_zero["pred_boxes"] - initial_disabled["pred_boxes"]).abs().max()
    )
    initial_logit_error = float(
        (initial_full["pred_logits"] - initial_disabled["pred_logits"]).abs().max()
    )
    if initial_box_error != 0.0 or initial_zero_error != 0.0:
        raise RuntimeError(
            "零初始化没有严格回到C基线："
            f"full={initial_box_error}, zero={initial_zero_error}"
        )
    if initial_logit_error != 0.0:
        raise RuntimeError("N-SQFR1初始状态改变了分类输出")

    model.train()
    criterion.train()
    refiner.feature_mode = "full"
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=128.0)
    step_reports = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(samples, targets=targets)
        total, losses = detection_loss(cfg, criterion, outputs, targets, step)
        if not torch.isfinite(total):
            raise RuntimeError(f"第{step + 1}步损失不是有限值")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        final_gradient = grad_norm(refiner.delta_head[-1].parameters())
        feature_gradient = grad_norm(refiner.feature_projection.parameters())
        all_gradient = grad_norm(refiner.parameters())
        if final_gradient <= 0.0:
            raise RuntimeError(f"第{step + 1}步零初始化输出层没有梯度")
        if step == 1 and feature_gradient <= 0.0:
            raise RuntimeError("第二步S8投影仍然没有梯度")
        scaler.step(optimizer)
        scaler.update()
        step_reports.append(
            {
                "step": step + 1,
                "总损失": float(total.detach()),
                "N模块总梯度L2": all_gradient,
                "末层梯度L2": final_gradient,
                "S8投影梯度L2": feature_gradient,
                "loss_bbox": float(losses["loss_bbox"].detach()),
                "loss_giou": float(losses["loss_giou"].detach()),
            }
        )

    # After opening, full/shifted S8 must cause a correction, zero S8 must be
    # exactly equivalent to physically disabling the refiner, and logits must
    # remain bit-identical in every mode.
    model.eval()
    mode_boxes = {}
    mode_logits = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for mode in ("full", "zero", "shifted"):
            refiner.feature_mode = mode
            outputs = model(samples[:1])
            mode_boxes[mode] = outputs["pred_boxes"].detach().float()
            mode_logits[mode] = outputs["pred_logits"].detach().float()
        model.decoder.sqfr_refiner = None
        disabled = model(samples[:1])
        model.decoder.sqfr_refiner = refiner
    refiner.feature_mode = "full"

    zero_disabled_error = float(
        (mode_boxes["zero"] - disabled["pred_boxes"].float()).abs().max()
    )
    full_zero_delta = float((mode_boxes["full"] - mode_boxes["zero"]).abs().mean())
    shifted_full_delta = float(
        (mode_boxes["shifted"] - mode_boxes["full"]).abs().mean()
    )
    logit_errors = {
        mode: float((values - mode_logits["zero"]).abs().max())
        for mode, values in mode_logits.items()
    }
    if zero_disabled_error != 0.0:
        raise RuntimeError(f"全零S8没有严格关闭修正：{zero_disabled_error}")
    if full_zero_delta <= 0.0 or shifted_full_delta <= 0.0:
        raise RuntimeError(
            "更新后N-SQFR1没有形成可干预的S8定位响应："
            f"full_zero={full_zero_delta}, shifted_full={shifted_full_delta}"
        )
    if max(logit_errors.values()) != 0.0:
        raise RuntimeError(f"S8干预改变了分类输出：{logit_errors}")
    selected = refiner.last_selected_indices
    if selected is None or selected.shape[1] != 64:
        raise RuntimeError("N-SQFR1没有固定选择64个标准查询")

    report = {
        "status": "PASS",
        "physical_batch": int(samples.shape[0]),
        "gradient_accumulation_steps": int(
            cfg.yaml_cfg["gradient_accumulation_steps"]
        ),
        "effective_batch_size": int(samples.shape[0])
        * int(cfg.yaml_cfg["gradient_accumulation_steps"]),
        "inference_refined_queries": int(selected.shape[1]),
        "initial_full_disabled_box_error": initial_box_error,
        "initial_zero_disabled_box_error": initial_zero_error,
        "zero_disabled_box_error": zero_disabled_error,
        "状态": "通过",
        "方案": "N-SQFR1稀疏查询引导的高分辨率定位精修",
        "配置": str(args.config.resolve()),
        "初始化权重": str(args.checkpoint.resolve()),
        "权重来源": source_name,
        "共享匹配张量数": len(matched),
        "N模块新张量": sqfr_missing,
        "N模块参数量": sum(parameter.numel() for parameter in refiner.parameters()),
        "训练batch": int(samples.shape[0]),
        "梯度累积": int(cfg.yaml_cfg["gradient_accumulation_steps"]),
        "有效batch": int(samples.shape[0])
        * int(cfg.yaml_cfg["gradient_accumulation_steps"]),
        "输入形状": list(samples.shape),
        "固定精修查询数": int(selected.shape[1]),
        "初始FULL相对禁用最大框误差": initial_box_error,
        "初始ZERO相对禁用最大框误差": initial_zero_error,
        "初始分类最大误差": initial_logit_error,
        "两步梯度": step_reports,
        "更新后FULL相对ZERO平均框变化": full_zero_delta,
        "更新后SHIFTED相对FULL平均框变化": shifted_full_delta,
        "ZERO相对物理禁用最大框误差": zero_disabled_error,
        "各模式分类最大误差": logit_errors,
        "优化器覆盖N模块": True,
        "峰值分配显存MiB": torch.cuda.max_memory_allocated(device) / 2**20,
        "峰值保留显存MiB": torch.cuda.max_memory_reserved(device) / 2**20,
    }
    if not all(
        math.isfinite(value)
        for value in (
            report["峰值分配显存MiB"],
            report["峰值保留显存MiB"],
        )
    ):
        raise RuntimeError("显存统计异常")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
