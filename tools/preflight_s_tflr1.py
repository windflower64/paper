#!/usr/bin/env python3
"""TFLR1零点等价、权重继承、两步梯度和真实batch16预检。"""

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
        / "experiments/phase_s/s_tflr1_joint_localization_b16_60e_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT.parent / "weights/dfine_n_coco.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "reports/27_tflr/S_TFLR1/preflight.json",
    )
    parser.add_argument("--expected-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260823)
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
        raise RuntimeError("TFLR1 batch16预检需要CUDA")
    if args.expected_batch != 16:
        raise RuntimeError("正式协议固定batch16")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    source, source_name = checkpoint_weights(args.checkpoint)
    model = cfg.model
    matched, missing, unexpected = load_compatible(model, source)
    prefix = "localization_refiner."
    tflr_missing = [key for key in missing if key.startswith(prefix)]
    allowed_dataset_transfer = {
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
    transfer_missing = [key for key in missing if key in allowed_dataset_transfer]
    other_missing = [
        key
        for key in missing
        if not key.startswith(prefix) and key not in allowed_dataset_transfer
    ]
    if not tflr_missing or other_missing or unexpected:
        raise RuntimeError(
            "权重隔离失败："
            f"tflr_missing={tflr_missing}, other_missing={other_missing}, "
            f"unexpected={unexpected}"
        )
    required_shared = (
        "backbone.stages.1.blocks.0.layers.0.conv.weight",
        "encoder.input_proj.0.conv.weight",
        "decoder.dec_bbox_head.0.layers.0.weight",
    )
    if any(key not in matched for key in required_shared):
        raise RuntimeError("COCO关键共享权重没有完整继承")

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device)
    criterion = cfg.criterion.to(device)
    optimizer = cfg.optimizer
    refiner = model.localization_refiner
    if refiner is None:
        raise RuntimeError("配置没有启用TFLR1")
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if not set(map(id, refiner.parameters())).issubset(optimizer_ids):
        raise RuntimeError("优化器没有覆盖全部TFLR1参数")

    samples, targets = next(iter(cfg.train_dataloader))
    if samples.shape[0] != args.expected_batch:
        raise RuntimeError(
            f"真实batch错误：预期{args.expected_batch}，实际{samples.shape[0]}"
        )
    samples = samples.to(device, non_blocking=True)
    targets = move_targets(targets, device)

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        refiner.frequency_mode = "full"
        initial_full = model(samples)
        refiner.frequency_mode = "zero"
        initial_zero = model(samples)
    initial_base_error = float(
        (initial_full["pred_boxes"] - initial_full["base_pred_boxes"]).abs().max()
    )
    initial_mode_error = float(
        (initial_full["pred_boxes"] - initial_zero["pred_boxes"]).abs().max()
    )
    initial_logit_error = float(
        (initial_full["pred_logits"] - initial_zero["pred_logits"]).abs().max()
    )
    if initial_base_error != 0.0 or initial_mode_error != 0.0:
        raise RuntimeError(
            "零点没有严格等价："
            f"base_error={initial_base_error}, mode_error={initial_mode_error}"
        )
    if initial_logit_error != 0.0:
        raise RuntimeError("TFLR改变了分类输出")

    model.train()
    criterion.train()
    refiner.frequency_mode = "full"
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=1024.0)
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
        final_gradient = grad_norm(refiner.regressor[-1].parameters())
        projection_gradient = grad_norm(refiner.channel_projection.parameters())
        all_refiner_gradient = grad_norm(refiner.parameters())
        if final_gradient <= 0.0:
            raise RuntimeError(f"第{step + 1}步零初始化输出层没有梯度")
        if step == 1 and projection_gradient <= 0.0:
            raise RuntimeError("第二步高频投影仍然没有梯度")
        scaler.step(optimizer)
        scaler.update()
        step_reports.append(
            {
                "step": step + 1,
                "总损失": float(total.detach()),
                "各损失": {
                    key: float(value.detach()) for key, value in losses.items()
                },
                "TFLR总梯度L2": all_refiner_gradient,
                "末层梯度L2": final_gradient,
                "高频投影梯度L2": projection_gradient,
            }
        )

    model.eval()
    mode_boxes = {}
    mode_logits = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for mode in sorted(refiner.VALID_FREQUENCY_MODES):
            refiner.frequency_mode = mode
            outputs = model(samples)
            mode_boxes[mode] = outputs["pred_boxes"].detach().float()
            mode_logits[mode] = outputs["pred_logits"].detach().float()
            if mode == "zero":
                zero_base_error = float(
                    (outputs["pred_boxes"] - outputs["base_pred_boxes"]).abs().max()
                )
    refiner.frequency_mode = "full"
    mode_box_delta = {
        mode: float((boxes - mode_boxes["zero"]).abs().mean())
        for mode, boxes in mode_boxes.items()
        if mode != "zero"
    }
    mode_logit_error = {
        mode: float((logits - mode_logits["zero"]).abs().max())
        for mode, logits in mode_logits.items()
        if mode != "zero"
    }
    if zero_base_error != 0.0:
        raise RuntimeError(f"ZERO模式没有严格回到原框：{zero_base_error}")
    if mode_box_delta["full"] <= 0.0:
        raise RuntimeError("两步更新后TFLR仍未产生框修正")
    if max(mode_logit_error.values()) != 0.0:
        raise RuntimeError(f"频率干预改变了分类输出：{mode_logit_error}")

    report = {
        "状态": "通过",
        "方案": "TFLR1目标引导频率定位残差",
        "配置": str(args.config),
        "COCO权重": str(args.checkpoint),
        "权重来源": source_name,
        "共享匹配张量数": len(matched),
        "TFLR新张量": tflr_missing,
        "数据集迁移允许缺失张量": transfer_missing,
        "TFLR参数量": sum(parameter.numel() for parameter in refiner.parameters()),
        "训练batch": int(samples.shape[0]),
        "输入形状": list(samples.shape),
        "初始相对原框最大误差": initial_base_error,
        "初始FULL相对ZERO最大误差": initial_mode_error,
        "初始分类最大误差": initial_logit_error,
        "两步梯度": step_reports,
        "更新后各模式相对ZERO平均框变化": mode_box_delta,
        "更新后各模式分类最大误差": mode_logit_error,
        "ZERO相对原框最大误差": zero_base_error,
        "优化器覆盖TFLR": True,
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
