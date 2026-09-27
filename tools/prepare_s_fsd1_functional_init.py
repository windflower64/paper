#!/usr/bin/env python3
"""让 FSD 先复现原预训练下采样功能，并生成可用于联合训练的权重。"""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


PREFIX = "backbone.stages.2.downsample."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "experiments/phase_s/s_fsd1_functional_init_s8_s16_b16_60e_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT.parent / "weights/dfine_n_coco.pth",
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=ROOT.parent / "weights/s_fsd1_functional_init_coco_tuning.pth",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT.parent
        / "reports/26_fsd_down/S_FSD1_FUNCTIONAL_INIT/functional_init.json",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--max-relative-rmse", type=float, default=0.25)
    parser.add_argument("--min-cosine", type=float, default=0.95)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_weights(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("ema", {}).get("module")
    source_name = "ema.module"
    if not isinstance(source, dict):
        source = checkpoint.get("model")
        source_name = "model"
    if not isinstance(source, dict):
        raise RuntimeError(f"权重文件中没有 model 或 ema.module：{path}")
    return source, source_name


def load_compatible(model, source):
    own = model.state_dict()
    matched = {
        key: value
        for key, value in source.items()
        if key in own and own[key].shape == value.shape
    }
    incompatible = model.load_state_dict(matched, strict=False)
    return {
        "matched_count": len(matched),
        "missing": list(incompatible.missing_keys),
        "source_only": [key for key in source if key not in matched],
    }


def make_config(config_path, fsd_stage):
    cfg = YAMLConfig(str(config_path))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["HGNetv2"]["fsd_stage"] = int(fsd_stage)
    cfg.yaml_cfg["HGNetv2"]["reinit_downsample_stage"] = -1
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = 16
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = 16
    return cfg


def s8_feature(backbone, samples):
    x = backbone.stem(samples)
    x = backbone.stages[0](x)
    x = backbone.stages[1](x)
    return x


def teacher_target(backbone, feature):
    stage = backbone.stages[2]
    return stage.downsample(stage.preblur(feature))


def student_prediction(backbone, feature):
    stage = backbone.stages[2]
    return stage.downsample(stage.preblur(feature))


@torch.no_grad()
def evaluate_match(teacher, student, loader, device, max_batches):
    teacher.eval()
    student.eval()
    error_sq = 0.0
    target_sq = 0.0
    absolute_error = 0.0
    element_count = 0
    cosine_sum = 0.0
    sample_count = 0
    used_batches = 0
    for samples, _ in loader:
        samples = samples.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            feature = s8_feature(teacher, samples)
            target = teacher_target(teacher, feature)
            prediction = student_prediction(student, feature)
        difference = prediction.float() - target.float()
        error_sq += float(difference.square().sum())
        target_sq += float(target.float().square().sum())
        absolute_error += float(difference.abs().sum())
        element_count += difference.numel()
        cosine_sum += float(
            F.cosine_similarity(
                prediction.float().flatten(1), target.float().flatten(1), dim=1
            ).sum()
        )
        sample_count += samples.shape[0]
        used_batches += 1
        if used_batches >= max_batches:
            break
    return {
        "验证批数": used_batches,
        "验证样本数": sample_count,
        "相对RMSE": math.sqrt(error_sq / max(target_sq, 1e-12)),
        "平均绝对误差": absolute_error / max(element_count, 1),
        "平均余弦相似度": cosine_sum / max(sample_count, 1),
    }


def build_merged_checkpoint(source, fsd_module, output_path, metadata):
    merged = {
        key: value.detach().cpu().clone()
        for key, value in source.items()
        if not key.startswith(PREFIX)
    }
    fsd_state = fsd_module.state_dict()
    for key, value in fsd_state.items():
        merged[PREFIX + key] = value.detach().cpu().clone()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": merged, "fsd1_functional_init": metadata}, output_path)
    return merged, sorted(PREFIX + key for key in fsd_state)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("功能初始化需要 CUDA")
    if args.batch_size != 16:
        raise RuntimeError("预注册协议固定 batch=16，不接受临时修改")
    if args.epochs != 3:
        raise RuntimeError("预注册协议固定功能初始化 3 轮，不进行轮数搜索")
    set_seed(args.seed)
    device = torch.device("cuda")
    source, source_name = checkpoint_weights(args.checkpoint)

    teacher_cfg = make_config(args.config, fsd_stage=-1)
    teacher_model = teacher_cfg.model
    teacher_load = load_compatible(teacher_model, source)
    teacher = teacher_model.backbone
    del teacher_model

    student_cfg = make_config(args.config, fsd_stage=2)
    student_model = student_cfg.model
    student_load = load_compatible(student_model, source)
    student = student_model.backbone
    del student_model

    teacher_missing = [key for key in teacher_load["missing"] if key.startswith(PREFIX)]
    fsd_missing = [key for key in student_load["missing"] if key.startswith(PREFIX)]
    if teacher_missing:
        raise RuntimeError(f"教师的原始下采样未继承完整：{teacher_missing}")
    if not fsd_missing:
        raise RuntimeError("FSD 参数没有处于新初始化状态，协议检查失败")

    teacher = teacher.to(device).eval()
    student = student.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    fsd = student.stages[2].downsample
    for parameter in fsd.parameters():
        parameter.requires_grad_(True)

    train_loader = student_cfg.train_dataloader
    val_loader = student_cfg.val_dataloader
    if train_loader.batch_size is not None and train_loader.batch_size != args.batch_size:
        raise RuntimeError(
            f"训练 batch 错误：预期 {args.batch_size}，实际 {train_loader.batch_size}"
        )
    optimizer = torch.optim.AdamW(
        fsd.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=1024.0)

    before = evaluate_match(
        teacher, student, val_loader, device, args.validation_batches
    )
    history = []
    for epoch in range(args.epochs):
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        fsd.train()
        loss_sum = 0.0
        mse_sum = 0.0
        cosine_loss_sum = 0.0
        batches = 0
        for samples, _ in train_loader:
            if samples.shape[0] != args.batch_size:
                continue
            samples = samples.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                feature = s8_feature(teacher, samples)
                target = teacher_target(teacher, feature)
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = student_prediction(student, feature.detach())
            prediction32 = prediction.float()
            target32 = target.float()
            normalized_mse = F.mse_loss(prediction32, target32) / (
                target32.square().mean().detach().clamp_min(1e-6)
            )
            cosine_loss = 1.0 - F.cosine_similarity(
                prediction32.flatten(1), target32.flatten(1), dim=1
            ).mean()
            loss = normalized_mse + 0.1 * cosine_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"功能初始化出现非有限损失：{float(loss)}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(fsd.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            mse_sum += float(normalized_mse.detach())
            cosine_loss_sum += float(cosine_loss.detach())
            batches += 1
        if batches == 0:
            raise RuntimeError("没有得到完整的 batch16 训练批次")
        metrics = evaluate_match(
            teacher, student, val_loader, device, args.validation_batches
        )
        history.append(
            {
                "epoch": epoch,
                "平均总损失": loss_sum / batches,
                "平均归一化MSE": mse_sum / batches,
                "平均余弦损失": cosine_loss_sum / batches,
                "训练批数": batches,
                "验证": metrics,
            }
        )
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)

    after = history[-1]["验证"]
    passed = (
        after["相对RMSE"] <= args.max_relative_rmse
        and after["平均余弦相似度"] >= args.min_cosine
    )
    if not passed:
        report = {
            "状态": "功能复现门槛未通过，禁止正式训练",
            "配置": str(args.config),
            "原始权重": str(args.checkpoint),
            "权重来源": source_name,
            "固定协议": {
                "轮数": args.epochs,
                "batch": args.batch_size,
                "学习率": args.lr,
                "最大相对RMSE": args.max_relative_rmse,
                "最小余弦相似度": args.min_cosine,
            },
            "训练前": before,
            "训练历史": history,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            "功能初始化未达到预注册门槛："
            f"relative_rmse={after['相对RMSE']:.6f}, "
            f"cosine={after['平均余弦相似度']:.6f}"
        )

    metadata = {
        "来源": str(args.checkpoint),
        "权重来源": source_name,
        "方法": "只训练 FSD，使其复现原预训练 S8→S16 下采样输出",
        "轮数": args.epochs,
        "batch": args.batch_size,
        "seed": args.seed,
        "训练前": before,
        "训练后": after,
    }
    merged, inserted = build_merged_checkpoint(
        source, fsd, args.output_checkpoint, metadata
    )

    # 保存前进行结构审计：新权重必须完整覆盖 FSD，旧下采样键不得残留。
    fsd_keys = set(inserted)
    if not fsd_keys.issubset(merged):
        raise RuntimeError("合并权重缺少 FSD 参数")
    old_keys = [
        key
        for key in merged
        if key.startswith(PREFIX) and key not in fsd_keys
    ]
    if old_keys:
        raise RuntimeError(f"合并权重仍含旧下采样参数：{old_keys}")

    report = {
        "状态": "通过，可进入一次完整联合训练",
        "配置": str(args.config),
        "原始权重": str(args.checkpoint),
        "输出权重": str(args.output_checkpoint),
        "权重来源": source_name,
        "教师匹配张量数": teacher_load["matched_count"],
        "学生匹配张量数": student_load["matched_count"],
        "FSD新参数张量": fsd_missing,
        "插入FSD张量数": len(inserted),
        "固定协议": {
            "轮数": args.epochs,
            "batch": args.batch_size,
            "学习率": args.lr,
            "最大相对RMSE": args.max_relative_rmse,
            "最小余弦相似度": args.min_cosine,
        },
        "训练前": before,
        "训练历史": history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    del teacher, student, fsd
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
