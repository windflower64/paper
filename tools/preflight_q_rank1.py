#!/usr/bin/env python3
"""Q-Rank1真实batch、零点等价、梯度与可关闭性预检。"""

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
        / "experiments/phase_q/q_rank1_c_gq1_b8a4_20e_testdev_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT.parent / "weights/m_sd2_joint_coco_thermal_identity_init.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "reports/96_q_rank1/Q_RANK1/preflight.json",
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
        raise RuntimeError("Q-Rank1正式预检需要CUDA")
    if args.expected_batch != 8:
        raise RuntimeError("首轮正式协议固定物理batch=8")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    source, source_name = checkpoint_weights(args.checkpoint)
    matched, missing, unexpected = load_compatible(model, source)

    rank_prefix = "decoder.qrank_score_bias."
    rank_missing = [key for key in missing if key.startswith(rank_prefix)]
    if rank_missing != ["decoder.qrank_score_bias.rank_bias"] or unexpected:
        raise RuntimeError(
            "Q-Rank1权重隔离失败："
            f"rank_missing={rank_missing}, unexpected={unexpected}"
        )
    rank_module = model.decoder.qrank_score_bias
    if not model.decoder.qrank_enabled or rank_module is None:
        raise RuntimeError("配置没有启用Q-Rank1")
    if tuple(rank_module.rank_bias.shape) != (3, 300, 1):
        raise RuntimeError(
            f"Q-Rank1名次表形状错误：{tuple(rank_module.rank_bias.shape)}"
        )
    if torch.count_nonzero(rank_module.rank_bias).item() != 0:
        raise RuntimeError("Q-Rank1没有从严格零偏置初始化")

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device)
    criterion = cfg.criterion.to(device)
    optimizer = cfg.optimizer
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if not set(map(id, rank_module.parameters())).issubset(optimizer_ids):
        raise RuntimeError("优化器没有覆盖Q-Rank1参数")

    samples, targets = next(iter(cfg.train_dataloader))
    if samples.shape[0] != args.expected_batch:
        raise RuntimeError(
            f"真实batch错误：预期{args.expected_batch}，实际{samples.shape[0]}"
        )
    samples = samples.to(device, non_blocking=True)
    targets = move_targets(targets, device)

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        rank_module.intervention_mode = "learned"
        initial_learned = model(samples[:1])
        rank_module.intervention_mode = "zero"
        initial_zero = model(samples[:1])
        model.decoder.qrank_score_bias = None
        initial_disabled = model(samples[:1])
        model.decoder.qrank_score_bias = rank_module
    initial_logit_error = float(
        (initial_learned["pred_logits"] - initial_disabled["pred_logits"])
        .abs()
        .max()
    )
    initial_zero_error = float(
        (initial_zero["pred_logits"] - initial_disabled["pred_logits"])
        .abs()
        .max()
    )
    initial_box_error = float(
        (initial_learned["pred_boxes"] - initial_disabled["pred_boxes"])
        .abs()
        .max()
    )
    if initial_logit_error != 0.0 or initial_zero_error != 0.0:
        raise RuntimeError(
            "Q-Rank1零初始化没有严格回到C："
            f"learned={initial_logit_error}, zero={initial_zero_error}"
        )
    if initial_box_error != 0.0:
        raise RuntimeError("Q-Rank1初始状态改变了预测框")

    model.train()
    criterion.train()
    rank_module.intervention_mode = "learned"
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
        rank_gradient = grad_norm(rank_module.parameters())
        if rank_gradient <= 0.0:
            raise RuntimeError(f"第{step + 1}步Q-Rank1没有梯度")
        scaler.step(optimizer)
        scaler.update()
        step_reports.append(
            {
                "step": step + 1,
                "总损失": float(total.detach()),
                "Q模块梯度L2": rank_gradient,
                "loss_vfl": float(losses["loss_vfl"].detach()),
                "loss_bbox": float(losses["loss_bbox"].detach()),
                "loss_giou": float(losses["loss_giou"].detach()),
            }
        )

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        rank_module.intervention_mode = "learned"
        learned = model(samples[:1])
        rank_module.intervention_mode = "zero"
        zero = model(samples[:1])
        model.decoder.qrank_score_bias = None
        disabled = model(samples[:1])
        model.decoder.qrank_score_bias = rank_module
    rank_module.intervention_mode = "learned"

    learned_zero_logit_delta = float(
        (learned["pred_logits"] - zero["pred_logits"]).abs().mean()
    )
    zero_disabled_logit_error = float(
        (zero["pred_logits"] - disabled["pred_logits"]).abs().max()
    )
    learned_disabled_box_error = float(
        (learned["pred_boxes"] - disabled["pred_boxes"]).abs().max()
    )
    query_slot_std = float(rank_module.rank_bias[-1, :, 0].float().std())
    diagnostic_snapshot = {
        "rank_bias_abs_max": float(rank_module.rank_bias.detach().abs().max()),
        "rank_bias_abs_mean": float(rank_module.rank_bias.detach().abs().mean()),
        "final_layer_rank_bias_std": query_slot_std,
        "learned_zero_logit_delta": learned_zero_logit_delta,
        "zero_disabled_logit_error": zero_disabled_logit_error,
        "learned_logits_dtype": str(learned["pred_logits"].dtype),
    }
    print(json.dumps({"diagnostic_snapshot": diagnostic_snapshot}, ensure_ascii=False))
    if learned_zero_logit_delta <= 0.0:
        raise RuntimeError("两步更新后Q-Rank1仍未改变最终分数")
    if zero_disabled_logit_error != 0.0:
        raise RuntimeError("Q-Rank1的zero干预不等价于物理关闭")
    if learned_disabled_box_error != 0.0:
        raise RuntimeError("Q-Rank1错误地改变了预测框")
    if query_slot_std <= 0.0:
        raise RuntimeError("Q-Rank1只学成了全查询相同常数")

    report = {
        "status": "PASS",
        "launch_checks": {
            "initial_learned_disabled_score_error": initial_logit_error,
            "initial_zero_disabled_score_error": initial_zero_error,
            "initial_learned_disabled_box_error": initial_box_error,
            "updated_zero_disabled_score_error": zero_disabled_logit_error,
            "updated_learned_disabled_box_error": learned_disabled_box_error,
            "q_parameter_count": sum(
                parameter.numel() for parameter in rank_module.parameters()
            ),
        },
        "方案": "Q-Rank1最终查询质量排序",
        "配置": str(args.config.resolve()),
        "初始化权重": str(args.checkpoint.resolve()),
        "权重来源": source_name,
        "共享匹配张量数": len(matched),
        "Q模块新张量": rank_missing,
        "Q模块参数量": sum(parameter.numel() for parameter in rank_module.parameters()),
        "physical_batch": int(samples.shape[0]),
        "gradient_accumulation_steps": int(
            cfg.yaml_cfg["gradient_accumulation_steps"]
        ),
        "effective_batch_size": int(samples.shape[0])
        * int(cfg.yaml_cfg["gradient_accumulation_steps"]),
        "输入形状": list(samples.shape),
        "名次表形状": list(rank_module.rank_bias.shape),
        "初始开启相对禁用最大分数误差": initial_logit_error,
        "初始ZERO相对禁用最大分数误差": initial_zero_error,
        "初始开启相对禁用最大框误差": initial_box_error,
        "两步梯度": step_reports,
        "更新后开启相对ZERO平均分数变化": learned_zero_logit_delta,
        "更新后ZERO相对禁用最大分数误差": zero_disabled_logit_error,
        "更新后开启相对禁用最大框误差": learned_disabled_box_error,
        "最终层查询偏置标准差": query_slot_std,
        "优化器覆盖Q模块": True,
        "峰值分配显存MiB": torch.cuda.max_memory_allocated(device) / 2**20,
        "峰值保留显存MiB": torch.cuda.max_memory_reserved(device) / 2**20,
    }
    numeric_values = (
        report["更新后开启相对ZERO平均分数变化"],
        report["最终层查询偏置标准差"],
        report["峰值分配显存MiB"],
        report["峰值保留显存MiB"],
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise RuntimeError("Q-Rank1预检产生非有限统计")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
