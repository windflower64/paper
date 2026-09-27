"""M-A1/P0结构、因果与显存预检；通过前禁止正式训练。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def clone_diagnostics(module):
    return {
        key: value.detach().float().cpu().clone()
        for key, value in module.last_diagnostics.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-batch", type=int, default=16)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("M-A1/P0要求CUDA环境")

    geometry = json.loads(args.geometry_report.read_text(encoding="utf-8"))
    geometry_error = geometry[
        "train_fitted_visible_to_thermal_on_test_center_error"
    ]
    if geometry_error["p50"] >= 0.05 or geometry_error["within_0p10"] < 0.80:
        raise RuntimeError("训练集仿射先验在test上覆盖不足，不允许启动M-A1")

    cfg = YAMLConfig(str(args.config.resolve()))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["ema"]["module"], strict=True)

    decoder = model.decoder
    calibrator = decoder.sdtec_aligned_calibrator
    if decoder.sdtec_fusion_mode != "soft_aligned" or calibrator is None:
        raise RuntimeError("当前配置没有构造M-A1软对齐读取器")
    if calibrator.num_levels != 2 or calibrator.num_points != 4:
        raise RuntimeError("M-A1尺度数或每尺度采样点数不符合P0协议")
    if calibrator.protected_tail_queries != 50:
        raise RuntimeError("M-A1没有保护最后50个HRQS查询")

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    invalid_trainable = [
        name for name in trainable if not name.startswith("decoder.sdtec_")
    ]
    if invalid_trainable:
        raise RuntimeError(f"存在越界可训练参数：{invalid_trainable[:10]}")
    final_key = "decoder.sdtec_aligned_calibrator.delta_head.2.weight"
    if torch.count_nonzero(trainable[final_key].detach()):
        raise RuntimeError("M-A1最终投影没有保持严格零初始化")

    frozen_before = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key not in trainable
    }
    device = torch.device("cuda")
    model.to(device)
    criterion = cfg.criterion.to(device)
    optimizer = cfg.optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=128.0)
    loader = cfg.train_dataloader
    samples, targets = next(iter(loader))
    samples = samples.to(device)
    targets = move_targets(targets, device)
    if samples.shape[0] != args.expected_batch or samples.shape[1] != 6:
        raise RuntimeError(
            f"M-A1输入应为[{args.expected_batch},6,H,W]，"
            f"实际{tuple(samples.shape)}"
        )

    model.eval()
    decoder.set_training_epoch(20)
    model.rgbt_thermal_intervention = "normal"
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        initial_normal = model(samples)
    initial_normal_diag = clone_diagnostics(calibrator)
    initial_normal_logits = initial_normal["pred_logits"].detach().cpu().clone()
    initial_normal_boxes = initial_normal["pred_boxes"].detach().cpu().clone()
    del initial_normal
    torch.cuda.empty_cache()
    model.rgbt_thermal_intervention = "zero_content_valid"
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        initial_zero = model(samples)
    initial_zero_diag = clone_diagnostics(calibrator)
    initial_zero_logits = initial_zero["pred_logits"].detach().cpu().clone()
    initial_zero_boxes = initial_zero["pred_boxes"].detach().cpu().clone()
    del initial_zero
    torch.cuda.empty_cache()

    if not torch.equal(initial_normal_logits, initial_zero_logits):
        raise RuntimeError("零初始化M-A1改变了C+D分类结果")
    if not torch.equal(initial_normal_boxes, initial_zero_boxes):
        raise RuntimeError("零初始化M-A1改变了C+D预测框")
    if torch.count_nonzero(initial_normal_diag["delta"]):
        raise RuntimeError("零初始化M-A1在正常红外下产生了非零校准")
    if torch.count_nonzero(initial_zero_diag["delta"]):
        raise RuntimeError("零内容红外在初始化时产生了非零校准")
    if not bool(initial_normal_diag["thermal_content_mask"].all()):
        raise RuntimeError("正常训练批次中发现了意外的全零红外图像")
    if bool(initial_zero_diag["thermal_content_mask"].any()):
        raise RuntimeError("零内容红外没有被结构掩码识别")
    if not torch.isfinite(initial_normal_diag["mapped_centres"]).all():
        raise RuntimeError("仿射映射中心包含非有限值")
    if initial_normal_diag["max_abs_residual_offset"].item() > (
        calibrator.search_radius + 1e-6
    ):
        raise RuntimeError("M-A1初始搜索偏移超出硬边界")

    records = []
    torch.cuda.reset_peak_memory_stats(device)
    model.rgbt_thermal_intervention = "normal"
    for step in range(4):
        decoder.set_training_epoch(2)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(samples, targets=targets)
        with torch.autocast(device_type="cuda", enabled=False):
            losses = criterion(
                outputs,
                targets,
                epoch=2,
                step=step,
                global_step=step,
                epoch_step=len(loader),
            )
            total_loss = sum(losses.values())
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"M-A1第{step}步损失非有限")
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

        delta = outputs["ma1_logit_delta"].detach()
        if torch.count_nonzero(delta[:, -50:]):
            raise RuntimeError("M-A1修改了受保护的HRQS分类结果")
        nonzero_gradients = [
            name
            for name, parameter in trainable.items()
            if parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad.detach()))
        ]
        scaler.step(optimizer)
        scaler.update()
        records.append(
            {
                "step": step,
                "loss": float(total_loss.detach().cpu()),
                "loss_components": {
                    name: float(value.detach().cpu())
                    for name, value in losses.items()
                },
                "ordinary_mean_abs_logit_delta": float(
                    delta[:, :-50].abs().mean().cpu()
                ),
                "gate_mean": float(outputs["ma1_gate"].detach().mean().cpu()),
                "offset_norm_mean": float(
                    outputs["ma1_effective_offset"]
                    .detach()
                    .norm(dim=-1)
                    .mean()
                    .cpu()
                ),
                "attention_entropy_mean": float(
                    outputs["ma1_attention_entropy"].detach().mean().cpu()
                ),
                "trainable_with_nonzero_gradient": len(nonzero_gradients),
                "nonzero_gradient_names": nonzero_gradients,
            }
        )

    model.eval()
    decoder.set_training_epoch(20)
    model.rgbt_thermal_intervention = "normal"
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        opened_normal = model(samples)
    opened_normal_diag = clone_diagnostics(calibrator)
    opened_normal_logits = opened_normal["pred_logits"].detach().cpu().clone()
    opened_normal_boxes = opened_normal["pred_boxes"].detach().cpu().clone()
    del opened_normal
    torch.cuda.empty_cache()
    model.rgbt_thermal_intervention = "zero_content_valid"
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        opened_zero = model(samples)
    opened_zero_diag = clone_diagnostics(calibrator)
    opened_zero_logits = opened_zero["pred_logits"].detach().cpu().clone()
    opened_zero_boxes = opened_zero["pred_boxes"].detach().cpu().clone()
    del opened_zero
    torch.cuda.empty_cache()

    if not torch.equal(opened_normal_boxes, opened_zero_boxes):
        raise RuntimeError("打开后的M-A1改变了RGB预测框支路")
    if torch.equal(opened_normal_logits, opened_zero_logits):
        raise RuntimeError("四步训练后M-A1仍未因果使用红外内容")
    if torch.count_nonzero(opened_zero_diag["delta"]):
        raise RuntimeError("打开后零红外仍产生分类校准，存在固定偏置捷径")
    if torch.count_nonzero(opened_normal_diag["delta"][:, -50:]):
        raise RuntimeError("打开后M-A1修改了受保护的HRQS分类结果")
    if opened_normal_diag["max_abs_residual_offset"].item() > (
        calibrator.search_radius + 1e-6
    ):
        raise RuntimeError("训练后的M-A1搜索偏移超出硬边界")
    affine_error = (
        opened_normal_diag["effective_affine"]
        - calibrator.affine_base.detach().float().cpu()
    ).abs().max().item()
    if affine_error > calibrator.affine_delta_scale + 1e-6:
        raise RuntimeError("训练后的仿射残差超出硬边界")

    model.cpu()
    frozen_changes = []
    for key, value in model.state_dict().items():
        if key in frozen_before and not torch.equal(
            value.detach().cpu(), frozen_before[key]
        ):
            frozen_changes.append(key)
    if frozen_changes:
        raise RuntimeError(f"预检修改了冻结张量：{frozen_changes[:10]}")

    final_nonzero_gradients = records[-1]["nonzero_gradient_names"]
    if final_key not in final_nonzero_gradients:
        raise RuntimeError("M-A1最终校准层没有收到检测梯度")
    critical_gradient_suffixes = (
        "affine_delta",
        "sampling_offsets.weight",
        "value_proj.weight",
    )
    missing_critical = [
        suffix
        for suffix in critical_gradient_suffixes
        if not any(name.endswith(suffix) for name in final_nonzero_gradients)
    ]
    if missing_critical:
        raise RuntimeError(f"软对齐关键参数没有收到梯度：{missing_critical}")

    result = {
        "status": "PASS",
        "protocol": "MA1-P0",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "geometry_report": str(args.geometry_report.resolve()),
        "seed": args.seed,
        "batch_size": int(loader.batch_size),
        "input_shape": list(samples.shape),
        "trainable_tensor_count": len(trainable),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in trainable.values()
        ),
        "protected_hrqs_queries": calibrator.protected_tail_queries,
        "alignment_levels": calibrator.num_levels,
        "sampling_points_per_level": calibrator.num_points,
        "search_radius": calibrator.search_radius,
        "affine_delta_scale": calibrator.affine_delta_scale,
        "max_logit_delta": calibrator.max_logit_delta,
        "geometry_test_median_error": geometry_error["p50"],
        "geometry_test_within_search_radius": geometry_error["within_0p10"],
        "initial_normal_equals_zero_logits": True,
        "initial_normal_equals_zero_boxes": True,
        "initial_zero_content_delta_exact_zero": True,
        "opened_normal_equals_zero_boxes": True,
        "opened_normal_differs_from_zero_logits": True,
        "opened_zero_content_delta_exact_zero": True,
        "opened_mean_abs_logit_delta": float(
            opened_normal_diag["delta"][:, :-50].abs().mean()
        ),
        "opened_max_abs_residual_offset": float(
            opened_normal_diag["max_abs_residual_offset"]
        ),
        "opened_effective_affine": opened_normal_diag[
            "effective_affine"
        ].tolist(),
        "steps": records,
        "frozen_tensor_changes": frozen_changes,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
