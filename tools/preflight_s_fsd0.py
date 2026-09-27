#!/usr/bin/env python3
"""FSD0公式、初始化、因果模式与batch16训练预检。"""

from __future__ import annotations

import argparse
import gc
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
        default=ROOT / "experiments/phase_s/s_fsd0_direct_s8_s16_b16_60e_local.yml",
    )
    parser.add_argument(
        "--control-config",
        type=Path,
        default=ROOT
        / "experiments/phase_s/s_fsd0_std_reinit_s8_s16_b16_60e_local.yml",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT.parent / "weights/dfine_n_coco.pth"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "reports/26_fsd_down/S_FSD0/preflight.json",
    )
    parser.add_argument("--expected-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--amp-init-scale", type=float, default=1024.0)
    return parser.parse_args()


def checkpoint_weights(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("ema", {}).get("module")
    source_name = "ema.module"
    if not isinstance(source, dict):
        source = checkpoint.get("model")
        source_name = "model"
    if not isinstance(source, dict):
        raise RuntimeError(f"checkpoint中没有模型权重：{path}")
    return source, source_name


def load_compatible(model, source):
    own = model.state_dict()
    matched = {
        key: value
        for key, value in source.items()
        if key in own and own[key].shape == value.shape
    }
    required = (
        "backbone.stages.1.blocks.0.layers.0.conv.weight",
        "encoder.input_proj.0.conv.weight",
        "decoder.dec_bbox_head.0.layers.0.weight",
    )
    missing_required = [key for key in required if key not in matched]
    if missing_required:
        raise RuntimeError(f"COCO checkpoint缺少关键共享张量：{missing_required}")
    incompatible = model.load_state_dict(matched, strict=False)
    return {
        "matched": matched,
        "missing": list(incompatible.missing_keys),
        "source_only": [key for key in source if key not in matched],
    }


def make_model(config_path, source, seed):
    torch.manual_seed(seed)
    cfg = YAMLConfig(str(config_path))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    load_report = load_compatible(model, source)
    return cfg, model, load_report


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def gradient_l2(parameters):
    terms = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.0


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FSD0 batch16预检需要CUDA")
    source, source_name = checkpoint_weights(args.checkpoint)
    cfg, model, fsd_load = make_model(args.config, source, args.seed)
    _, control_model, control_load = make_model(args.control_config, source, args.seed)

    fsd_prefix = "backbone.stages.2.downsample."
    old_prefix = "backbone.stages.2.downsample."
    fsd_missing = [key for key in fsd_load["missing"] if key.startswith(fsd_prefix)]
    control_missing = [
        key for key in control_load["missing"] if key.startswith(old_prefix)
    ]
    if not fsd_missing or not control_missing:
        raise RuntimeError("FSD或STD-REINIT没有真正处于重初始化状态")
    old_source_keys = [
        key
        for key in source
        if key.startswith(old_prefix)
        and (".conv." in key or ".bn." in key or ".lab." in key)
    ]
    if not old_source_keys:
        raise RuntimeError("COCO checkpoint中未找到原S8->S16下采样权重")
    if any(key in fsd_load["matched"] for key in old_source_keys):
        raise RuntimeError("FSD错误继承了被替换的标准下采样权重")

    fsd_state = model.state_dict()
    control_state = control_model.state_dict()
    shared_keys = [
        key
        for key in fsd_state
        if key in control_state
        and fsd_state[key].shape == control_state[key].shape
        and not key.startswith(fsd_prefix)
    ]
    shared_errors = []
    for key in shared_keys:
        left, right = fsd_state[key], control_state[key]
        if left.is_floating_point() or left.is_complex():
            shared_errors.append(float((left - right).abs().max()))
        else:
            shared_errors.append(0.0 if torch.equal(left, right) else 1.0)
    shared_max_error = max(shared_errors)
    if shared_max_error != 0.0:
        raise RuntimeError(f"配对模型共享权重不一致：max_error={shared_max_error}")

    fsd = model.backbone.stages[2].downsample
    torch.manual_seed(args.seed + 1)
    feature = torch.randn(2, 256, 64, 80)
    bands = fsd.haar_dwt(feature)
    dwt = torch.cat(bands, dim=1)
    energy_error = abs(
        float(dwt.float().square().sum() / feature.float().square().sum()) - 1.0
    )
    if energy_error > 1e-5:
        raise RuntimeError(f"Haar能量不守恒：relative_error={energy_error}")

    full_hf_energy = float(dwt[:, 256:].float().square().sum())
    mode_outputs = {}
    fsd.eval()
    with torch.no_grad():
        for mode in sorted(fsd.VALID_FREQUENCY_MODES):
            fsd.frequency_mode = mode
            mode_outputs[mode] = fsd(feature).cpu()
    fsd.frequency_mode = "full"
    mode_mean_abs_delta = {
        mode: float((output - mode_outputs["full"]).abs().mean())
        for mode, output in mode_outputs.items()
        if mode != "full"
    }
    if any(value <= 0.0 for value in mode_mean_abs_delta.values()):
        raise RuntimeError(f"存在无效因果模式：{mode_mean_abs_delta}")
    intervention_energy_error = {}
    for mode in ("shift_hf", "phase_permute"):
        fsd.frequency_mode = mode
        intervened = fsd._intervene_frequency(bands)
        intervention_energy_error[mode] = abs(
            float(intervened[:, 256:].float().square().sum() / full_hf_energy) - 1.0
        )
    fsd.frequency_mode = "full"
    if max(intervention_energy_error.values()) > 1e-6:
        raise RuntimeError(f"高频干预没有保持能量：{intervention_energy_error}")

    fsd_parameters = list(fsd.parameters())
    optimizer_parameter_ids = {
        id(parameter)
        for group in cfg.optimizer.param_groups
        for parameter in group["params"]
    }
    if not set(map(id, fsd_parameters)).issubset(optimizer_parameter_ids):
        raise RuntimeError("优化器遗漏FSD参数")

    del control_model, control_state
    gc.collect()
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = model.to(device).train()
    criterion = cfg.criterion.to(device).train()
    optimizer = cfg.optimizer
    samples, targets = next(iter(cfg.train_dataloader))
    if samples.shape[0] != args.expected_batch:
        raise RuntimeError(
            f"训练batch错误：期望{args.expected_batch}，实际{samples.shape[0]}"
        )
    samples = samples.to(device)
    targets = move_targets(targets, device)
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler(
        enabled=True, init_scale=float(args.amp_init_scale)
    )
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(samples, targets=targets)
    with torch.autocast("cuda", enabled=False):
        losses = criterion(
            outputs,
            targets,
            epoch=0,
            step=0,
            global_step=0,
            epoch_step=len(cfg.train_dataloader),
        )
        total_loss = sum(losses.values())
    if not torch.isfinite(total_loss):
        raise RuntimeError(f"总损失非有限值：{float(total_loss.detach())}")
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    nonfinite_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if nonfinite_gradients:
        raise RuntimeError(f"出现非有限梯度：{nonfinite_gradients[:20]}")
    fsd_gradient = gradient_l2(model.backbone.stages[2].downsample.parameters())
    if fsd_gradient <= 0.0:
        raise RuntimeError("FSD没有收到检测梯度")
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)

    fsd_module = model.backbone.stages[2].downsample
    report = {
        "状态": "通过",
        "方案": "FSD0公式级迁移",
        "配置": str(args.config),
        "配对控制配置": str(args.control_config),
        "COCO权重": str(args.checkpoint),
        "权重来源": source_name,
        "FSD匹配张量数": len(fsd_load["matched"]),
        "FSD新初始化张量": fsd_missing,
        "标准重初始化控制新张量": control_missing,
        "原下采样checkpoint张量数": len(old_source_keys),
        "共享张量数": len(shared_keys),
        "共享张量最大误差": shared_max_error,
        "Haar能量相对误差": energy_error,
        "等能量高频干预误差": intervention_energy_error,
        "各模式相对FULL平均绝对变化": mode_mean_abs_delta,
        "FSD参数量": sum(parameter.numel() for parameter in fsd_parameters),
        "优化器覆盖FSD": True,
        "训练batch": int(samples.shape[0]),
        "输入形状": list(samples.shape),
        "总损失": float(total_loss.detach()),
        "各损失": {name: float(value.detach()) for name, value in losses.items()},
        "所有损失有限": all(
            math.isfinite(float(value.detach())) for value in losses.values()
        ),
        "FSD梯度L2": fsd_gradient,
        "优化器步完成": True,
        "AMP初始scale": float(args.amp_init_scale),
        "AMP更新后scale": float(scaler.get_scale()),
        "显存峰值MiB": torch.cuda.max_memory_allocated(device) / 2**20,
        "显存保留峰值MiB": torch.cuda.max_memory_reserved(device) / 2**20,
        "设备": torch.cuda.get_device_name(device),
        "初始rho": float(fsd_module.last_rho),
        "初始频率门均值": float(fsd_module.last_frequency_gate_mean),
        "初始空间门均值": float(fsd_module.last_spatial_gate_mean),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
