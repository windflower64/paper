#!/usr/bin/env python3
"""L-DQ1 batch16 structure, loss, AMP and backward preflight."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_l/l_dq1_gq1_dense_o2o_mal_b16_60e_local.yml",
    )
    parser.add_argument(
        "--reference-config",
        type=Path,
        default=ROOT / "experiments/phase_c/c_pat_gq_s32_r4.yml",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT.parent / "weights/dfine_n_coco.pth"
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT.parent / "reports/50_supervision/L_DQ1_AUGMENT_AUDIT/summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "reports/50_supervision/L_DQ1_PREFLIGHT/preflight.json",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def checkpoint_weights(path: Path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    source = state.get("ema", {}).get("module")
    source_name = "ema.module"
    if not isinstance(source, dict):
        source = state.get("model")
        source_name = "model"
    if not isinstance(source, dict):
        raise RuntimeError("COCO checkpoint has neither ema.module nor model weights")
    return source, source_name


def compatible_load(model, source):
    own = model.state_dict()
    matched = {key: value for key, value in source.items() if key in own and own[key].shape == value.shape}
    incompatible = model.load_state_dict(matched, strict=False)
    required = (
        "backbone.stages.1.blocks.0.layers.0.conv.weight",
        "encoder.input_proj.0.conv.weight",
        "decoder.dec_bbox_head.0.layers.0.weight",
    )
    absent = [key for key in required if key not in matched]
    if absent:
        raise RuntimeError(f"Missing required COCO tensors: {absent}")
    return len(matched), list(incompatible.missing_keys), list(incompatible.unexpected_keys)


def move_targets(targets, device):
    return [
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("L-DQ1 preflight requires CUDA")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if not audit.get("eligible_for_training"):
        raise RuntimeError("The 1,000-sample augmentation audit did not pass")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = args.batch
    cfg.yaml_cfg["train_dataloader"]["num_workers"] = 0
    reference_cfg = YAMLConfig(str(args.reference_config))
    reference_cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    model = cfg.model
    reference_model = reference_cfg.model
    model_shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    reference_shapes = {key: tuple(value.shape) for key, value in reference_model.state_dict().items()}
    if model_shapes != reference_shapes:
        raise RuntimeError("L-DQ1 inference model differs from frozen GQ1")
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    reference_parameters = sum(parameter.numel() for parameter in reference_model.parameters())
    del reference_model

    source, source_name = checkpoint_weights(args.checkpoint)
    matched, missing, unexpected = compatible_load(model, source)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = model.to(device).train()
    criterion = cfg.criterion.to(device).train()
    optimizer = cfg.optimizer
    dataloader = cfg.train_dataloader
    dataloader.set_epoch(4)
    samples, targets = next(iter(dataloader))
    if samples.shape[0] != args.batch:
        raise RuntimeError(f"Expected batch {args.batch}, got {samples.shape[0]}")
    target_counts = [len(target["labels"]) for target in targets]
    samples = samples.to(device)
    targets = move_targets(targets, device)
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=1024.0)
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(samples, targets=targets)
    with torch.autocast("cuda", enabled=False):
        losses = criterion(
            outputs,
            targets,
            epoch=4,
            step=0,
            global_step=0,
            epoch_step=len(dataloader),
        )
        total = sum(losses.values())
    if not torch.isfinite(total):
        raise RuntimeError(f"Non-finite loss: {float(total.detach())}")
    if "loss_mal" not in losses or any(name.startswith("loss_vfl") for name in losses):
        raise RuntimeError("MAL did not replace VFL cleanly")
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    nonfinite_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if nonfinite_gradients:
        raise RuntimeError(f"Non-finite gradients: {nonfinite_gradients[:20]}")
    gradient_tensors = sum(parameter.grad is not None for parameter in model.parameters())
    gradient_l2 = math.sqrt(
        sum(
            float(parameter.grad.detach().float().square().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)

    report = {
        "status_ascii": "pass",
        "augmentation_audit_passed": True,
        "batch_ascii": int(samples.shape[0]),
        "状态": "通过",
        "配置": str(args.config.resolve()),
        "增强审计通过": True,
        "推理结构与GQ1完全一致": True,
        "L_DQ1参数量": model_parameters,
        "GQ1参数量": reference_parameters,
        "batch": int(samples.shape[0]),
        "输入形状": list(samples.shape),
        "目标数": target_counts,
        "本batch目标总数": sum(target_counts),
        "MAL替代VFL": True,
        "总损失": float(total.detach()),
        "各损失": {name: float(value.detach()) for name, value in losses.items()},
        "全部损失有限": all(math.isfinite(float(value.detach())) for value in losses.values()),
        "输出有限": bool(
            torch.isfinite(outputs["pred_boxes"]).all()
            and torch.isfinite(outputs["pred_logits"]).all()
        ),
        "梯度张量数": gradient_tensors,
        "梯度L2": gradient_l2,
        "梯度全部有限": True,
        "优化器步完成": True,
        "COCO权重来源": source_name,
        "匹配权重张量数": matched,
        "新初始化张量数": len(missing),
        "unexpected": unexpected,
        "显存峰值MiB": torch.cuda.max_memory_allocated(device) / 2**20,
        "设备": torch.cuda.get_device_name(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
