"""联合实验启动前检查：C-GQ1 + M-SD2.2 + SGC2-SAM。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
CONFIG = REPO / "experiments/phase_s/s_sgc2_sam_c_plus_m_sd22_b8a4_20e.yml"
CHECKPOINT = ROOT / "weights/m_sd2_joint_coco_thermal_identity_init.pth"
REPORT = ROOT / "reports/110_sgc2_rgbt_joint/preflight.json"
sys.path.insert(0, str(REPO))

from src.core import YAMLConfig
from src.solver import BaseSolver
from tools.preflight_q_rank1 import move_targets


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def grad_summary(model, prefix):
    rows = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith(prefix) and parameter.requires_grad
    ]
    return {
        "tensor_count": len(rows),
        "with_gradient": sum(item is not None for item in rows),
        "with_nonzero_gradient": sum(
            item is not None
            and bool(torch.isfinite(item).all())
            and float(item.detach().abs().max()) > 0.0
            for item in rows
        ),
        "all_finite": all(item is None or bool(torch.isfinite(item).all()) for item in rows),
    }


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    cfg = YAMLConfig(str(CONFIG))
    model = cfg.model
    shim = BaseSolver.__new__(BaseSolver)
    shim.model = model
    shim.load_tuning_state(str(CHECKPOINT))

    assert model.rgbt_enabled
    assert model.sd2_conditioner is not None
    assert model.sd2_conditioner.thermal_contrastive
    assert model.rgbt_freeze_thermal_stream
    assert not model.decoder.hrqs_enabled
    assert model.sgc_enabled and model.sgc_supervision == "sam"
    assert model.backbone.return_idx == [1, 2, 3]
    assert model.thermal_backbone.return_idx == [1, 2, 3]
    assert cfg.val_dataloader.dataset.sam_mask_root is None
    assert len(cfg.val_dataloader.dataset) == 1820

    schedule = {}
    for epoch in range(20):
        model.set_training_epoch(epoch)
        schedule[str(epoch)] = model._sgc_supervision_scale()
    assert list(schedule.values()) == [1.0] * 10 + [0.8, 0.6, 0.4, 0.2] + [0.0] * 6

    samples, targets = next(iter(cfg.train_dataloader))
    accumulation = int(cfg.yaml_cfg["gradient_accumulation_steps"])
    assert tuple(samples.shape) == (8, 6, 512, 640)
    assert accumulation == 4
    # 数据集沿用COCO通用的 masks 字段承载SAM掩码；sam_quality区分其可信度。
    assert all("masks" in target and "sam_quality" in target for target in targets)

    device = torch.device("cuda")
    samples = samples.to(device)
    targets = move_targets(targets, device)
    model = model.to(device)
    criterion = cfg.criterion.to(device)
    optimizer = cfg.optimizer

    # SGC2在推理时应严格等价于删除该训练监督，M模块始终保持开启。
    model.eval()
    probe = samples[:2]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        joint_output = model(probe)
        model.sgc_enabled = False
        model.backbone.return_idx = [2, 3]
        model.thermal_backbone.return_idx = [2, 3]
        control_output = model(probe)
    inference_errors = {
        key: float((joint_output[key].float() - control_output[key].float()).abs().max())
        for key in ("pred_logits", "pred_boxes")
    }
    assert max(inference_errors.values()) == 0.0
    model.sgc_enabled = True
    model.backbone.return_idx = [1, 2, 3]
    model.thermal_backbone.return_idx = [1, 2, 3]
    del joint_output, control_output, probe

    # 在退场前应有SAM损失，退场后不应再生成该项。
    model.train()
    model.set_training_epoch(12)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        epoch12 = model(samples, targets)
    assert "sgc_group_loss" in epoch12 and float(epoch12["sgc_group_loss"]) > 0.0
    epoch12_aux = float(epoch12["sgc_group_loss"])
    del epoch12
    model.set_training_epoch(14)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        epoch14 = model(samples, targets)
    assert "sgc_group_loss" not in epoch14
    del epoch14

    # 单个联合批次验证梯度边界：SAM直接到RGB S8，不直接进入M；检测损失可训练M。
    model.set_training_epoch(0)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(samples, targets)
    aux = output["sgc_group_loss"]
    visible_parameter = model.backbone.stages[1].blocks[0].layers[0].conv.weight
    m_parameter = next(parameter for parameter in model.sd2_conditioner.parameters() if parameter.requires_grad)
    visible_aux_grad, m_aux_grad = torch.autograd.grad(
        aux,
        (visible_parameter, m_parameter),
        retain_graph=True,
        allow_unused=True,
    )
    assert visible_aux_grad is not None and float(visible_aux_grad.abs().max()) > 0.0
    assert m_aux_grad is None or float(m_aux_grad.abs().max()) == 0.0

    with torch.autocast("cuda", enabled=False):
        losses = criterion(
            output,
            targets,
            epoch=0,
            step=0,
            global_step=0,
            epoch_step=len(cfg.train_dataloader),
        )
        total_loss = sum(losses.values()) / accumulation
    assert bool(torch.isfinite(total_loss))
    total_loss.backward()
    visible_summary = grad_summary(model, "backbone.")
    m_summary = grad_summary(model, "sd2_conditioner.")
    thermal_summary = grad_summary(model, "thermal_backbone.")
    assert visible_summary["with_nonzero_gradient"] > 0 and visible_summary["all_finite"]
    assert m_summary["with_nonzero_gradient"] > 0 and m_summary["all_finite"]
    assert thermal_summary["with_gradient"] == 0

    result = {
        "status": "PASS",
        "config": str(CONFIG),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "initial_model_sha256": model_digest(model.cpu()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "physical_batch": 8,
        "gradient_accumulation_steps": accumulation,
        "effective_batch": 32,
        "input_shape": list(samples.shape),
        "val_image_count": len(cfg.val_dataloader.dataset),
        "sam_schedule": schedule,
        "epoch12_weighted_sgc_loss": epoch12_aux,
        "epoch14_sgc_absent": True,
        "inference_exact_without_sgc": inference_errors,
        "sgc_direct_visible_s8_gradient_max": float(visible_aux_grad.abs().max()),
        "sgc_direct_m_gradient_max": 0.0 if m_aux_grad is None else float(m_aux_grad.abs().max()),
        "full_loss": float(total_loss.detach()),
        "visible_gradient": visible_summary,
        "m_gradient": m_summary,
        "thermal_gradient": thermal_summary,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
