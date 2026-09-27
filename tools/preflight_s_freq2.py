#!/usr/bin/env python3
"""S-FREQ2训练前硬门：公开FreeKD公式在D-FINE上的工程与梯度预检。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from src.misc import dist_utils
from src.solver.sam_frequency_distillation import SAMFrequencyDistiller


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "experiments/phase_s/s_freq2_freekd_public_s8_b32_60e_local.yml",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent
        / "reports/22_sam_frequency_detail/S_FREQ2_FREEKD_PUBLIC_S8/preflight_batch32.json",
    )
    return parser.parse_args()


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def gradient_list(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return [gradient for gradient in gradients if gradient is not None]


def gradients_l2(gradients):
    if not gradients:
        return 0.0
    return float(torch.stack([gradient.float().square().sum() for gradient in gradients]).sum().sqrt())


def gradient_cosine(left, right):
    if len(left) != len(right) or not left:
        return None
    numerator = sum((a.float() * b.float()).sum() for a, b in zip(left, right))
    left_norm = torch.stack([value.float().square().sum() for value in left]).sum().sqrt()
    right_norm = torch.stack([value.float().square().sum() for value in right]).sum().sqrt()
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        return None
    return float(numerator / (left_norm * right_norm))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("S-FREQ2 preflight requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = args.batch_size
    model = cfg.model.to(device).train()
    criterion = cfg.criterion.to(device).train()
    optimizer = cfg.optimizer
    loader = cfg.train_dataloader

    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = torch.cuda.get_rng_state_all()
    student = dist_utils.de_parallel(model)
    distiller = SAMFrequencyDistiller(
        student, cfg.yaml_cfg["frequency_distillation"], device
    )
    cpu_rng_equal = torch.equal(cpu_rng_before, torch.random.get_rng_state())
    cuda_rng_equal = all(
        torch.equal(before, after)
        for before, after in zip(cuda_rng_before, torch.cuda.get_rng_state_all())
    )

    samples, targets = next(iter(loader))
    samples = samples.to(device)
    targets = move_targets(targets, device)
    torch.cuda.reset_peak_memory_stats()

    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(samples, targets=targets)
    with torch.autocast("cuda", enabled=False):
        losses = criterion(outputs, targets)
        detection_loss = sum(losses.values())
        frequency_loss = distiller(samples, targets)

    s8 = distiller.student_cache["s8"]
    detection_s8_grad = torch.autograd.grad(
        detection_loss, s8, retain_graph=True
    )[0]
    frequency_s8_grad = torch.autograd.grad(
        frequency_loss, s8, retain_graph=True
    )[0]
    detection_l2 = float(detection_s8_grad.float().norm())
    frequency_l2 = float(frequency_s8_grad.float().norm())
    shared_ratio = frequency_l2 / max(detection_l2, 1e-12)
    shared_cosine = float(
        F.cosine_similarity(
            detection_s8_grad.float().flatten(), frequency_s8_grad.float().flatten(), dim=0
        )
    )

    # S8在stage2入口截取，因此频率损失与检测损失真正共享的是stage0/1。
    student_stage_parameters = [
        parameter
        for parameter in student.backbone.stages[0:2].parameters()
        if parameter.requires_grad
    ]
    detection_parameter_gradients = gradient_list(
        detection_loss, student_stage_parameters, retain_graph=True
    )
    frequency_parameter_gradients = gradient_list(
        frequency_loss, student_stage_parameters, retain_graph=True
    )
    detection_parameter_l2 = gradients_l2(detection_parameter_gradients)
    frequency_parameter_l2 = gradients_l2(frequency_parameter_gradients)
    frequency_parameter_ratio = frequency_parameter_l2 / max(
        detection_parameter_l2, 1e-12
    )
    frequency_parameter_cosine = gradient_cosine(
        frequency_parameter_gradients, detection_parameter_gradients
    )

    teacher_has_gradient = any(
        parameter.grad is not None for parameter in distiller.teacher.parameters()
    )
    prompt_requires_grad = bool(distiller.prompt.requires_grad)
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    optimizer_covers_student = trainable_ids == optimizer_ids
    safe_gradient_limit = 0.10 if (
        frequency_parameter_cosine is not None and frequency_parameter_cosine < 0
    ) else 0.30

    model.eval()
    with torch.inference_mode():
        before = model(samples)
        after = model(samples)
    box_error = float((before["pred_boxes"] - after["pred_boxes"]).abs().max())
    logit_error = float((before["pred_logits"] - after["pred_logits"]).abs().max())
    peak_memory_mib = torch.cuda.max_memory_allocated() / (1024**2)

    result = {
        "protocol": {
            "config": str(args.config),
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": torch.cuda.get_device_name(0),
            "distiller": distiller.describe(),
        },
        "checks": {
            "finite_positive_frequency_loss": bool(
                torch.isfinite(frequency_loss) and float(frequency_loss) > 0
            ),
            "frequency_gradient_reaches_student_s8": frequency_l2 > 0,
            "frequency_gradient_reaches_student_parameters": frequency_parameter_l2 > 0,
            "teacher_has_no_gradient": not teacher_has_gradient,
            "prompt_is_frozen": not prompt_requires_grad,
            "optimizer_exactly_covers_student": optimizer_covers_student,
            "teacher_construction_preserves_cpu_rng": cpu_rng_equal,
            "teacher_construction_preserves_cuda_rng": cuda_rng_equal,
            "eval_predictions_unchanged": box_error == 0.0 and logit_error == 0.0,
            "batch_is_finite": bool(
                torch.isfinite(detection_loss) and torch.isfinite(frequency_loss)
            ),
            "shared_parameter_gradient_not_dead": frequency_parameter_ratio >= 0.02,
            "shared_parameter_gradient_is_safe": (
                frequency_parameter_ratio <= safe_gradient_limit
            ),
        },
        "measurements": {
            "detection_loss": float(detection_loss),
            "frequency_loss": float(frequency_loss),
            "frequency_raw_stats": distiller.last_stats,
            "detection_s8_gradient_l2": detection_l2,
            "frequency_s8_gradient_l2": frequency_l2,
            "frequency_detection_s8_gradient_ratio": shared_ratio,
            "frequency_detection_s8_gradient_cosine": shared_cosine,
            "frequency_student_parameter_gradient_l2": frequency_parameter_l2,
            "detection_shared_parameter_gradient_l2": detection_parameter_l2,
            "frequency_detection_shared_parameter_gradient_ratio": frequency_parameter_ratio,
            "frequency_detection_shared_parameter_gradient_cosine": frequency_parameter_cosine,
            "shared_parameter_gradient_safety_limit": safe_gradient_limit,
            "box_identity_error": box_error,
            "logit_identity_error": logit_error,
            "peak_memory_mib": peak_memory_mib,
            "trainable_student_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "extra_trainable_parameter_count": 0,
        },
    }
    result["pass"] = all(result["checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    distiller.close()
    if not result["pass"]:
        raise RuntimeError("S-FREQ2 preflight failed")


if __name__ == "__main__":
    main()
