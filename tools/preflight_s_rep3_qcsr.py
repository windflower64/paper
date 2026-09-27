#!/usr/bin/env python3
"""Strict batch-16 preflight for REP3 query-conditioned cross-scale retention."""

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
        default=ROOT / "experiments/phase_s/s_rep3_qcsr_w025_ft6_lr02_local.yml",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=ROOT / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT.parent / "outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "reports/21_public_reproduction/S_REP3_QCSR/preflight_batch16.json",
    )
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--min-shift-difference", type=float, default=1e-6)
    parser.add_argument("--min-transfer-detection-cosine", type=float, default=-0.20)
    parser.add_argument("--max-transfer-stage2-gradient", type=float, default=-1.0)
    parser.add_argument("--max-transfer-stage3-gradient", type=float, default=-1.0)
    parser.add_argument("--max-amplitude-invariance-error", type=float, default=1e-5)
    parser.add_argument("--replay-diagnosis", type=Path)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_weights(checkpoint):
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"], "ema.module"
    model = checkpoint.get("model")
    if isinstance(model, dict):
        return model, "model"
    return checkpoint, "root"


def load_matching(model, weights):
    state = model.state_dict()
    matched = {
        key: value
        for key, value in weights.items()
        if key in state and state[key].shape == value.shape
    }
    model.load_state_dict(matched, strict=False)
    return sorted(set(state).difference(matched))


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def max_shared_error(reference, candidate):
    candidate_state = candidate.state_dict()
    error = 0.0
    missing = []
    for key, value in reference.state_dict().items():
        if key not in candidate_state:
            missing.append(key)
            continue
        other = candidate_state[key]
        if value.dtype == torch.bool:
            current = 0.0 if torch.equal(value, other) else 1.0
        else:
            current = float((value - other).abs().max())
        error = max(error, current)
    return error, missing


def grad_list(loss, parameters, retain_graph=True):
    parameters = list(parameters)
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return [
        gradient.detach().float() if gradient is not None else None
        for gradient in gradients
    ]


def grad_norm(gradients):
    squares = [gradient.square().sum() for gradient in gradients if gradient is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def grad_cosine(left, right):
    pairs = [
        (left_grad, right_grad)
        for left_grad, right_grad in zip(left, right)
        if left_grad is not None and right_grad is not None
    ]
    if not pairs:
        return float("nan")
    dot = torch.stack([(a * b).sum() for a, b in pairs]).sum()
    left_norm = torch.stack([a.square().sum() for a, _ in pairs]).sum().sqrt()
    right_norm = torch.stack([b.square().sum() for _, b in pairs]).sum().sqrt()
    return float(dot / (left_norm * right_norm).clamp_min(1e-12))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("REP3-QCSR preflight requires CUDA")
    device = torch.device("cuda")

    seed_everything(args.seed)
    baseline_cfg = YAMLConfig(str(args.baseline_config))
    baseline_model = baseline_cfg.model
    seed_everything(args.seed)
    cfg = YAMLConfig(str(args.config))
    model = cfg.model
    initialization_error, initialization_missing = max_shared_error(baseline_model, model)
    if initialization_missing or initialization_error != 0.0:
        raise RuntimeError(
            "QCSR changed A00 shared initialization: "
            f"missing={initialization_missing}, max_error={initialization_error}"
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights, weight_source = checkpoint_weights(checkpoint)
    baseline_missing = load_matching(baseline_model, weights)
    qcsr_missing = load_matching(model, weights)
    allowed_missing = {
        key for key in model.state_dict() if key.startswith("decoder.qcsr_")
    }
    if set(qcsr_missing) != allowed_missing:
        raise RuntimeError(
            "A00 transfer left unexpected QCSR parameters unmatched: "
            f"missing={qcsr_missing}, allowed={sorted(allowed_missing)}"
        )
    checkpoint_error, checkpoint_missing = max_shared_error(baseline_model, model)
    if baseline_missing or checkpoint_missing or checkpoint_error != 0.0:
        raise RuntimeError(
            "QCSR and A00 differ after checkpoint transfer: "
            f"baseline_missing={baseline_missing}, shared_missing={checkpoint_missing}, "
            f"max_error={checkpoint_error}"
        )

    samples, targets = next(iter(cfg.train_dataloader))
    if samples.shape[0] != 16:
        raise RuntimeError(f"REP3 config must use batch16, got {samples.shape[0]}")
    valid_samples = sum(
        int("masks" in target and bool(target["masks"].any())) for target in targets
    )
    if valid_samples <= 0:
        raise RuntimeError("The first REP3 batch has no accepted SAM masks")
    train_samples = samples.to(device)
    train_targets = move_targets(targets, device)

    baseline_model = baseline_model.to(device).eval()
    model = model.to(device).eval()
    with torch.no_grad():
        baseline_output = baseline_model(train_samples[:1])
        qcsr_output = model(train_samples[:1])
    identity_box_error = float(
        (baseline_output["pred_boxes"] - qcsr_output["pred_boxes"]).abs().max()
    )
    identity_logit_error = float(
        (baseline_output["pred_logits"] - qcsr_output["pred_logits"]).abs().max()
    )
    if identity_box_error != 0.0 or identity_logit_error != 0.0:
        raise RuntimeError(
            f"Training-only QCSR changed inference: boxes={identity_box_error}, "
            f"logits={identity_logit_error}"
        )
    del baseline_model, baseline_output, qcsr_output
    torch.cuda.empty_cache()

    criterion = cfg.criterion.to(device).train()
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    outputs = model(train_samples, targets=train_targets)
    expected_shapes = {
        "qcsr_query_embeddings": (16, 300, 64),
        "qcsr_teacher_pixels": (16, 64, 64, 80),
        "qcsr_student_pixels": (16, 64, 32, 40),
    }
    for key, shape in expected_shapes.items():
        if key not in outputs or tuple(outputs[key].shape) != shape:
            actual = None if key not in outputs else tuple(outputs[key].shape)
            raise RuntimeError(f"Unexpected {key} shape: {actual}, expected {shape}")

    match_input = {key: value for key, value in outputs.items() if "aux" not in key}
    indices = criterion.matcher(match_input, train_targets)["indices"]
    matched_sam_pairs = 0
    for batch_index, (source_indices, target_indices) in enumerate(indices):
        if source_indices.unique().numel() != source_indices.numel():
            raise RuntimeError("One QCSR detector query matched multiple targets")
        if target_indices.unique().numel() != target_indices.numel():
            raise RuntimeError("One QCSR target matched multiple detector queries")
        masks = train_targets[batch_index]["masks"]
        matched_sam_pairs += sum(bool(masks[index].any()) for index in target_indices.tolist())
    if matched_sam_pairs != valid_samples:
        raise RuntimeError(
            "Accepted SAM samples were not paired one-to-one: "
            f"valid={valid_samples}, matched={matched_sam_pairs}"
        )

    raw_teacher, raw_transfer = criterion._qcsr_losses(
        outputs["qcsr_query_embeddings"],
        outputs["qcsr_teacher_pixels"],
        outputs["qcsr_student_pixels"],
        train_targets,
        indices,
    )
    shifted_targets = []
    for target in train_targets:
        shifted = dict(target)
        shifted["masks"] = torch.roll(
            target["masks"], shifts=target["masks"].shape[-1] // 2, dims=-1
        )
        shifted_targets.append(shifted)
    shifted_teacher, shifted_transfer = criterion._qcsr_losses(
        outputs["qcsr_query_embeddings"],
        outputs["qcsr_teacher_pixels"],
        outputs["qcsr_student_pixels"],
        shifted_targets,
        indices,
    )
    shift_difference = float((raw_transfer - shifted_transfer).abs().detach())
    if shift_difference <= args.min_shift_difference:
        raise RuntimeError(
            "QCSR transfer is insensitive to SAM location: "
            f"difference={shift_difference}"
        )

    _, amplified_transfer = criterion._qcsr_losses(
        outputs["qcsr_query_embeddings"],
        outputs["qcsr_teacher_pixels"] * 10.0,
        outputs["qcsr_student_pixels"],
        train_targets,
        indices,
    )
    amplitude_invariance_error = float(
        (raw_transfer - amplified_transfer).abs().detach()
    )
    if criterion.qcsr_transfer_mode == "shape_correlation":
        if not 0.0 <= float(raw_transfer.detach()) <= 1.0:
            raise RuntimeError(
                f"Shape correlation loss escaped [0, 1]: {float(raw_transfer.detach())}"
            )
        if amplitude_invariance_error > args.max_amplitude_invariance_error:
            raise RuntimeError(
                "Shape transfer changed when teacher amplitude was multiplied by 10: "
                f"error={amplitude_invariance_error}"
            )

    losses = criterion(outputs, train_targets, epoch=0)
    weighted_teacher = losses.get("loss_qcsr_teacher")
    weighted_transfer = losses.get("loss_qcsr_transfer")
    if weighted_teacher is None or weighted_transfer is None:
        raise RuntimeError("Criterion omitted QCSR losses")
    if not torch.allclose(
        weighted_teacher,
        criterion.qcsr_teacher_weight * raw_teacher,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError("QCSR teacher weight was applied incorrectly")
    if not torch.allclose(
        weighted_transfer,
        criterion.qcsr_transfer_weight * raw_transfer,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError("QCSR transfer weight was applied incorrectly")
    if min(float(weighted_teacher.detach()), float(weighted_transfer.detach())) <= 0:
        raise RuntimeError("QCSR produced a non-positive loss")

    query_head = list(model.decoder.qcsr_query_proj.parameters())
    teacher_head = list(model.decoder.qcsr_teacher_proj.parameters())
    student_head = list(model.decoder.qcsr_student_proj.parameters())
    stage2 = list(model.backbone.stages[1].parameters())
    stage3 = list(model.backbone.stages[2].parameters())
    final_decoder = list(model.decoder.decoder.layers[-1].parameters())
    optimizer_ids = {
        id(parameter)
        for group in cfg.optimizer.param_groups
        for parameter in group["params"]
    }
    all_heads = query_head + teacher_head + student_head
    if not set(map(id, all_heads)).issubset(optimizer_ids):
        raise RuntimeError("Optimizer omitted QCSR head parameters")

    groups = query_head + teacher_head + student_head + stage2 + stage3 + final_decoder
    teacher_gradients = grad_list(weighted_teacher, groups, retain_graph=True)
    transfer_gradients = grad_list(weighted_transfer, groups, retain_graph=True)
    sizes = [len(query_head), len(teacher_head), len(student_head), len(stage2), len(stage3)]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    offsets.append(len(groups))

    def group_norm(gradients, index):
        return grad_norm(gradients[offsets[index] : offsets[index + 1]])

    teacher_norms = {
        "query_head": group_norm(teacher_gradients, 0),
        "teacher_head": group_norm(teacher_gradients, 1),
        "student_head": group_norm(teacher_gradients, 2),
        "stage2_s8": group_norm(teacher_gradients, 3),
        "stage3_s16": group_norm(teacher_gradients, 4),
        "final_decoder": group_norm(teacher_gradients, 5),
    }
    transfer_norms = {
        "query_head": group_norm(transfer_gradients, 0),
        "teacher_head": group_norm(transfer_gradients, 1),
        "student_head": group_norm(transfer_gradients, 2),
        "stage2_s8": group_norm(transfer_gradients, 3),
        "stage3_s16": group_norm(transfer_gradients, 4),
        "final_decoder": group_norm(transfer_gradients, 5),
    }
    if min(teacher_norms["query_head"], teacher_norms["teacher_head"]) <= 0:
        raise RuntimeError(f"Teacher loss missed private heads: {teacher_norms}")
    if any(teacher_norms[key] != 0.0 for key in ("student_head", "stage2_s8", "stage3_s16", "final_decoder")):
        raise RuntimeError(f"Teacher SAM gradient leaked into detector: {teacher_norms}")
    if min(
        transfer_norms["student_head"],
        transfer_norms["stage2_s8"],
        transfer_norms["stage3_s16"],
    ) <= 0:
        raise RuntimeError(f"Transfer missed the S8->S16 student path: {transfer_norms}")
    if any(transfer_norms[key] != 0.0 for key in ("query_head", "teacher_head", "final_decoder")):
        raise RuntimeError(f"Transfer leaked into teacher/query path: {transfer_norms}")
    if (
        args.max_transfer_stage2_gradient > 0
        and transfer_norms["stage2_s8"] > args.max_transfer_stage2_gradient
    ):
        raise RuntimeError(
            "Transfer stage2 gradient exceeds the REP3-calibrated ceiling: "
            f"{transfer_norms['stage2_s8']} > {args.max_transfer_stage2_gradient}"
        )
    if (
        args.max_transfer_stage3_gradient > 0
        and transfer_norms["stage3_s16"] > args.max_transfer_stage3_gradient
    ):
        raise RuntimeError(
            "Transfer stage3 gradient exceeds the REP3-calibrated ceiling: "
            f"{transfer_norms['stage3_s16']} > {args.max_transfer_stage3_gradient}"
        )

    shared = stage2 + stage3
    detection_loss = sum(
        value for name, value in losses.items() if not name.startswith("loss_qcsr_")
    )
    detection_gradients = grad_list(detection_loss, shared, retain_graph=False)
    transfer_shared = transfer_gradients[
        offsets[3] : offsets[5]
    ]
    transfer_detection_cosine = grad_cosine(transfer_shared, detection_gradients)
    if not math.isfinite(transfer_detection_cosine) or (
        transfer_detection_cosine < args.min_transfer_detection_cosine
    ):
        raise RuntimeError(
            "QCSR transfer is too strongly opposed to detection at initialization: "
            f"cosine={transfer_detection_cosine}"
        )

    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    head_parameters = sum(parameter.numel() for parameter in all_heads)
    replay_gate = None
    if args.replay_diagnosis is not None:
        replay = json.loads(args.replay_diagnosis.read_text(encoding="utf-8"))
        shape = replay["metrics"]["shape_mse"]
        replay_gate = {
            "path": str(args.replay_diagnosis),
            "aligned_mean": float(shape["aligned_mean"]),
            "shifted_mean": float(shape["shifted_mean"]),
            "aligned_lower_fraction": float(shape["aligned_lower_fraction"]),
        }
        if not (
            replay_gate["aligned_mean"] < replay_gate["shifted_mean"]
            and replay_gate["aligned_lower_fraction"] >= 0.90
        ):
            raise RuntimeError(
                "REP3 replay does not support shape-normalized transfer: "
                f"{replay_gate}"
            )

    report = {
        "status": "PASS",
        "method": (
            "REP3.1-QCSR-SN"
            if criterion.qcsr_transfer_mode == "shape_correlation"
            else "REP3-QCSR"
        ),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_weight_source": weight_source,
        "configured_train_batch": int(samples.shape[0]),
        "valid_sam_samples": valid_samples,
        "matched_sam_pairs": matched_sam_pairs,
        "query_shape": list(outputs["qcsr_query_embeddings"].shape),
        "teacher_s8_shape": list(outputs["qcsr_teacher_pixels"].shape),
        "student_s16_shape": list(outputs["qcsr_student_pixels"].shape),
        "model_parameters": model_parameters,
        "qcsr_head_parameters": head_parameters,
        "added_parameters_vs_a00": head_parameters,
        "shared_initialization_max_error_vs_a00_same_seed": initialization_error,
        "shared_checkpoint_max_error_vs_a00": checkpoint_error,
        "identity_box_error": identity_box_error,
        "identity_logit_error": identity_logit_error,
        "raw_teacher_loss": float(raw_teacher.detach()),
        "weighted_teacher_loss": float(weighted_teacher.detach()),
        "raw_transfer_loss": float(raw_transfer.detach()),
        "weighted_transfer_loss": float(weighted_transfer.detach()),
        "qcsr_transfer_mode": criterion.qcsr_transfer_mode,
        "qcsr_shape_eps": criterion.qcsr_shape_eps,
        "shifted_teacher_loss": float(shifted_teacher.detach()),
        "shifted_transfer_loss": float(shifted_transfer.detach()),
        "aligned_shifted_transfer_absolute_difference": shift_difference,
        "teacher_amplified_x10_transfer_loss": float(amplified_transfer.detach()),
        "teacher_amplitude_invariance_absolute_error": amplitude_invariance_error,
        "maximum_amplitude_invariance_error": args.max_amplitude_invariance_error,
        "rep3_last_shape_replay_gate": replay_gate,
        "teacher_gradient_l2": teacher_norms,
        "transfer_gradient_l2": transfer_norms,
        "transfer_detection_gradient_cosine_s8_s16": transfer_detection_cosine,
        "minimum_allowed_transfer_detection_cosine": args.min_transfer_detection_cosine,
        "maximum_transfer_stage2_gradient": args.max_transfer_stage2_gradient,
        "maximum_transfer_stage3_gradient": args.max_transfer_stage3_gradient,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
