#!/usr/bin/env python3
"""P0/P1 preflight for the strict S-BPC2-QRL-R1 detail-only delta."""

from __future__ import annotations

import argparse
import gc
import json
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
        / "experiments/phase_s/s_bpc2_qrl_r1_detail_only_s8_s16_local.yml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent
        / "reports/20_spatial_importance/S_BPC2_QRL_R1_DETAIL_ONLY/preflight.json",
    )
    parser.add_argument("--gradient-batch", type=int, default=2)
    return parser.parse_args()


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def autograd_l2(loss, parameters, retain_graph=True):
    parameters = list(parameters)
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    squares = [
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def selected_loss(losses, fragments):
    values = [
        value
        for key, value in losses.items()
        if any(fragment in key for fragment in fragments)
    ]
    if not values:
        raise RuntimeError(f"no losses selected for {fragments}")
    return sum(values)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("S-BPC2-QRL-R1 preflight requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(20260815)

    cfg = YAMLConfig(str(args.config))
    loader = cfg.train_dataloader
    samples, targets = next(iter(loader))
    if samples.shape[0] != 32:
        raise RuntimeError(f"QRL-R1 requires batch32 preflight, got {samples.shape[0]}")

    model = cfg.model.to(device).train()
    criterion = cfg.criterion.to(device).train()
    optimizer = cfg.optimizer
    qrl = model.decoder.qrl
    if qrl is None or not qrl.detail_only_delta:
        raise RuntimeError("R1 config did not construct the detail-only QRL carrier")
    forbidden_attributes = [
        name
        for name in ("query_context", "side_embedding", "delta_head")
        if hasattr(qrl, name)
    ]
    if forbidden_attributes:
        raise RuntimeError(f"R1 retained forbidden bypasses: {forbidden_attributes}")
    if len(qrl.side_projections) != 4:
        raise RuntimeError("R1 must contain exactly four side projections")
    if any(projection.bias is not None for projection in qrl.side_projections):
        raise RuntimeError("R1 side projections must not contain bias")
    if qrl.detail_norm.elementwise_affine:
        raise RuntimeError("R1 detail LayerNorm must not contain affine parameters")

    side_projection_parameters = list(qrl.side_projections.parameters())
    side_projection_parameter_count = sum(
        parameter.numel() for parameter in side_projection_parameters
    )
    expected_projection_parameters = 4 * qrl.detail_channels * (qrl.reg_max + 1)
    if side_projection_parameter_count != expected_projection_parameters:
        raise RuntimeError(
            "unexpected R1 projection parameter count: "
            f"{side_projection_parameter_count} != {expected_projection_parameters}"
        )

    qrl_parameters = list(qrl.parameters())
    qrl_parameter_count = sum(parameter.numel() for parameter in qrl_parameters)
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    baseline_parameter_count = model_parameter_count - qrl_parameter_count
    if qrl_parameter_count > 50_000:
        raise RuntimeError(f"QRL-R1 parameter budget exceeded: {qrl_parameter_count}")

    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    if len(optimizer_parameters) != len({id(parameter) for parameter in optimizer_parameters}):
        raise RuntimeError("optimizer contains duplicate parameter tensors")
    if not set(map(id, qrl_parameters)).issubset(set(map(id, optimizer_parameters))):
        raise RuntimeError("optimizer omitted QRL-R1 parameters")

    # The causal zero-detail invariant must hold even for nonzero projection
    # weights.  Restore exact zero initialization before equivalence checks.
    saved_projection_weights = [
        projection.weight.detach().clone() for projection in qrl.side_projections
    ]
    with torch.no_grad():
        for projection in qrl.side_projections:
            projection.weight.normal_(mean=0.0, std=0.1)
        synthetic_zero = torch.zeros(
            2, 7, 4, qrl.detail_channels, device=device
        )
        random_weight_zero_detail_error = float(
            qrl.detail_only_delta_from_tokens(synthetic_zero).abs().max()
        )
        for projection, saved_weight in zip(
            qrl.side_projections, saved_projection_weights
        ):
            projection.weight.copy_(saved_weight)
    if random_weight_zero_detail_error > 1e-7:
        raise RuntimeError(
            "zero-detail invariant failed with nonzero weights: "
            f"{random_weight_zero_detail_error}"
        )

    # P0: compare the zero-initialized R1 against the same model with QRL
    # disabled, avoiding checkpoint/config ordering confounds.
    eval_samples = samples[:1].to(device)
    model.eval()
    with torch.no_grad():
        r1_outputs = model(eval_samples)
        model.decoder.qrl = None
        model.decoder.qrl_enabled = False
        baseline_outputs = model(eval_samples)
        model.decoder.qrl = qrl
        model.decoder.qrl_enabled = True
    identity_box_error = float(
        (r1_outputs["pred_boxes"] - baseline_outputs["pred_boxes"]).abs().max()
    )
    identity_logit_error = float(
        (r1_outputs["pred_logits"] - baseline_outputs["pred_logits"]).abs().max()
    )
    if identity_box_error != 0.0 or identity_logit_error != 0.0:
        raise RuntimeError(
            "zero-initialized QRL-R1 changed A00 outputs: "
            f"boxes={identity_box_error}, logits={identity_logit_error}"
        )

    # P1 phase 1: exact zero projection receives localization gradient while
    # the upstream detail path remains blocked as expected.
    grad_batch = int(args.gradient_batch)
    train_samples = samples[:grad_batch].to(device)
    train_targets = move_targets(targets[:grad_batch], device)
    model.train()
    outputs = model(train_samples, targets=train_targets)
    losses = criterion(outputs, train_targets, epoch=0)
    if "loss_qrl_region" not in losses:
        raise RuntimeError("QRL-R1 region loss is missing")
    classification_loss = selected_loss(losses, ("loss_vfl", "loss_focal"))
    localization_loss = selected_loss(
        losses, ("loss_bbox", "loss_giou", "loss_fgl", "loss_ddf")
    )
    region_loss = losses["loss_qrl_region"]

    classification_to_qrl = autograd_l2(
        classification_loss, qrl_parameters, retain_graph=True
    )
    localization_to_side_projections = autograd_l2(
        localization_loss, side_projection_parameters, retain_graph=True
    )
    localization_to_reduce_initial = autograd_l2(
        localization_loss, qrl.reduce.parameters(), retain_graph=True
    )
    region_to_student = autograd_l2(
        region_loss, qrl.region_student.parameters(), retain_graph=True
    )
    region_to_reduce = autograd_l2(
        region_loss, qrl.reduce.parameters(), retain_graph=True
    )
    region_to_backbone = autograd_l2(
        region_loss, model.backbone.parameters(), retain_graph=True
    )
    if classification_to_qrl != 0.0:
        raise RuntimeError(
            f"classification gradient leaked into QRL-R1: {classification_to_qrl}"
        )
    if localization_to_side_projections <= 0.0:
        raise RuntimeError("localization did not reach zero-initialized side projections")
    if localization_to_reduce_initial != 0.0:
        raise RuntimeError(
            "zero side projections unexpectedly passed initial localization gradient "
            f"upstream: {localization_to_reduce_initial}"
        )
    if region_to_student <= 0.0 or region_to_reduce <= 0.0:
        raise RuntimeError(
            "region supervision missed QRL-R1 adapters: "
            f"student={region_to_student}, reduce={region_to_reduce}"
        )
    if region_to_backbone != 0.0:
        raise RuntimeError(f"region loss leaked into shared backbone: {region_to_backbone}")

    projection_gradients = torch.autograd.grad(
        localization_loss, side_projection_parameters, retain_graph=False
    )
    # Exercise the configured AdamW first step exactly.  A raw ``-lr * grad``
    # update is not equivalent to AdamW's bias-corrected, normalized first
    # moment and substantially underestimates the first-step delta.
    optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(side_projection_parameters, projection_gradients):
        parameter.grad = gradient.detach().clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # P1 phase 2: after projections leave zero, localization must train the
    # entire detached detail reader, while zero detail remains exactly zero.
    outputs_second = model(train_samples, targets=train_targets)
    losses_second = criterion(outputs_second, train_targets, epoch=0)
    localization_second = selected_loss(
        losses_second, ("loss_bbox", "loss_giou", "loss_fgl", "loss_ddf")
    )
    second_gradients = {
        "reduce": autograd_l2(
            localization_second, qrl.reduce.parameters(), retain_graph=True
        ),
        "region_student": autograd_l2(
            localization_second, qrl.region_student.parameters(), retain_graph=True
        ),
        "side_projections": autograd_l2(
            localization_second, side_projection_parameters, retain_graph=True
        ),
    }
    if any(value <= 0.0 for value in second_gradients.values()):
        raise RuntimeError(
            f"localization missed upstream QRL-R1 adapters: {second_gradients}"
        )
    post_update_delta_rms = float(qrl.last_delta_rms)
    if post_update_delta_rms <= 1e-5:
        raise RuntimeError(
            "QRL-R1 delta did not leave zero strongly enough after projection update: "
            f"{post_update_delta_rms}"
        )
    with torch.no_grad():
        zero_tokens = torch.zeros_like(qrl.last_side_tokens)
        post_update_zero_detail_error = float(
            qrl.detail_only_delta_from_tokens(zero_tokens).abs().max()
        )
        changed_tokens = torch.roll(qrl.last_side_tokens, shifts=1, dims=-2)
        changed_token_delta_rms = float(
            (
                qrl.detail_only_delta_from_tokens(qrl.last_side_tokens)
                - qrl.detail_only_delta_from_tokens(changed_tokens)
            )
            .float()
            .square()
            .mean()
            .sqrt()
        )
    if post_update_zero_detail_error > 1e-7:
        raise RuntimeError(
            "trained R1 violated the zero-detail invariant: "
            f"{post_update_zero_detail_error}"
        )
    if changed_token_delta_rms <= 0.0:
        raise RuntimeError("changing side tokens did not change the R1 delta")

    region_logits = outputs_second["qrl_region_logits"]
    region_target, quality = criterion._qrl_region_targets(
        region_logits.float(), train_targets
    )
    if not bool(region_target.sum() > 0) or not bool(quality.sum() > 0):
        raise RuntimeError("preflight batch contains no valid QRL region supervision")

    del outputs, outputs_second, losses, losses_second
    gc.collect()
    torch.cuda.empty_cache()

    # Batch32 full forward/backward is the memory and finite-value gate.
    full_samples = samples.to(device)
    full_targets = move_targets(targets, device)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.float16):
        full_outputs = model(full_samples, targets=full_targets)
    with torch.autocast("cuda", enabled=False):
        full_losses = criterion(full_outputs, full_targets, epoch=0)
        full_loss = sum(full_losses.values())
    if not torch.isfinite(full_loss):
        raise RuntimeError(f"non-finite batch32 QRL-R1 loss: {full_losses}")
    full_loss.backward()
    peak_memory_mib = torch.cuda.max_memory_allocated(device) / (1024**2)

    report = {
        "config": str(args.config),
        "device": torch.cuda.get_device_name(0),
        "batch_size": int(samples.shape[0]),
        "image_shape": list(samples.shape),
        "detail_only_delta": qrl.detail_only_delta,
        "forbidden_attributes_present": forbidden_attributes,
        "side_projection_count": len(qrl.side_projections),
        "side_projection_parameters": side_projection_parameter_count,
        "qrl_parameters": qrl_parameter_count,
        "baseline_parameters": baseline_parameter_count,
        "qrl_parameter_fraction": qrl_parameter_count / baseline_parameter_count,
        "random_weight_zero_detail_max_abs_error": random_weight_zero_detail_error,
        "identity_box_max_abs_error": identity_box_error,
        "identity_logit_max_abs_error": identity_logit_error,
        "classification_to_qrl_gradient_l2": classification_to_qrl,
        "localization_to_side_projections_initial_gradient_l2": localization_to_side_projections,
        "localization_to_reduce_initial_gradient_l2": localization_to_reduce_initial,
        "region_to_student_gradient_l2": region_to_student,
        "region_to_reduce_gradient_l2": region_to_reduce,
        "region_to_backbone_gradient_l2": region_to_backbone,
        "localization_upstream_after_projection_update_l2": second_gradients,
        "post_update_delta_rms": post_update_delta_rms,
        "post_update_zero_detail_max_abs_error": post_update_zero_detail_error,
        "changed_side_token_delta_difference_rms": changed_token_delta_rms,
        "region_logits_shape": list(region_logits.shape),
        "region_target_mean": float(region_target.mean()),
        "quality_mean": float(quality.mean()),
        "quality_nonzero": int((quality > 0).sum()),
        "loss_qrl_region": float(region_loss.detach()),
        "batch32_total_loss": float(full_loss.detach()),
        "batch32_peak_cuda_memory_mib": peak_memory_mib,
        "batch32_all_losses_finite": all(
            bool(torch.isfinite(value)) for value in full_losses.values()
        ),
        "inference_without_masks": True,
        "p0_p1_pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
