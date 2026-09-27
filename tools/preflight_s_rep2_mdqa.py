#!/usr/bin/env python3
"""Strict preflight for REP2-MDQA query-matched SAM supervision."""

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
        default=ROOT / "experiments/phase_s/s_rep2_mdqa_w025_ft6_lr02_local.yml",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=ROOT / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT.parent
        / "outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent
        / "reports/21_public_reproduction/S_REP2_MDQA/preflight_batch16.json",
    )
    parser.add_argument("--gradient-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--min-shift-difference", type=float, default=1e-6)
    parser.add_argument("--min-shared-gradient-cosine", type=float, default=-0.20)
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
    missing = sorted(set(state).difference(matched))
    return missing


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


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


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("REP2-MDQA preflight requires CUDA")
    if args.gradient_batch != 16:
        raise ValueError("The formal REP2 preflight must use gradient batch 16")
    device = torch.device("cuda")

    # First prove that adding the optional head does not alter the ordinary
    # model initialization when both configs use the same seed.
    seed_everything(args.seed)
    baseline_cfg = YAMLConfig(str(args.baseline_config))
    baseline_model = baseline_cfg.model
    seed_everything(args.seed)
    cfg = YAMLConfig(str(args.config))
    model = cfg.model
    initialization_error, initialization_missing = max_shared_error(
        baseline_model, model
    )
    if initialization_missing or initialization_error != 0.0:
        raise RuntimeError(
            "MDQA changed A00 shared initialization: "
            f"missing={initialization_missing}, max_error={initialization_error}"
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights, weight_source = checkpoint_weights(checkpoint)
    baseline_missing = load_matching(baseline_model, weights)
    mdqa_missing = load_matching(model, weights)
    allowed_missing = {
        key
        for key in model.state_dict()
        if key.startswith("decoder.mdqa_query_proj.")
        or key.startswith("decoder.mdqa_pixel_proj.")
    }
    if set(mdqa_missing) != allowed_missing:
        raise RuntimeError(
            "A00 transfer left unexpected MDQA-model parameters unmatched: "
            f"missing={mdqa_missing}, allowed={sorted(allowed_missing)}"
        )
    checkpoint_shared_error, checkpoint_shared_missing = max_shared_error(
        baseline_model, model
    )
    if baseline_missing or checkpoint_shared_missing or checkpoint_shared_error != 0.0:
        raise RuntimeError(
            "MDQA and A00 differ after checkpoint transfer: "
            f"baseline_missing={baseline_missing}, "
            f"shared_missing={checkpoint_shared_missing}, "
            f"max_error={checkpoint_shared_error}"
        )

    samples, targets = next(iter(cfg.train_dataloader))
    if samples.shape[0] != 16:
        raise RuntimeError(f"REP2 config must use batch16, got {samples.shape[0]}")
    valid_samples = sum(
        int("masks" in target and bool(target["masks"].any())) for target in targets
    )
    if valid_samples <= 0:
        raise RuntimeError("The first REP2 batch has no accepted SAM masks")
    train_samples = samples.to(device)
    train_targets = move_targets(targets, device)

    baseline_model = baseline_model.to(device).eval()
    model = model.to(device).eval()
    with torch.no_grad():
        baseline_output = baseline_model(train_samples[:1])
        mdqa_eval_output = model(train_samples[:1])
    identity_box_error = float(
        (baseline_output["pred_boxes"] - mdqa_eval_output["pred_boxes"])
        .abs()
        .max()
    )
    identity_logit_error = float(
        (baseline_output["pred_logits"] - mdqa_eval_output["pred_logits"])
        .abs()
        .max()
    )
    if identity_box_error != 0.0 or identity_logit_error != 0.0:
        raise RuntimeError(
            "Training-only MDQA changed eval detection outputs: "
            f"boxes={identity_box_error}, logits={identity_logit_error}"
        )
    del baseline_model, baseline_output, mdqa_eval_output
    torch.cuda.empty_cache()

    criterion = cfg.criterion.to(device).train()
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    outputs = model(train_samples, targets=train_targets)
    if "mdqa_query_embeddings" not in outputs or "mdqa_pixel_features" not in outputs:
        raise RuntimeError("MDQA model did not expose its training-only tensors")
    if outputs["mdqa_query_embeddings"].shape[:2] != (16, 300):
        raise RuntimeError(
            "Unexpected MDQA query shape: "
            f"{tuple(outputs['mdqa_query_embeddings'].shape)}"
        )
    if outputs["mdqa_pixel_features"].shape != (16, 64, 64, 80):
        raise RuntimeError(
            "Unexpected MDQA S8 pixel shape: "
            f"{tuple(outputs['mdqa_pixel_features'].shape)}"
        )

    outputs_without_aux = {key: value for key, value in outputs.items() if "aux" not in key}
    indices = criterion.matcher(outputs_without_aux, train_targets)["indices"]
    matched_sam_pairs = 0
    for batch_index, (source_indices, target_indices) in enumerate(indices):
        if source_indices.unique().numel() != source_indices.numel():
            raise RuntimeError("One MDQA detector query matched multiple targets")
        if target_indices.unique().numel() != target_indices.numel():
            raise RuntimeError("One MDQA target matched multiple detector queries")
        masks = train_targets[batch_index]["masks"]
        for target_index in target_indices.tolist():
            if bool(masks[target_index].any()):
                matched_sam_pairs += 1
    if matched_sam_pairs != valid_samples:
        raise RuntimeError(
            "Accepted SAM samples were not paired one-to-one with detector queries: "
            f"valid={valid_samples}, matched={matched_sam_pairs}"
        )

    raw_mdqa_loss = criterion._mdqa_loss(
        outputs["mdqa_query_embeddings"],
        outputs["mdqa_pixel_features"],
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
    shifted_mdqa_loss = criterion._mdqa_loss(
        outputs["mdqa_query_embeddings"],
        outputs["mdqa_pixel_features"],
        shifted_targets,
        indices,
    )
    shift_difference = float((raw_mdqa_loss - shifted_mdqa_loss).abs().detach())
    if shift_difference <= args.min_shift_difference:
        raise RuntimeError(
            "MDQA loss is not measurably sensitive to shifted SAM masks: "
            f"difference={shift_difference}"
        )

    losses = criterion(outputs, train_targets, epoch=0)
    if "loss_mdqa" not in losses:
        raise RuntimeError("Criterion did not return loss_mdqa")
    weighted_mdqa_loss = losses["loss_mdqa"]
    expected = criterion.mdqa_aux_weight * raw_mdqa_loss
    if not torch.allclose(weighted_mdqa_loss, expected, rtol=1e-6, atol=1e-7):
        raise RuntimeError(
            "Unexpected MDQA global weighting: "
            f"actual={float(weighted_mdqa_loss.detach())}, "
            f"expected={float(expected.detach())}"
        )
    if not bool(torch.isfinite(weighted_mdqa_loss)) or float(weighted_mdqa_loss) <= 0:
        raise RuntimeError(f"Invalid MDQA loss: {float(weighted_mdqa_loss.detach())}")

    head_parameters = list(model.decoder.mdqa_query_proj.parameters()) + list(
        model.decoder.mdqa_pixel_proj.parameters()
    )
    backbone_parameters = list(model.backbone.stages[1].parameters())
    decoder_parameters = list(model.decoder.decoder.layers[-1].parameters())
    shared_parameters = backbone_parameters + decoder_parameters
    optimizer_parameter_ids = {
        id(parameter)
        for group in cfg.optimizer.param_groups
        for parameter in group["params"]
    }
    if not set(map(id, head_parameters)).issubset(optimizer_parameter_ids):
        raise RuntimeError("Optimizer omitted MDQA head parameters")

    mdqa_gradients = grad_list(
        weighted_mdqa_loss, shared_parameters + head_parameters, retain_graph=True
    )
    shared_mdqa_gradients = mdqa_gradients[: len(shared_parameters)]
    head_gradients = mdqa_gradients[len(shared_parameters) :]
    detection_loss = sum(
        value for name, value in losses.items() if name != "loss_mdqa"
    )
    detection_gradients = grad_list(
        detection_loss, shared_parameters, retain_graph=False
    )
    backbone_count = len(backbone_parameters)
    shared_cosine = grad_cosine(shared_mdqa_gradients, detection_gradients)
    backbone_cosine = grad_cosine(
        shared_mdqa_gradients[:backbone_count],
        detection_gradients[:backbone_count],
    )
    detach_query = bool(model.decoder.mdqa_detach_query)
    decoder_cosine = (
        None
        if detach_query
        else grad_cosine(
            shared_mdqa_gradients[backbone_count:],
            detection_gradients[backbone_count:],
        )
    )
    head_gradient_l2 = grad_norm(head_gradients)
    backbone_gradient_l2 = grad_norm(shared_mdqa_gradients[:backbone_count])
    decoder_gradient_l2 = grad_norm(shared_mdqa_gradients[backbone_count:])
    if min(head_gradient_l2, backbone_gradient_l2) <= 0:
        raise RuntimeError(
            "MDQA missed an intended gradient path: "
            f"head={head_gradient_l2}, backbone={backbone_gradient_l2}, "
            f"decoder={decoder_gradient_l2}"
        )
    if detach_query and decoder_gradient_l2 != 0.0:
        raise RuntimeError(
            "MDQA query stop-gradient leaked into the final decoder: "
            f"gradient={decoder_gradient_l2}"
        )
    if not detach_query and decoder_gradient_l2 <= 0:
        raise RuntimeError("Ordinary MDQA failed to reach the final decoder")
    if not math.isfinite(shared_cosine) or shared_cosine < args.min_shared_gradient_cosine:
        raise RuntimeError(
            "MDQA is too strongly opposed to the detector at initialization: "
            f"cosine={shared_cosine}, minimum={args.min_shared_gradient_cosine}"
        )

    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    head_parameter_count = sum(parameter.numel() for parameter in head_parameters)
    report = {
        "status": "PASS",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_weight_source": weight_source,
        "gradient_batch": args.gradient_batch,
        "configured_train_batch": int(samples.shape[0]),
        "valid_sam_samples": valid_samples,
        "matched_sam_pairs": matched_sam_pairs,
        "query_embedding_shape": list(outputs["mdqa_query_embeddings"].shape),
        "pixel_feature_shape": list(outputs["mdqa_pixel_features"].shape),
        "model_parameters": model_parameters,
        "mdqa_head_parameters": head_parameter_count,
        "added_parameters_vs_a00": model_parameters
        - sum(parameter.numel() for parameter in cfg.model.parameters())
        + head_parameter_count,
        "shared_initialization_max_error_vs_a00_same_seed": initialization_error,
        "shared_checkpoint_max_error_vs_a00": checkpoint_shared_error,
        "identity_box_error": identity_box_error,
        "identity_logit_error": identity_logit_error,
        "mdqa_aux_weight": criterion.mdqa_aux_weight,
        "mdqa_detach_query": detach_query,
        "raw_mdqa_loss": float(raw_mdqa_loss.detach()),
        "weighted_mdqa_loss": float(weighted_mdqa_loss.detach()),
        "shifted_mdqa_loss": float(shifted_mdqa_loss.detach()),
        "aligned_shifted_absolute_difference": shift_difference,
        "aligned_preferred_at_random_init": bool(
            float(raw_mdqa_loss.detach()) < float(shifted_mdqa_loss.detach())
        ),
        "mdqa_head_gradient_l2": head_gradient_l2,
        "s8_backbone_gradient_l2": backbone_gradient_l2,
        "final_decoder_gradient_l2": decoder_gradient_l2,
        "shared_gradient_cosine_mdqa_vs_detection": shared_cosine,
        "s8_backbone_gradient_cosine_mdqa_vs_detection": backbone_cosine,
        "final_decoder_gradient_cosine_mdqa_vs_detection": decoder_cosine,
        "minimum_allowed_shared_gradient_cosine": args.min_shared_gradient_cosine,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }
    # The only parameters absent from A00 are the two training-only projections.
    report["added_parameters_vs_a00"] = head_parameter_count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
