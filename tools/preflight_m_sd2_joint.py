"""CUDA identity, invariance, gradient and memory preflight for M-SD2 variants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def checkpoint_state(checkpoint):
    if isinstance(checkpoint.get("ema"), dict) and isinstance(
        checkpoint["ema"].get("module"), dict
    ):
        return checkpoint["ema"]["module"]
    return checkpoint.get("model", checkpoint)


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def max_error(left, right):
    return float((left.float() - right.float()).abs().max().cpu())


def gradient_summary(model, prefix):
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith(prefix)
    }
    non_finite = sum(
        gradient is not None
        and not bool(torch.isfinite(gradient).all().cpu())
        for gradient in gradients.values()
    )
    return {
        "tensor_count": len(gradients),
        "with_gradient": sum(gradient is not None for gradient in gradients.values()),
        "with_nonzero_gradient": sum(
            gradient is not None
            and bool(torch.isfinite(gradient).all().cpu())
            and float(gradient.detach().abs().max().cpu()) > 0.0
            for gradient in gradients.values()
        ),
        "non_finite_gradient_tensors": non_finite,
        "max_abs_gradient": max(
            (
                float(gradient.detach().abs().max().cpu())
                for gradient in gradients.values()
                if gradient is not None
                and bool(torch.isfinite(gradient).all().cpu())
            ),
            default=0.0,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="Use two samples to validate structure without granting PASS",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    cfg = YAMLConfig(str(args.config.resolve()))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    # Windows worker processes import CUDA-aware project modules.  Keep the
    # structural preflight single-process so workers cannot reserve WDDM GPU
    # address space before the parent allocates the formal batch-32 tensor.
    cfg.yaml_cfg["train_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["train_dataloader"]["persistent_workers"] = False
    model = cfg.model
    source = checkpoint_state(
        torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    )
    destination = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in destination and value.shape == destination[key].shape
    }
    model.load_state_dict(compatible, strict=False)

    if not model.rgbt_enabled or model.sd2_conditioner is None:
        raise RuntimeError("configured model did not construct M-SD2")
    if not model.decoder.hrqs_enabled:
        raise RuntimeError("M-SD2 joint model must retain D-HRQS1")
    if model.decoder.sdtec_enabled:
        raise RuntimeError("legacy SDTEC query fusion must be disabled")

    device = torch.device("cuda")
    optimizer = cfg.optimizer
    loader = cfg.train_dataloader
    samples, targets = next(iter(loader))
    if args.structural_only:
        samples = samples[:2]
        targets = targets[:2]
    print(
        "cuda_before_samples_gib="
        f"{torch.cuda.memory_allocated(device) / 2**30:.4f}/"
        f"{torch.cuda.memory_reserved(device) / 2**30:.4f} "
        f"driver_free_total_gib="
        f"{torch.cuda.mem_get_info(0)[0] / 2**30:.4f}/"
        f"{torch.cuda.mem_get_info(0)[1] / 2**30:.4f} "
        f"sample_shape={tuple(samples.shape)} sample_gib="
        f"{samples.numel() * samples.element_size() / 2**30:.4f}",
        flush=True,
    )
    gpu_samples = torch.empty(
        samples.shape,
        dtype=samples.dtype,
        device=device,
    )
    gpu_samples.copy_(samples)
    samples = gpu_samples
    targets = move_targets(targets, device)
    model.to(device)
    criterion = cfg.criterion.to(device)
    accumulation_steps = int(
        cfg.yaml_cfg.get("gradient_accumulation_steps", 1)
    )
    if (
        not args.structural_only
        and (samples.shape[0] != 8 or accumulation_steps != 4)
    ):
        raise RuntimeError(
            "formal preflight requires physical batch 8 x accumulation 4, "
            f"got {samples.shape[0]} x {accumulation_steps}"
        )

    trainable = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    missing_optimizer_parameters = sorted(
        name for identifier, name in trainable.items() if identifier not in optimized
    )
    if missing_optimizer_parameters:
        raise RuntimeError(
            "optimizer omitted trainable parameters: "
            f"{missing_optimizer_parameters[:10]}"
        )

    probe = samples[:2]
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.float16
    ):
        identity_outputs = model(probe)
        conditioner = model.sd2_conditioner
        model.sd2_conditioner = None
        base_outputs = model(probe)
        model.sd2_conditioner = conditioner
    identity_logit_error = max_error(
        identity_outputs["pred_logits"], base_outputs["pred_logits"]
    )
    identity_box_error = max_error(
        identity_outputs["pred_boxes"], base_outputs["pred_boxes"]
    )
    if identity_logit_error != 0.0 or identity_box_error != 0.0:
        raise RuntimeError(
            "zero M-SD2 residual failed exact identity: "
            f"logits={identity_logit_error} boxes={identity_box_error}"
        )

    with torch.no_grad():
        conditioner_state = {
            key: value.detach().clone()
            for key, value in model.sd2_conditioner.state_dict().items()
        }
        if hasattr(model.sd2_conditioner, "residual_scales"):
            model.sd2_conditioner.residual_scales.fill_(0.5)
        else:
            # M-SD2.1 starts with zero output projections instead of a zero
            # scalar.  Open only those projections for the structural probe.
            generator = torch.Generator(device=device).manual_seed(20260901)
            for channel_generator in model.sd2_conditioner.channel_generators:
                channel_generator[-1].weight.normal_(
                    mean=0.0,
                    std=0.02,
                    generator=generator,
                )
                if channel_generator[-1].bias is not None:
                    channel_generator[-1].bias.zero_()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            visible_features = model.backbone(probe[:, :3])
            thermal_features = model.thermal_backbone(probe[:, 3:])
            if model.decoder.hrqs_enabled:
                visible_features = visible_features[1:]
                thermal_features = thermal_features[1:]
            visible_features = model.encoder(visible_features)
            thermal_features = model.thermal_encoder(thermal_features)
            normal_open = model.sd2_conditioner(
                visible_features,
                thermal_features,
            )
        open_rms_ratio = model.sd2_conditioner.last_rms_ratio_by_level.clone()
        null_conditioned_feature_error = None
        if (
            getattr(model.sd2_conditioner, "thermal_contrastive", False)
            or getattr(model.sd2_conditioner, "zero_anchored", False)
        ):
            intervention = model.sd2_conditioner.intervention_token_mode
            model.sd2_conditioner.intervention_token_mode = "zero"
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                null_open = model.sd2_conditioner(
                    visible_features,
                    thermal_features,
                )
            model.sd2_conditioner.intervention_token_mode = intervention
            null_conditioned_feature_error = max(
                max_error(visible, conditioned)
                for visible, conditioned in zip(visible_features, null_open)
            )
            if null_conditioned_feature_error != 0.0:
                raise RuntimeError(
                    "M-SD2 null thermal condition did not return exact identity: "
                    f"{null_conditioned_feature_error}"
                )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            permuted_features = [
                feature.flatten(2).flip(-1).reshape_as(feature)
                for feature in thermal_features
            ]
            permuted_open = model.sd2_conditioner(
                visible_features,
                permuted_features,
            )
        model.sd2_conditioner.load_state_dict(conditioner_state)
    permutation_feature_error = max(
        max_error(normal, permuted)
        for normal, permuted in zip(normal_open, permuted_open)
    )
    # The FP32 set pooling itself is invariant to roughly 1e-6.  Returning to
    # the detector's FP16 memory can move a value by one half-precision ULP;
    # allow that representation-level bound, but nothing larger.
    permutation_tolerance = (
        3e-4
        if (
            getattr(model.sd2_conditioner, "thermal_contrastive", False)
            or getattr(model.sd2_conditioner, "zero_anchored", False)
        )
        else 2e-4
    )
    if permutation_feature_error > permutation_tolerance:
        raise RuntimeError(
            "M-SD2 is not spatial-permutation invariant: "
            f"conditioned_feature={permutation_feature_error} "
            f"tolerance={permutation_tolerance}"
        )
    if float(open_rms_ratio.max().cpu()) > model.sd2_conditioner.max_rms_ratio + 1e-4:
        raise RuntimeError("M-SD2 exceeded its true RMS update bound")

    del (
        identity_outputs,
        base_outputs,
        normal_open,
        permuted_open,
        visible_features,
        thermal_features,
        permuted_features,
        probe,
    )
    torch.cuda.empty_cache()
    print(
        "cuda_before_training_gib="
        f"{torch.cuda.memory_allocated(device) / 2**30:.4f}/"
        f"{torch.cuda.memory_reserved(device) / 2**30:.4f}",
        flush=True,
    )

    model.train()
    model.set_training_epoch(0)
    scaler = torch.cuda.amp.GradScaler(init_scale=128.0)
    torch.cuda.reset_peak_memory_stats(device)
    steps = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(samples, targets=targets)
        with torch.autocast(device_type="cuda", enabled=False):
            losses = criterion(
                outputs,
                targets,
                epoch=0,
                step=step,
                global_step=step,
                epoch_step=len(loader),
            )
            total_loss = sum(losses.values())
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite loss at step {step}: {total_loss}")
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        step_result = {
            "step": step,
            "loss": float(total_loss.detach().cpu()),
            "sd2": gradient_summary(model, "sd2_conditioner."),
            "visible_backbone": gradient_summary(model, "backbone."),
            "thermal_backbone": gradient_summary(model, "thermal_backbone."),
            "hrqs": gradient_summary(model, "decoder.hrqs_adapter."),
            "rms_ratio_mean": [
                float(value)
                for value in model.sd2_conditioner.last_rms_ratio_by_level.mean(
                    dim=0
                ).cpu()
            ],
            "rms_ratio_max": float(
                model.sd2_conditioner.last_rms_ratio_by_level.max().cpu()
            ),
            "null_raw_rms_mean": float(
                model.sd2_conditioner.last_null_raw_rms_by_level.mean().cpu()
            ),
            "contrast_raw_rms_mean": float(
                model.sd2_conditioner.last_contrast_raw_rms_by_level.mean().cpu()
            ),
        }
        steps.append(step_result)
        for branch, summary in step_result.items():
            if isinstance(summary, dict) and summary.get(
                "non_finite_gradient_tensors", 0
            ):
                raise RuntimeError(
                    f"non-finite gradient before clipping: {branch} "
                    f"count={summary['non_finite_gradient_tensors']}"
                )
        if cfg.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.clip_max_norm
            )
        scaler.step(optimizer)
        scaler.update()

    print(json.dumps({"gradient_steps": steps}, ensure_ascii=False, indent=2))
    if steps[0]["sd2"]["with_nonzero_gradient"] < 1:
        raise RuntimeError("M-SD2 residual scales received no opening gradient")
    if steps[1]["sd2"]["with_nonzero_gradient"] <= steps[0]["sd2"]["with_nonzero_gradient"]:
        raise RuntimeError("M-SD2 internal MLPs did not open after the first update")
    for branch in ("visible_backbone", "hrqs"):
        if steps[1][branch]["with_nonzero_gradient"] == 0:
            raise RuntimeError(f"joint branch has no gradient: {branch}")
    if steps[1]["thermal_backbone"]["with_gradient"] != 0:
        raise RuntimeError("frozen thermal extractor unexpectedly received gradients")
    if steps[1]["rms_ratio_max"] > model.sd2_conditioner.max_rms_ratio + 1e-4:
        raise RuntimeError("trained M-SD2 update exceeded true RMS bound")

    selected = model.decoder.last_hrqs_selected_count
    if selected is None or int(selected.min().cpu()) != model.decoder.hrqs_num_queries:
        raise RuntimeError("D-HRQS1 did not retain exactly 50 S8 candidates")

    result = {
        "status": "STRUCTURAL_PASS" if args.structural_only else "PASS",
        "schema": "m_sd2_joint_preflight_v2",
        "conditioner_class": type(model.sd2_conditioner).__name__,
        "thermal_contrastive": bool(
            getattr(model.sd2_conditioner, "thermal_contrastive", False)
        ),
        "zero_anchored": bool(
            getattr(model.sd2_conditioner, "zero_anchored", False)
        ),
        "zero_anchored_levels": list(
            getattr(model.sd2_conditioner, "zero_anchored_levels", ())
        ),
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "batch_size": int(samples.shape[0]),
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": int(samples.shape[0]) * accumulation_steps,
        "input_shape": list(samples.shape),
        "loaded_tuning_tensors": len(compatible),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "sd2_parameter_count": sum(
            parameter.numel() for parameter in model.sd2_conditioner.parameters()
        ),
        "identity_logit_max_error": identity_logit_error,
        "identity_box_max_error": identity_box_error,
        "spatial_permutation_conditioned_feature_max_error": (
            permutation_feature_error
        ),
        "spatial_permutation_feature_tolerance": permutation_tolerance,
        "null_conditioned_feature_max_error": null_conditioned_feature_error,
        "opened_probe_rms_ratio_max": float(open_rms_ratio.max().cpu()),
        "configured_max_rms_ratio": model.sd2_conditioner.max_rms_ratio,
        "token_diversity": (
            float(model.sd2_conditioner.last_token_diversity.cpu())
            if getattr(model.sd2_conditioner, "last_token_diversity", None)
            is not None
            else None
        ),
        "hrqs_selected_min": int(selected.min().cpu()),
        "hrqs_selected_max": int(selected.max().cpu()),
        "optimizer_group_count": len(optimizer.param_groups),
        "missing_optimizer_parameters": missing_optimizer_parameters,
        "steps": steps,
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
