#!/usr/bin/env python3
"""Preflight S-RES1 identity, tensor path, gradients, and cost-relevant metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def compatible_load(model, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = payload.get("ema", {}).get("module") or payload.get("model", payload)
    own = model.state_dict()
    compatible = {
        key: value for key, value in weights.items() if key in own and own[key].shape == value.shape
    }
    result = model.load_state_dict(compatible, strict=False)
    return len(compatible), list(result.missing_keys), list(result.unexpected_keys)


def reset_hgnet(cfg, **overrides):
    defaults = {
        "srtod_stage": -1,
        "srtod_student_only": False,
        "srtod_enhancer": "dgfe",
        "srtod_detail_groups": 16,
        "srtod_detail_init_scale": 0.0,
        "pat_stage": -1,
        "pat_sf_stage": -1,
        "lad_stage": -1,
        "preserve_stage": -1,
        "rpc_stage": -1,
    }
    defaults.update(overrides)
    cfg.yaml_cfg["HGNetv2"].update(defaults)
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False


def grad_norm(module):
    values = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--base-config", type=Path, required=True)
    p.add_argument("--res-config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.repo))
    from src.core import YAMLConfig

    base_cfg = YAMLConfig(str(args.base_config))
    reset_hgnet(base_cfg)
    base = base_cfg.model.cuda().eval()
    base_loaded, base_missing, base_unexpected = compatible_load(base, args.checkpoint)
    if base_missing or base_unexpected:
        raise RuntimeError(f"A00 checkpoint mismatch: {base_missing}, {base_unexpected}")

    res_cfg = YAMLConfig(str(args.res_config))
    reset_hgnet(
        res_cfg,
        srtod_stage=2,
        srtod_student_only=False,
        srtod_enhancer="haar_detail",
    )
    res = res_cfg.model.cuda().eval()
    res_loaded, res_missing, res_unexpected = compatible_load(res, args.checkpoint)
    allowed_missing_prefixes = (
        "backbone.srtod_rh.",
        "backbone.srtod_learnable_thresh",
        "backbone.srtod_detail_residual.",
    )
    forbidden_missing = [
        key for key in res_missing if not key.startswith(allowed_missing_prefixes)
    ]
    if forbidden_missing or res_unexpected:
        raise RuntimeError(f"S-RES1 checkpoint mismatch: {forbidden_missing}, {res_unexpected}")

    samples, _ = next(iter(base_cfg.val_dataloader))
    samples = samples.cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        base_out = base(samples)
        res_out = res(samples)
    identity = {
        name: float((base_out[name].float() - res_out[name].float()).abs().max())
        for name in ("pred_logits", "pred_boxes")
    }

    detail_module = res.backbone.srtod_detail_residual
    parameter_count = sum(parameter.numel() for parameter in detail_module.parameters())
    projection_groups = detail_module.projection_groups
    # 512x640 input -> S8=64x80, Haar/S16=32x40.
    h16, w16 = samples.shape[-2] // 16, samples.shape[-1] // 16
    in_channels, out_channels = detail_module.in_channels, detail_module.out_channels
    mac = h16 * w16 * (
        (3 * in_channels * out_channels) // projection_groups
        + out_channels * 9
    )

    # Gradient audit at the exact zero start.
    # Keep evaluation behavior so D-FINE does not require denoising targets;
    # autograd remains enabled and is sufficient for the path audit.
    res.eval()
    res.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        output = res(samples[:2])
        loss = output["pred_logits"].float().mean() + output["pred_boxes"].float().mean()
        loss = loss + output["srtod_reconstruction_loss"]
    loss.backward()
    zero_start = {
        "scale": float(detail_module.scale.detach()),
        "scale_grad_abs": float(detail_module.scale.grad.detach().abs()),
        "orientation_projection_grad_norm": grad_norm(detail_module.orientation_mix),
        "local_refine_grad_norm": grad_norm(detail_module.local_refine),
        "rh_grad_norm": grad_norm(res.backbone.srtod_rh),
    }

    # Once scale leaves zero, the detail branch itself must become trainable.
    with torch.no_grad():
        detail_module.scale.fill_(0.01)
    res.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        output = res(samples[:2])
        loss = output["pred_logits"].float().mean() + output["pred_boxes"].float().mean()
        loss = loss + output["srtod_reconstruction_loss"]
    loss.backward()
    nonzero_start = {
        "scale": float(detail_module.scale.detach()),
        "scale_grad_abs": float(detail_module.scale.grad.detach().abs()),
        "orientation_projection_grad_norm": grad_norm(detail_module.orientation_mix),
        "local_refine_grad_norm": grad_norm(detail_module.local_refine),
        "rh_grad_norm": grad_norm(res.backbone.srtod_rh),
    }

    report = {
        "base_loaded_tensors": base_loaded,
        "res_loaded_tensors": res_loaded,
        "res_expected_missing": res_missing,
        "identity_max_abs_error_at_scale_zero": identity,
        "detail_parameter_count": parameter_count,
        "detail_mac_512x640": mac,
        "detail_projection_groups": projection_groups,
        "zero_start_gradient_audit": zero_start,
        "scale_0p01_gradient_audit": nonzero_start,
        "detail_residual_shape": list(detail_module.last_gate.shape),
        "applied_gate_mean": float(detail_module.last_gate.float().mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
