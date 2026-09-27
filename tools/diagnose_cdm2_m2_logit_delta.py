"""Measure M2/M3 logit correction on the full development set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def checkpoint_state(checkpoint):
    if isinstance(checkpoint.get("ema"), dict):
        return checkpoint["ema"]["module"], "ema.module"
    return checkpoint["model"], "model"


def summarize(values):
    if not values:
        return {"count": 0}
    tensor = torch.cat(values).float()
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "max": float(tensor.max()),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "p99": float(torch.quantile(tensor, 0.99)),
        "nonzero_fraction": float((tensor > 0).float().mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fusion-progress", type=float, default=1.0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config.resolve()))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    state, state_field = checkpoint_state(checkpoint)
    model.load_state_dict(state, strict=True)
    if model.decoder.sdtec_logit_calibrator is None:
        raise RuntimeError("checkpoint/config is not M2")
    if not 0.0 <= args.fusion_progress <= 1.0:
        raise ValueError("fusion-progress must be in [0, 1]")
    model.decoder.sdtec_fusion_progress = float(args.fusion_progress)
    model.decoder.sdtec_logit_calibrator.set_fusion_progress(
        args.fusion_progress
    )

    device = torch.device("cuda")
    model.to(device).eval()
    model.rgbt_thermal_intervention = "normal"
    loader = cfg.val_dataloader
    ordinary_values = []
    ordinary_signed_values = []
    hrqs_values = []
    ordinary_gates = []
    hrqs_gates = []
    attention_entropies = []

    def capture(_module, _inputs, output):
        _calibrated, diagnostics = output
        signed_delta = diagnostics["delta"].detach().float().cpu()
        delta = signed_delta.abs()
        gate = diagnostics["gate"].detach().float().cpu()
        attention = diagnostics["attention"].detach().float().cpu()
        ordinary_values.append(delta[:, :-50].reshape(-1))
        ordinary_signed_values.append(signed_delta[:, :-50].reshape(-1))
        hrqs_values.append(delta[:, -50:].reshape(-1))
        ordinary_gates.append(gate[:, :-50].reshape(-1))
        hrqs_gates.append(gate[:, -50:].reshape(-1))
        entropy = -(attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()).sum(-1)
        entropy = entropy / torch.log(attention.new_tensor(attention.shape[-1]))
        attention_entropies.append(entropy.reshape(-1))

    handle = model.decoder.sdtec_logit_calibrator.register_forward_hook(capture)
    try:
        with torch.no_grad():
            for samples, _targets in loader:
                samples = samples.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model(samples)
    finally:
        handle.remove()

    result = {
        "status": "PASS",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_state_field": state_field,
        "samples": len(loader.dataset),
        "ordinary_query_abs_logit_delta": summarize(ordinary_values),
        "ordinary_query_signed_logit_delta": summarize(ordinary_signed_values),
        "protected_hrqs_abs_logit_delta": summarize(hrqs_values),
        "ordinary_query_gate": summarize(ordinary_gates),
        "protected_hrqs_raw_gate_before_hard_mask": summarize(hrqs_gates),
        "attention_normalized_entropy": summarize(attention_entropies),
        "hard_max_logit_delta": model.decoder.sdtec_max_logit_delta,
        "fusion_progress": float(args.fusion_progress),
    }
    if result["protected_hrqs_abs_logit_delta"].get("max", 0.0) != 0.0:
        raise RuntimeError("M2 changed a protected HRQS logit on the full set")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
