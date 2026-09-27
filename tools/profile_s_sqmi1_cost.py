#!/usr/bin/env python3
"""Measure S-QMI1 parameter and batch-1 CUDA latency overhead without training."""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=ROOT / "experiments/phase_m/c_only_gq1_b8a4_20e_testdev_local.yml",
    )
    parser.add_argument(
        "--init-config",
        type=Path,
        default=ROOT / "experiments/phase_s/s_sqmi1_init_c_gq1_b16a2_12e_testdev_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=WORKSPACE
        / "outputs/C_ONLY_GQ1_B8A4_20E_TESTDEV/seed0/best_stg1.pth",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "reports/144_sqmi1_preflight/cost.json",
    )
    return parser.parse_args()


def weights(checkpoint):
    if isinstance(checkpoint.get("ema"), dict):
        return checkpoint["ema"].get("module", checkpoint["ema"])
    return checkpoint.get("model", checkpoint)


def load_matching(model, state):
    own = model.state_dict()
    model.load_state_dict(
        {key: value for key, value in state.items() if key in own and own[key].shape == value.shape},
        strict=False,
    )


def timed_forward(model, sample, warmup, repeats):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        torch.cuda.synchronize()
        values = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(sample)
            end.record()
            torch.cuda.synchronize()
            values.append(float(start.elapsed_time(end)))
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples_ms": values,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = weights(checkpoint)
    torch.manual_seed(20260916)
    baseline = YAMLConfig(str(args.baseline_config)).model
    torch.manual_seed(20260916)
    init = YAMLConfig(str(args.init_config)).model
    load_matching(baseline, state)
    load_matching(init, state)

    baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
    init_params = sum(parameter.numel() for parameter in init.parameters())
    sqmi_params = sum(parameter.numel() for parameter in init.decoder.sqmi.parameters())
    if init_params - baseline_params != sqmi_params:
        raise RuntimeError("Unexpected non-SQMI parameter delta")

    sample = torch.randn(1, 3, 512, 640, device="cuda")
    baseline = baseline.cuda()
    init = init.cuda()
    baseline_time = timed_forward(baseline, sample, args.warmup, args.repeats)
    init_time = timed_forward(init, sample, args.warmup, args.repeats)

    # Analytical MAC count for the private S-QMI path at RGB S8=64x80.
    module = init.decoder.sqmi
    query_count = init.decoder.num_queries
    selected = module.topk
    hidden = module.query_proj[0].in_features
    mask_dim = module.pixel_proj[0].out_channels
    source_channels = module.pixel_proj[0].in_channels
    height, width = 64, 80
    query_projection_macs = query_count * (hidden * hidden + hidden * mask_dim)
    pixel_projection_macs = height * width * source_channels * mask_dim
    mask_dot_macs = selected * height * width * mask_dim
    gate_hidden = module.gate[0].out_features
    gate_macs = selected * ((hidden + 7) * gate_hidden + gate_hidden)
    private_macs = (
        query_projection_macs + pixel_projection_macs + mask_dot_macs + gate_macs
    )
    result = {
        "status": "pass",
        "input": [1, 3, 512, 640],
        "warmup": args.warmup,
        "repeats": args.repeats,
        "baseline_params": baseline_params,
        "sqmi_params": sqmi_params,
        "init_params": init_params,
        "parameter_increase_percent": 100.0 * sqmi_params / baseline_params,
        "sqmi_private_macs": private_macs,
        "sqmi_private_gmac": private_macs / 1e9,
        "baseline_latency": baseline_time,
        "sqmi_latency": init_time,
        "median_latency_increase_ms": init_time["median_ms"] - baseline_time["median_ms"],
        "median_latency_increase_percent": 100.0
        * (init_time["median_ms"] / baseline_time["median_ms"] - 1.0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
