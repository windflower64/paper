#!/usr/bin/env python3
"""Benchmark one D-FINE experiment under a fixed, reproducible GPU protocol."""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


def load_compatible(model, checkpoint: Path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state.get("ema", {}).get("module") or state.get("model") or state
    own = model.state_dict()
    compatible = {k: v for k, v in weights.items() if k in own and own[k].shape == v.shape}
    result = model.load_state_dict(compatible, strict=False)
    return len(compatible), list(result.missing_keys), list(result.unexpected_keys)


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--input-channels",
        type=int,
        choices=(3, 6),
        help="Defaults to 6 for an RGB-T model and 3 otherwise.",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    from src.core import YAMLConfig

    torch.backends.cudnn.benchmark = True
    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model.cuda().eval()
    loaded, missing, unexpected = load_compatible(model, args.checkpoint)
    input_channels = args.input_channels
    if input_channels is None:
        input_channels = 6 if cfg.yaml_cfg.get("DFINE", {}).get("rgbt_enabled") else 3
    sample = torch.randn(
        args.batch_size,
        input_channels,
        args.height,
        args.width,
        device="cuda",
    )
    use_amp = args.precision == "fp16"

    with torch.inference_mode():
        for _ in range(args.warmup):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                model(sample)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(args.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                model(sample)
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))

    result = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": torch.cuda.get_device_name(),
        "input_shape": [args.batch_size, input_channels, args.height, args.width],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "precision": args.precision,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "latency_ms_mean": statistics.fmean(timings),
        "latency_ms_median": statistics.median(timings),
        "latency_ms_p90": percentile(timings, 0.90),
        "latency_ms_p95": percentile(timings, 0.95),
        "throughput_fps": args.batch_size * 1000.0 / statistics.fmean(timings),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "loaded_tensors": loaded,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
