#!/usr/bin/env python3
"""GPU smoke test for official and Anti-UAV-6K D-FINE-N configurations."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def describe_output(output):
    if isinstance(output, dict):
        return {key: tuple(value.shape) if torch.is_tensor(value) else type(value).__name__ for key, value in output.items()}
    return type(output).__name__


def timed_forward(model, images, repeats=20):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(5):
            output = model(images)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            output = model(images)
        torch.cuda.synchronize()
    return output, (time.perf_counter() - start) * 1000 / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve(strict=True)
    sys.path.insert(0, str(repo))
    from src.core import YAMLConfig  # noqa: PLC0415

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")["model"]
    print("device", torch.cuda.get_device_name(0), flush=True)

    official_cfg = YAMLConfig(str(repo / "configs/dfine/dfine_hgnetv2_n_coco.yml"))
    official_cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    official_model = official_cfg.model
    official_result = official_model.load_state_dict(checkpoint, strict=True)
    print("official_strict_load", official_result, flush=True)
    official_model = official_model.to(device).eval()
    official_input = torch.rand(1, 3, 640, 640, device=device)
    torch.cuda.reset_peak_memory_stats()
    official_output, official_ms = timed_forward(official_model, official_input)
    print("official_output", describe_output(official_output), flush=True)
    print("official_fp16_ms_batch1", round(official_ms, 3), flush=True)
    print("official_peak_mib", round(torch.cuda.max_memory_allocated() / 1024**2, 2), flush=True)
    del official_model, official_input, official_output
    torch.cuda.empty_cache()

    custom_cfg = YAMLConfig(str(repo / "experiments/dfine/dfine_n_visible_640x512.yml"))
    custom_cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    custom_cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = 1
    custom_cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    custom_model = custom_cfg.model
    custom_state = custom_model.state_dict()
    compatible = {key: value for key, value in checkpoint.items() if key in custom_state and custom_state[key].shape == value.shape}
    skipped = sorted(key for key, value in checkpoint.items() if key not in custom_state or custom_state[key].shape != value.shape)
    result = custom_model.load_state_dict(compatible, strict=False)
    print("custom_pretrained_loaded", len(compatible), "of", len(custom_state), flush=True)
    print("custom_skipped_checkpoint", len(skipped), skipped[:12], flush=True)
    print("custom_missing_after_load", len(result.missing_keys), result.missing_keys[:12], flush=True)
    custom_model = custom_model.to(device).eval()
    val_images, targets = next(iter(custom_cfg.val_dataloader))
    val_images = val_images.to(device)
    torch.cuda.reset_peak_memory_stats()
    custom_output, custom_ms = timed_forward(custom_model, val_images)
    print("custom_input", tuple(val_images.shape), flush=True)
    print("custom_output", describe_output(custom_output), flush=True)
    print("custom_fp16_ms_batch1", round(custom_ms, 3), flush=True)
    print("custom_peak_mib", round(torch.cuda.max_memory_allocated() / 1024**2, 2), flush=True)
    print("target_boxes", tuple(targets[0]["boxes"].shape), "labels", targets[0]["labels"].tolist(), flush=True)
    print("GPU_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
