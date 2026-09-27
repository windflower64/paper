#!/usr/bin/env python3
"""Cache frozen Visible and Thermal detector outputs for the M-A2 audit.

M-A2 is an inference-first upper-bound experiment.  This script deliberately
does not contain any fusion logic: it runs the two mature single-modality
detectors on exactly paired samples and stores their raw logits and boxes.
Subsequent matching/calibration experiments can therefore be repeated without
rerunning either detector or changing their weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--paired-config", type=Path, required=True)
    parser.add_argument("--visible-config", type=Path, required=True)
    parser.add_argument("--visible-checkpoint", type=Path, required=True)
    parser.add_argument("--thermal-config", type=Path, required=True)
    parser.add_argument("--thermal-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_weights(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    ema = checkpoint.get("ema") if isinstance(checkpoint, dict) else None
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"], "ema.module"
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"], "model"
    if isinstance(checkpoint, dict):
        return checkpoint, "raw"
    raise TypeError(f"Unsupported checkpoint payload: {type(checkpoint)!r}")


def reset_yaml_load_accumulator():
    # Upstream yaml_utils.load_config uses a mutable default dictionary.  That
    # is harmless for the usual one-config-per-process CLI, but this audit
    # intentionally builds three independent configs in one process.  Clear
    # only that parser accumulator before every independent load.
    from src.core.yaml_utils import load_config

    defaults = load_config.__defaults__
    if not defaults or not isinstance(defaults[0], dict):
        raise RuntimeError("Unexpected load_config signature; cannot isolate configs")
    defaults[0].clear()


def build_model(YAMLConfig, config_path: Path, checkpoint_path: Path, device):
    reset_yaml_load_accumulator()
    cfg = YAMLConfig(str(config_path.resolve()))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    weights, field = checkpoint_weights(checkpoint_path)
    incompatible = model.load_state_dict(weights, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Strict load unexpectedly returned incompatibilities: {incompatible}"
        )
    model.to(device).eval()
    return model, field


def build_loader(YAMLConfig, config_path: Path, split: str, batch_size: int, num_workers: int):
    reset_yaml_load_accumulator()
    cfg = YAMLConfig(str(config_path.resolve()))
    base = cfg.train_dataloader if split == "train" else cfg.val_dataloader
    dataset = base.dataset
    collate_fn = base.collate_fn

    # The selected M-A1 reader config contains deterministic resize/convert
    # operations only.  A fixed late epoch also prevents any inherited
    # schedule from accidentally switching behavior across cache runs.
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(1000000)
    if hasattr(collate_fn, "set_epoch"):
        collate_fn.set_epoch(1000000)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return loader


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    for path in (
        args.repo,
        args.paired_config,
        args.visible_config,
        args.visible_checkpoint,
        args.thermal_config,
        args.thermal_checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the two-detector cache")
    device = torch.device("cuda")
    # Model configs must be materialized before the paired-reader config.
    # D-FINE's component registry is process-global; constructing the RGB-T
    # reader config first can otherwise leak its DFINE options into a later
    # single-modality model construction.
    visible, visible_field = build_model(
        YAMLConfig, args.visible_config, args.visible_checkpoint, device
    )
    thermal, thermal_field = build_model(
        YAMLConfig, args.thermal_config, args.thermal_checkpoint, device
    )
    loader = build_loader(
        YAMLConfig,
        args.paired_config,
        args.split,
        args.batch_size,
        args.num_workers,
    )

    records = {
        "image_ids": [],
        "orig_sizes": [],
        "file_names": [],
        "visible_logits": [],
        "visible_boxes": [],
        "thermal_logits": [],
        "thermal_boxes": [],
    }
    start = time.time()
    expected_batches = len(loader)
    if args.max_batches is not None:
        expected_batches = min(expected_batches, args.max_batches)

    amp_context = lambda: torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=args.amp
    )
    with torch.inference_mode():
        for batch_index, (samples, targets) in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            if samples.ndim != 4 or samples.shape[1] != 6:
                raise RuntimeError(
                    f"M-A2 requires paired [B,6,H,W] samples, got {tuple(samples.shape)}"
                )
            samples = samples.to(device, non_blocking=True)
            with amp_context():
                visible_output = visible(samples[:, :3])
            visible_logits = visible_output["pred_logits"].detach().float().cpu()
            visible_boxes = visible_output["pred_boxes"].detach().float().cpu()
            del visible_output
            with amp_context():
                thermal_output = thermal(samples[:, 3:])
            thermal_logits = thermal_output["pred_logits"].detach().float().cpu()
            thermal_boxes = thermal_output["pred_boxes"].detach().float().cpu()
            del thermal_output, samples

            for name, tensor in (
                ("visible_logits", visible_logits),
                ("visible_boxes", visible_boxes),
                ("thermal_logits", thermal_logits),
                ("thermal_boxes", thermal_boxes),
            ):
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f"Non-finite values in {name}, batch {batch_index}")

            records["visible_logits"].append(visible_logits)
            records["visible_boxes"].append(visible_boxes)
            records["thermal_logits"].append(thermal_logits)
            records["thermal_boxes"].append(thermal_boxes)
            records["image_ids"].append(
                torch.cat([target["image_id"].reshape(-1).cpu() for target in targets])
            )
            records["orig_sizes"].append(
                torch.stack([target["orig_size"].reshape(2).cpu() for target in targets])
            )
            records["file_names"].extend(
                Path(target["image_path"]).name for target in targets
            )

            if batch_index == 0 or (batch_index + 1) % 20 == 0 or batch_index + 1 == expected_batches:
                seen = sum(chunk.shape[0] for chunk in records["image_ids"])
                elapsed = time.time() - start
                print(
                    f"[{batch_index + 1}/{expected_batches}] samples={seen} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    tensor_keys = (
        "image_ids",
        "orig_sizes",
        "visible_logits",
        "visible_boxes",
        "thermal_logits",
        "thermal_boxes",
    )
    for key in tensor_keys:
        records[key] = torch.cat(records[key], dim=0)

    count = int(records["image_ids"].shape[0])
    if args.max_batches is None and count != len(loader.dataset):
        raise RuntimeError(f"Expected {len(loader.dataset)} samples, cached {count}")
    if len(records["file_names"]) != count:
        raise RuntimeError("file_names and tensor record counts disagree")
    if torch.unique(records["image_ids"]).numel() != count:
        raise RuntimeError("Duplicate image ids in sequential cache")

    payload = {
        "metadata": {
            "schema": "ma2_detector_cache_v1",
            "split": args.split,
            "samples": count,
            "batch_size": args.batch_size,
            "amp": bool(args.amp),
            "seed": args.seed,
            "paired_config": str(args.paired_config.resolve()),
            "visible_config": str(args.visible_config.resolve()),
            "visible_checkpoint": str(args.visible_checkpoint.resolve()),
            "visible_checkpoint_sha256": sha256(args.visible_checkpoint),
            "visible_checkpoint_field": visible_field,
            "thermal_config": str(args.thermal_config.resolve()),
            "thermal_checkpoint": str(args.thermal_checkpoint.resolve()),
            "thermal_checkpoint_sha256": sha256(args.thermal_checkpoint),
            "thermal_checkpoint_field": thermal_field,
            "elapsed_seconds": time.time() - start,
        },
        **records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    summary = {
        **payload["metadata"],
        "visible_logits_shape": list(records["visible_logits"].shape),
        "visible_boxes_shape": list(records["visible_boxes"].shape),
        "thermal_logits_shape": list(records["thermal_logits"].shape),
        "thermal_boxes_shape": list(records["thermal_boxes"].shape),
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
