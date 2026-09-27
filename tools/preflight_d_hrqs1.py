"""CUDA structural and gradient preflight for D-HRQS1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def checkpoint_state(checkpoint):
    if "ema" in checkpoint:
        return checkpoint["ema"]["module"]
    return checkpoint.get("model", checkpoint)


def matched_state(model, source):
    destination = model.state_dict()
    return {
        key: value
        for key, value in source.items()
        if key in destination and destination[key].shape == value.shape
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    config = YAMLConfig(str(args.config.resolve()))
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = config.model
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source = checkpoint_state(checkpoint)
    loaded = matched_state(model, source)
    incompatible = model.load_state_dict(loaded, strict=False)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("decoder.hrqs_adapter")
        and ".pat_sf." not in key
        and key not in {"decoder.anchors", "decoder.valid_mask"}
        and "score_head" not in key
        and "denoising_class_embed" not in key
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected load mismatch: missing={invalid_missing[:10]}, "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )

    if model.backbone.return_idx != [1, 2, 3]:
        raise RuntimeError(f"unexpected return_idx={model.backbone.return_idx}")
    if not model.decoder.hrqs_enabled or model.decoder.hrqs_adapter is None:
        raise RuntimeError("HRQS adapter was not constructed")
    if len(model.encoder.in_channels) != 2:
        raise RuntimeError("HRQS must preserve the two-level HybridEncoder")

    device = torch.device("cuda")
    model.to(device).train()
    criterion = config.criterion.to(device)
    optimizer = config.optimizer
    loader = config.train_dataloader
    samples, targets = next(iter(loader))
    samples = samples.to(device)
    targets = [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]

    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(samples, targets=targets)
    with torch.autocast(device_type="cuda", enabled=False):
        losses = criterion(
            outputs,
            targets,
            epoch=0,
            step=0,
            global_step=0,
            epoch_step=len(loader),
        )
        total_loss = sum(losses.values())
    encoder_matches = criterion.matcher(
        outputs["enc_aux_outputs"][0], targets
    )["indices"]
    ordinary_query_count = model.decoder.num_queries - model.decoder.hrqs_num_queries
    matched_hrqs_queries = sum(
        int((source_indices >= ordinary_query_count).sum().cpu())
        for source_indices, _ in encoder_matches
    )
    matched_encoder_queries = sum(
        int(source_indices.numel()) for source_indices, _ in encoder_matches
    )
    if not torch.isfinite(total_loss):
        raise RuntimeError(f"non-finite loss: {total_loss}")
    total_loss.backward()

    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("decoder.hrqs_adapter")
    }
    missing_gradients = [name for name, gradient in gradients.items() if gradient is None]
    nonzero_gradient_tensors = sum(
        int(float(gradient.detach().abs().max().cpu()) > 0.0)
        for gradient in gradients.values()
        if gradient is not None
    )
    selected = model.decoder.last_hrqs_selected_count
    if selected is None:
        raise RuntimeError("HRQS did not report selected queries")
    selected_mean = float(selected.float().mean().cpu())
    if selected_mean <= 0.0:
        raise RuntimeError("HRQS admitted no S8 query and cannot learn")
    if missing_gradients or nonzero_gradient_tensors == 0:
        raise RuntimeError(
            f"HRQS gradient failure: missing={missing_gradients}, "
            f"nonzero={nonzero_gradient_tensors}"
        )
    if matched_hrqs_queries == 0:
        raise RuntimeError(
            "HRQS received gradients but no S8 query matched a target in the "
            "preflight batch"
        )

    result = {
        "status": "PASS",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "batch_size": int(loader.batch_size),
        "input_shape": list(samples.shape),
        "loss": float(total_loss.detach().cpu()),
        "hrqs_parameters": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("decoder.hrqs_adapter")
        ),
        "selected_s8_queries_mean": selected_mean,
        "selected_s8_queries_min": int(selected.min().cpu()),
        "selected_s8_queries_max": int(selected.max().cpu()),
        "selected_s8_query_ratio_mean": selected_mean / model.decoder.num_queries,
        "matched_s8_queries": matched_hrqs_queries,
        "matched_encoder_queries": matched_encoder_queries,
        "hrqs_gradient_tensors": len(gradients),
        "hrqs_nonzero_gradient_tensors": nonzero_gradient_tensors,
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
