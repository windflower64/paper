"""Frozen QDMF v1 T1--T10 and 5+1 optimizer-step acceptance run."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state, prefixes):
    digest = hashlib.sha256()
    for name in sorted(state):
        if name.startswith(prefixes):
            value = state[name].detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def move_targets(targets, device):
    return [
        {key: value.to(device) if isinstance(value, torch.Tensor) else value
         for key, value in target.items()}
        for target in targets
    ]


def qdmf_gradient_report(model):
    groups = {
        "projection": ("qdmf.feature_projections.",),
        "attention": ("qdmf.local_attention.",),
        "scale_mlp": ("qdmf.scale_mlp.",),
        "gate_mlp": ("qdmf.gate_", "qdmf.gate_mlp."),
        "output": ("qdmf.output_projection.",),
        "thermal": ("thermal_backbone.", "thermal_encoder."),
    }
    report = {}
    for group, prefixes in groups.items():
        values = [
            parameter.grad.detach().float()
            for name, parameter in model.named_parameters()
            if name.startswith(prefixes) and parameter.grad is not None
        ]
        report[group] = {
            "tensors": len(values),
            "norm": float(sum(value.square().sum() for value in values).sqrt())
            if values else 0.0,
            "finite": all(torch.isfinite(value).all().item() for value in values),
        }
    return report


def build_runtime(repo, config_path, init_path, seed):
    from src.core import YAMLConfig

    seed_all(seed)
    cfg = YAMLConfig(str(config_path.resolve()))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
    source = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    current = model.state_dict()
    compatible = {
        key: value for key, value in source.items()
        if key in current and current[key].shape == value.shape
    }
    incompatible = model.load_state_dict(compatible, strict=False)
    missing = sorted(incompatible.missing_keys)
    allowed = sorted(
        key for key in current
        if key.startswith(("qdmf.", "stql_qcer_query.", "stql_pixel_projection."))
    )
    unexpected = sorted(incompatible.unexpected_keys)
    if missing != allowed or unexpected:
        raise RuntimeError(
            f"checkpoint whitelist failed: missing={missing}, allowed={allowed}, "
            f"unexpected={unexpected}"
        )
    device = torch.device("cuda")
    model = model.to(device)
    criterion = cfg.criterion.to(device)
    optimizer = cfg.optimizer
    scheduler = cfg.lr_scheduler
    warmup = cfg.lr_warmup_scheduler
    ema = cfg.ema.to(device)
    scaler = cfg.scaler
    return {
        "cfg": cfg, "model": model, "criterion": criterion,
        "optimizer": optimizer, "scheduler": scheduler, "warmup": warmup,
        "ema": ema, "scaler": scaler, "loader": cfg.train_dataloader,
        "val_loader": cfg.val_dataloader, "missing": missing,
    }


def run_updates(runtime, count, start_micro_step=0):
    model = runtime["model"]
    criterion = runtime["criterion"]
    optimizer = runtime["optimizer"]
    scaler = runtime["scaler"]
    loader = runtime["loader"]
    accumulation = int(runtime["cfg"].yaml_cfg.get("gradient_accumulation_steps", 1))
    model.train()
    model.set_training_epoch(0)
    criterion.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    rows = []
    micro_step = start_micro_step
    for update in range(count):
        started = time.perf_counter()
        micro_rows = []
        for accumulation_index in range(accumulation):
            try:
                samples, targets = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                samples, targets = next(iterator)
            samples = samples.cuda(non_blocking=False)
            targets = move_targets(targets, samples.device)
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(samples, targets=targets)
            with torch.autocast("cuda", enabled=False):
                losses = criterion(
                    outputs, targets, epoch=0, step=micro_step,
                    global_step=micro_step, epoch_step=len(loader),
                )
                total = sum(losses.values())
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite loss at {micro_step}: {losses}")
            scaler.scale(total / accumulation).backward()
            keys = (
                "qdmf_gate_mean", "qdmf_gate_std", "qdmf_gate_p10",
                "qdmf_gate_p50", "qdmf_gate_p90", "qdmf_gate_s8",
                "qdmf_gate_s16", "qdmf_gate_s32", "qdmf_scale_weight_s8",
                "qdmf_scale_weight_s16", "qdmf_scale_weight_s32",
                "qdmf_delta_abs_mean", "qdmf_delta_query_norm_ratio",
                "qdmf_logits_delta_abs_mean", "qdmf_boxes_delta_abs_mean",
                "qdmf_missing_residual_abs_max", "qdmf_gate_small_mean",
                "qdmf_gate_medium_mean", "qdmf_gate_matched_mean",
                "qdmf_gate_unmatched_mean",
            )
            micro_rows.append({
                "micro_step": micro_step,
                "loss": float(total.detach()),
                **{
                    key: float(outputs[key].detach())
                    for key in keys if key in outputs
                },
            })
            micro_step += 1
            del samples, targets, outputs, losses, total
        scaler.unscale_(optimizer)
        gradients = qdmf_gradient_report(model)
        required = ("projection", "attention", "scale_mlp", "gate_mlp", "output")
        if any(gradients[key]["tensors"] == 0 or not gradients[key]["finite"] for key in required):
            raise RuntimeError(f"QDMF gradient acceptance failed: {gradients}")
        if gradients["thermal"]["tensors"]:
            raise RuntimeError("detached/frozen thermal stream received gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        runtime["ema"].update(model)
        if runtime["warmup"] is not None:
            runtime["warmup"].step()
        torch.cuda.synchronize()
        row = {
            "optimizer_step": update + 1,
            "seconds": time.perf_counter() - started,
            "micro_batches": micro_rows,
            "gradients": gradients,
            "ema_updates": int(runtime["ema"].updates),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows, micro_step


@torch.no_grad()
def small_validation(runtime):
    model = runtime["ema"].module
    model.eval()
    samples, _targets = next(iter(runtime["val_loader"]))
    samples = samples.cuda(non_blocking=False)
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = model(samples)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "batch": int(samples.shape[0]),
        "seconds": elapsed,
        "images_per_second": float(samples.shape[0] / elapsed),
        "pred_logits_finite": bool(torch.isfinite(outputs["pred_logits"]).all()),
        "pred_boxes_finite": bool(torch.isfinite(outputs["pred_boxes"]).all()),
        "shape_logits": list(outputs["pred_logits"].shape),
        "shape_boxes": list(outputs["pred_boxes"].shape),
        "sam_cache_required": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    sys.path.insert(0, str(args.repo.resolve()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    runtime = build_runtime(args.repo, args.config, args.init, args.seed)
    model = runtime["model"]
    physical_batch = int(runtime["loader"].batch_size)
    accumulation = int(runtime["cfg"].yaml_cfg["gradient_accumulation_steps"])
    qdmf_parameters = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if name.startswith("qdmf.")
    )
    thermal_before = state_hash(
        model.state_dict(), ("thermal_backbone.", "thermal_encoder.")
    )
    output_before = model.qdmf.output_projection.weight.detach().cpu().clone()
    rows, next_micro = run_updates(runtime, 5)
    thermal_after = state_hash(
        model.state_dict(), ("thermal_backbone.", "thermal_encoder.")
    )
    output_update = float(
        (model.qdmf.output_projection.weight.detach().cpu() - output_before).abs().max()
    )
    if thermal_before != thermal_after or output_update == 0.0:
        raise RuntimeError("parameter update/frozen thermal acceptance failed")

    checkpoint_path = args.output_dir / "qdmf_preflight_step5.pth"
    checkpoint = {
        "model": model.state_dict(),
        "criterion": runtime["criterion"].state_dict(),
        "optimizer": runtime["optimizer"].state_dict(),
        "scheduler": runtime["scheduler"].state_dict(),
        "warmup": runtime["warmup"].state_dict() if runtime["warmup"] else None,
        "ema": runtime["ema"].state_dict(),
        "scaler": runtime["scaler"].state_dict(),
        "next_micro_step": next_micro,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_digest = sha256(checkpoint_path)
    probe = model.qdmf.output_projection.weight.detach().cpu().clone()
    progress_probe = model.qdmf.training_progress.detach().cpu().clone()

    del runtime, model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    runtime = build_runtime(args.repo, args.config, args.init, args.seed)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    runtime["model"].load_state_dict(saved["model"], strict=True)
    runtime["criterion"].load_state_dict(saved["criterion"], strict=True)
    runtime["optimizer"].load_state_dict(saved["optimizer"])
    runtime["scheduler"].load_state_dict(saved["scheduler"])
    if runtime["warmup"] is not None:
        runtime["warmup"].load_state_dict(saved["warmup"])
    runtime["ema"].load_state_dict(saved["ema"], strict=True)
    runtime["scaler"].load_state_dict(saved["scaler"])
    if not torch.equal(runtime["model"].qdmf.output_projection.weight.cpu(), probe):
        raise RuntimeError("QDMF checkpoint restore mismatch")
    if not torch.equal(runtime["model"].qdmf.training_progress.cpu(), progress_probe):
        raise RuntimeError("QDMF warmup progress restore mismatch")
    resume_rows, final_micro = run_updates(runtime, 1, int(saved["next_micro_step"]))
    validation = small_validation(runtime)
    ema_names = set(dict(runtime["ema"].module.named_parameters()))
    model_qdmf_names = {
        name for name, _ in runtime["model"].named_parameters()
        if name.startswith("qdmf.")
    }
    if not model_qdmf_names.issubset(ema_names):
        raise RuntimeError("EMA is missing QDMF parameters")

    report = {
        "schema": "qdmf_v1_preflight_5plus1",
        "status": "PASS",
        "config": str(args.config.resolve()),
        "init": str(args.init.resolve()),
        "init_sha256": sha256(args.init),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_missing_whitelist": runtime["missing"],
        "physical_batch": physical_batch,
        "accumulation": accumulation,
        "effective_batch": physical_batch * accumulation,
        "qdmf_parameters": qdmf_parameters,
        "qdmf_output_max_update": output_update,
        "thermal_state_unchanged": thermal_before == thermal_after,
        "ema_contains_all_qdmf_parameters": True,
        "warmup_progress_restored": True,
        "final_micro_step": final_micro,
        "steps": rows,
        "resume_step": resume_rows,
        "validation": validation,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    report_path = args.output_dir / "qdmf_preflight_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
