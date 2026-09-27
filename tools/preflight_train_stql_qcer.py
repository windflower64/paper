"""Run the v1.1 real-data short train and checkpoint-resume acceptance test.

This intentionally stops after five optimizer updates (ten micro-batches with
accumulation=2), writes a dedicated preflight checkpoint, reconstructs the
runtime from that checkpoint, and performs one additional optimizer update.
It never starts the 20-epoch B0--B5 experiment queue.
"""

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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state, prefixes) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        if not key.startswith(prefixes):
            continue
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def gradient_norms(model):
    buckets = {
        "shared_query": ("stql_qcer_query.",),
        "stql_pixel": ("stql_pixel_projection.",),
        "qcer_objectness": ("qcer.objectness_",),
        "qcer_key": ("qcer.key_proj.",),
        "qcer_value": ("qcer.value_proj.",),
        "qcer_gate": ("qcer.gate.",),
        "qcer_output": ("qcer.output.",),
        "rgb_backbone": ("backbone.",),
        "thermal_source": ("thermal_backbone.", "thermal_encoder."),
    }
    result = {}
    for bucket, prefixes in buckets.items():
        squared = 0.0
        tensors = 0
        for name, parameter in model.named_parameters():
            if name.startswith(prefixes) and parameter.grad is not None:
                squared += float(parameter.grad.detach().float().square().sum())
                tensors += 1
        result[bucket] = {"norm": squared ** 0.5, "gradient_tensors": tensors}
    return result


def build_runtime(repo: Path, config_path: Path, init_path: Path, seed: int):
    from src.core import YAMLConfig
    from src.solver import BaseSolver

    seed_all(seed)
    cfg = YAMLConfig(str(config_path.resolve()))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    shim = BaseSolver.__new__(BaseSolver)
    shim.model = model
    # The common checkpoint already contains the one-class heads, so this list
    # is not used.  It is provided only because BaseSolver owns the loader.
    shim.obj365_ids = list(range(80))
    shim.load_tuning_state(str(init_path.resolve()))

    device = torch.device("cuda")
    model = model.to(device)
    criterion = cfg.criterion.to(device)
    optimizer = cfg.optimizer
    lr_scheduler = cfg.lr_scheduler
    warmup = cfg.lr_warmup_scheduler
    ema = cfg.ema.to(device)
    scaler = cfg.scaler
    loader = cfg.train_dataloader
    return cfg, model, criterion, optimizer, lr_scheduler, warmup, ema, scaler, loader


def run_updates(
    model,
    criterion,
    optimizer,
    warmup,
    ema,
    scaler,
    loader,
    optimizer_updates,
    start_micro_step,
    accumulation_steps,
):
    model.train()
    criterion.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    rows = []
    micro_step = int(start_micro_step)
    completed_updates = 0
    update_started = time.perf_counter()
    while completed_updates < optimizer_updates:
        micro_rows = []
        for accumulation_index in range(accumulation_steps):
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
                loss_dict = criterion(
                    outputs,
                    targets,
                    epoch=0,
                    step=micro_step,
                    global_step=micro_step,
                    epoch_step=len(loader),
                )
                total = sum(loss_dict.values())
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite loss at micro-step {micro_step}: {loss_dict}")
            scaler.scale(total / accumulation_steps).backward()
            micro_rows.append({
                "micro_step": micro_step,
                "accumulation_index": accumulation_index,
                "total_loss": float(total.detach()),
                "loss_stql": float(loss_dict.get("loss_stql", total.new_zeros(()))),
                "loss_qcer_objectness": float(
                    loss_dict.get("loss_qcer_objectness", total.new_zeros(()))
                ),
                "stql_schedule": float(
                    outputs.get("stql_schedule", total.new_zeros(())).detach()
                ),
                "stql_valid_instances": float(
                    outputs.get("stql_valid_instances", total.new_zeros(())).detach()
                ),
                "qcer_delta_abs": float(
                    outputs.get("qcer_delta_logits", total.new_zeros(())).detach().abs().mean()
                ),
                "qcer_gate_mean": float(
                    outputs.get("qcer_gate", total.new_zeros(())).detach().mean()
                ),
                "qcer_positive_tokens": float(
                    outputs.get("qcer_positive_tokens", total.new_zeros(())).detach()
                ),
                "qcer_fallback_tokens": float(
                    outputs.get("qcer_fallback_tokens", total.new_zeros(())).detach()
                ),
            })
            del samples, targets, outputs, loss_dict, total
            micro_step += 1

        scaler.unscale_(optimizer)
        gradients = gradient_norms(model)
        if gradients["thermal_source"]["gradient_tensors"] != 0:
            raise RuntimeError("frozen thermal source unexpectedly received gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)
        if warmup is not None:
            warmup.step()
        completed_updates += 1
        torch.cuda.synchronize()
        rows.append({
            "optimizer_step": completed_updates,
            "elapsed_seconds": time.perf_counter() - update_started,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "gradients": gradients,
            "micro_batches": micro_rows,
            "ema_updates": int(ema.updates),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        update_started = time.perf_counter()
    return rows, micro_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--optimizer-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the AMP short-training acceptance test")
    if args.optimizer_steps < 5:
        raise ValueError("acceptance protocol requires at least five optimizer steps")

    sys.path.insert(0, str(args.repo.resolve()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    runtime = build_runtime(args.repo, args.config, args.init, args.seed)
    cfg, model, criterion, optimizer, scheduler, warmup, ema, scaler, loader = runtime
    accumulation_steps = int(cfg.yaml_cfg.get("gradient_accumulation_steps", 1))
    physical_batch = int(loader.batch_size)
    if physical_batch != 16 or accumulation_steps != 2:
        raise RuntimeError(
            f"protocol drift: physical_batch={physical_batch}, "
            f"accumulation={accumulation_steps}"
        )
    thermal_before = state_sha256(
        model.state_dict(), ("thermal_backbone.", "thermal_encoder.")
    )
    output_before = model.qcer.output.weight.detach().cpu().clone()
    rows, next_micro_step = run_updates(
        model, criterion, optimizer, warmup, ema, scaler, loader,
        args.optimizer_steps, 0, accumulation_steps,
    )
    thermal_after = state_sha256(
        model.state_dict(), ("thermal_backbone.", "thermal_encoder.")
    )
    if thermal_before != thermal_after:
        raise RuntimeError("frozen thermal parameters or buffers changed")
    output_update = float(
        (model.qcer.output.weight.detach().cpu() - output_before).abs().max()
    )
    if output_update <= 0:
        raise RuntimeError("QCER output layer did not update")

    checkpoint_path = args.output_dir / "preflight_step5.pth"
    checkpoint = {
        "schema": "stql_qcer_short_train_v1",
        "model": model.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "lr_warmup_scheduler": warmup.state_dict() if warmup is not None else None,
        "ema": ema.state_dict(),
        "scaler": scaler.state_dict(),
        "optimizer_steps": args.optimizer_steps,
        "next_micro_step": next_micro_step,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = file_sha256(checkpoint_path)
    resume_probe = model.qcer.output.weight.detach().cpu().clone()

    del checkpoint, runtime, cfg, model, criterion, optimizer, scheduler
    del warmup, ema, scaler, loader
    gc.collect()
    torch.cuda.empty_cache()

    runtime = build_runtime(args.repo, args.config, args.init, args.seed)
    cfg, model, criterion, optimizer, scheduler, warmup, ema, scaler, loader = runtime
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    criterion.load_state_dict(saved["criterion"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["lr_scheduler"])
    if warmup is not None and saved["lr_warmup_scheduler"] is not None:
        warmup.load_state_dict(saved["lr_warmup_scheduler"])
    ema.load_state_dict(saved["ema"], strict=True)
    scaler.load_state_dict(saved["scaler"])
    if not torch.equal(model.qcer.output.weight.detach().cpu(), resume_probe):
        raise RuntimeError("checkpoint resume did not restore QCER output exactly")
    resume_rows, final_micro_step = run_updates(
        model, criterion, optimizer, warmup, ema, scaler, loader,
        1, int(saved["next_micro_step"]), accumulation_steps,
    )

    report = {
        "schema": "stql_qcer_short_train_report_v1",
        "status": "PASS",
        "config": str(args.config.resolve()),
        "common_init": str(args.init.resolve()),
        "physical_batch": physical_batch,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch": physical_batch * accumulation_steps,
        "amp_enabled": bool(scaler.is_enabled()),
        "initial_optimizer_steps": args.optimizer_steps,
        "resume_optimizer_steps": 1,
        "final_micro_step": final_micro_step,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "qcer_output_max_update_after_five_steps": output_update,
        "thermal_source_sha256_before": thermal_before,
        "thermal_source_sha256_after": thermal_after,
        "thermal_source_unchanged": thermal_before == thermal_after,
        "steps": rows,
        "resume_steps": resume_rows,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    report_path = args.output_dir / "short_train_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
