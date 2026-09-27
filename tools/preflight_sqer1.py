"""Run the frozen S-QER1 preflight; never launches full training."""

from __future__ import annotations

import argparse
import ast
import copy
import contextlib
import gc
import hashlib
import json
import io
import random
import shutil
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state, prefix: str) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        if not name.startswith(prefix):
            continue
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def build_config(repo: Path, config_path: Path):
    sys.path.insert(0, str(repo.resolve()))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(config_path.resolve()))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["use_amp"] = True
    return cfg


def load_tuning_state(model, init_path: Path) -> None:
    from src.solver import BaseSolver
    import torch

    shim = BaseSolver.__new__(BaseSolver)
    shim.model = model
    shim.obj365_ids = list(range(80))
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        shim.load_tuning_state(str(init_path.resolve()))
    text = captured.getvalue()
    marker = "Load model.state_dict, "
    if marker not in text:
        raise RuntimeError("BaseSolver did not report the tuning checkpoint load")
    info = ast.literal_eval(text.split(marker, 1)[1].strip())
    checkpoint = torch.load(str(init_path.resolve()), map_location="cpu")
    source = checkpoint.get("ema", {}).get("module", checkpoint.get("model", {}))
    current = model.state_dict()
    matched = [
        key for key, value in current.items()
        if key in source and value.shape == source[key].shape
    ]
    expected_missing = [
        key for key in info["missed"]
        if key.startswith(("sqer.", "sd2_conditioner."))
    ]
    return {
        "checkpoint_tensor_count": len(source),
        "shape_matched_tensor_count": len(matched),
        "model_missing_tensor_count": len(info["missed"]),
        "shape_mismatch_tensor_count": len(info["unmatched"]),
        "expected_new_module_missing_count": len(expected_missing),
        "other_missing_sample": [
            key for key in info["missed"]
            if not key.startswith(("sqer.", "sd2_conditioner."))
        ][:20],
    }


def compare_configs(repo: Path, sam_config: Path, box_config: Path):
    sam_cfg = build_config(repo, sam_config)
    box_cfg = build_config(repo, box_config)
    left = copy.deepcopy(sam_cfg.yaml_cfg)
    right = copy.deepcopy(box_cfg.yaml_cfg)
    left["output_dir"] = right["output_dir"] = "<paired-output>"
    left["DFINECriterion"]["sqer_supervision"] = "<paired-teacher>"
    right["DFINECriterion"]["sqer_supervision"] = "<paired-teacher>"
    if left != right:
        raise RuntimeError("SAM/BOX resolved configurations differ beyond teacher/output")
    return sam_cfg, {
        "only_allowed_differences": [
            "output_dir",
            "DFINECriterion.sqer_supervision",
        ],
        "resolved_config_sha256_sam": hashlib.sha256(
            json.dumps(sam_cfg.yaml_cfg, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "resolved_config_sha256_box": hashlib.sha256(
            json.dumps(box_cfg.yaml_cfg, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
    }


def measure_query_coverage(model, matcher, loader, device):
    groups = {
        "all": [0, 0],
        "small_lt32": [0, 0],
        "lt8": [0, 0],
        "8_to_16": [0, 0],
        "16_to_32": [0, 0],
        "ge32": [0, 0],
    }
    image_coverage = []
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for batch_index, (samples, targets) in enumerate(loader):
            samples = samples.to(device, non_blocking=False)
            targets = move_targets(targets, device)
            outputs = model(samples)
            matches = matcher(outputs, targets)["indices"]
            selected = outputs["sqer_query_indices"]
            for image_index, (source_ids, target_ids) in enumerate(matches):
                target_boxes = targets[image_index]["boxes"]
                image_hit, image_total = 0, 0
                for source_id, target_id in zip(source_ids.tolist(), target_ids.tolist()):
                    hit = bool((selected[image_index] == source_id).any())
                    box = target_boxes[target_id]
                    size = float(
                        (box[2] * box[3] * 512.0 * 640.0).clamp_min(0).sqrt()
                    )
                    bucket_names = ["all"]
                    if size < 32:
                        bucket_names.append("small_lt32")
                    if size < 8:
                        bucket_names.append("lt8")
                    elif size < 16:
                        bucket_names.append("8_to_16")
                    elif size < 32:
                        bucket_names.append("16_to_32")
                    else:
                        bucket_names.append("ge32")
                    for name in bucket_names:
                        groups[name][1] += 1
                        groups[name][0] += int(hit)
                    image_total += 1
                    image_hit += int(hit)
                if image_total:
                    image_coverage.append(image_hit / image_total)
            if (batch_index + 1) % 50 == 0:
                print(
                    f"SQER coverage {batch_index + 1}/{len(loader)} batches",
                    flush=True,
                )
            del samples, targets, outputs, matches
    result = {
        name: {
            "covered": values[0],
            "matched": values[1],
            "coverage": values[0] / max(values[1], 1),
        }
        for name, values in groups.items()
    }
    result["image_mean_coverage"] = sum(image_coverage) / max(len(image_coverage), 1)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def gradient_bucket_norm(model, prefixes):
    squared = 0.0
    tensors = 0
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes) and parameter.grad is not None:
            squared += float(parameter.grad.detach().float().square().sum())
            tensors += 1
    return {"norm": squared ** 0.5, "gradient_tensors": tensors}


def loss_gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared = sum(
        float(value.detach().float().square().sum())
        for value in gradients
        if value is not None
    )
    return squared ** 0.5


def sqer_mac_estimate(module, batch: int = 1, height: int = 128, width: int = 160):
    total = 0
    handles = []

    def conv_hook(layer, _inputs, output):
        nonlocal total
        total += int(
            output.numel()
            * (layer.in_channels // layer.groups)
            * layer.kernel_size[0]
            * layer.kernel_size[1]
        )

    def linear_hook(layer, _inputs, output):
        nonlocal total
        total += int(output.numel() * layer.in_features)

    for layer in module.modules():
        if isinstance(layer, torch.nn.Conv2d):
            handles.append(layer.register_forward_hook(conv_hook))
        elif isinstance(layer, torch.nn.Linear):
            handles.append(layer.register_forward_hook(linear_hook))
    device = next(module.parameters()).device
    s4 = torch.zeros(batch, module.s4_projection[0].in_channels, height, width, device=device)
    s8 = torch.zeros(batch, module.s8_projection[0].in_channels, height // 2, width // 2, device=device)
    # S-QER1's quality head includes the decoder query and four box scalars;
    # S-QER2 is evidence-only, so its head has no query concatenation at all.
    # The reader stores the actual decoder query width in both variants.
    query = torch.zeros(batch, 300, module.query_dim, device=device)
    logits = torch.zeros(batch, 300, module.quality_head[-1].out_features, device=device)
    boxes = torch.full((batch, 300, 4), 0.5, device=device)
    boxes[..., 2:] = 0.1
    with torch.inference_mode():
        result = module(s4, s8, query, logits, boxes)
    for handle in handles:
        handle.remove()
    # nn.MultiheadAttention invokes F.linear directly, so count its projections
    # and score/value products explicitly for three queries x 256 tokens.
    instances = batch * module.topk
    q_len, kv_len, dim = 3, module.roi_size**2, module.dim
    mha_per_block = (
        3 * q_len * dim * dim
        + 2 * kv_len * dim * dim
        + q_len * dim * dim
        + 2 * q_len * kv_len * dim
    )
    total += len(module.query_blocks) * instances * mha_per_block
    return total


def inference_timing(model, samples, device, repeats=25):
    model.eval()
    samples = samples[:1].to(device)
    model.sqer_enabled = False
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(5):
            model(samples)
        torch.cuda.synchronize()
        base = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(samples)
            torch.cuda.synchronize()
            base.append((time.perf_counter() - start) * 1000)
        model.sqer_enabled = True
        for _ in range(5):
            model(samples)
        torch.cuda.synchronize()
        enabled = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(samples)
            torch.cuda.synchronize()
            enabled.append((time.perf_counter() - start) * 1000)
    model.sqer_enabled = True
    return {
        "batch": 1,
        "repeats": repeats,
        "base_median_ms": float(np.median(base)),
        "sqer_median_ms": float(np.median(enabled)),
        "relative_overhead_percent": float(
            (np.median(enabled) / max(np.median(base), 1e-9) - 1) * 100
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--sam-config", type=Path, required=True)
    parser.add_argument("--box-config", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("S-QER1 preflight requires CUDA")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse preflight output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    seed_all(0)

    repo = args.repo.resolve()
    sam_cfg, paired_config = compare_configs(
        repo, args.sam_config.resolve(), args.box_config.resolve()
    )
    source_files = [
        repo / "src/zoo/dfine/sam_query_evidence_reader.py",
        repo / "src/zoo/dfine/dfine.py",
        repo / "src/zoo/dfine/dfine_criterion.py",
        repo / "src/nn/backbone/hgnetv2.py",
        repo / "src/solver/det_engine.py",
        repo / "experiments/phase_s/sqer1_common_b8a4_20e.yml",
        args.sam_config.resolve(),
        args.box_config.resolve(),
    ]
    (args.output_dir / "source_snapshot").mkdir()
    for path in source_files:
        if path.is_file():
            shutil.copy2(path, args.output_dir / "source_snapshot" / path.name)
    (args.output_dir / "resolved_config_sam.json").write_text(
        json.dumps(sam_cfg.yaml_cfg, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    paired_config["source_sha256"] = {
        path.name: sha256_file(path) for path in source_files if path.is_file()
    }
    paired_config["init_checkpoint"] = str(args.init.resolve())
    paired_config["init_sha256"] = sha256_file(args.init.resolve())
    mask_records = Path(
        sam_cfg.yaml_cfg["train_dataloader"]["dataset"]["sam_mask_root"]
    ) / "records.json"
    paired_config["sam_mask_manifest"] = str(mask_records)
    paired_config["sam_mask_manifest_sha256"] = sha256_file(mask_records)

    # Same seed, same checkpoint, and the sole A/B config difference is the
    # criterion teacher.  Verify the newly introduced model tensors match.
    from src.core import YAMLConfig

    hashes = {}
    for arm_config in (args.sam_config.resolve(), args.box_config.resolve()):
        seed_all(0)
        arm_cfg = YAMLConfig(str(arm_config))
        arm_cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        arm_model = arm_cfg.model
        arm_load = load_tuning_state(arm_model, args.init.resolve())
        hashes[arm_config.name] = state_digest(arm_model.state_dict(), "sqer.")
        del arm_model, arm_cfg
        gc.collect()
    if len(set(hashes.values())) != 1:
        raise RuntimeError("SAM/BOX S-QER initial parameters are not identical")
    paired_config["initial_sqer_sha256_by_arm"] = hashes
    paired_config["base_checkpoint_load"] = arm_load

    model = sam_cfg.model
    paired_config["base_checkpoint_load_runtime"] = load_tuning_state(
        model, args.init.resolve()
    )
    device = torch.device("cuda")
    model = model.to(device)
    criterion = sam_cfg.criterion.cuda()
    optimizer = sam_cfg.optimizer
    warmup = sam_cfg.lr_warmup_scheduler
    ema = sam_cfg.ema.to(device)
    sam_cfg.yaml_cfg["use_amp"] = True
    scaler = sam_cfg.scaler
    loader = sam_cfg.train_dataloader
    paired_config["data"] = {
        "train_images": len(loader.dataset),
        "train_batches": len(loader),
        "physical_batch": loader.batch_size,
        "accumulation_steps": int(sam_cfg.yaml_cfg.get("gradient_accumulation_steps", 1)),
        "accepted_sam_ids": len(loader.dataset.sam_mask_accepted_ids),
    }

    # The output path must preserve every box and begin as exact identity.
    first_samples, _ = next(iter(loader))
    first_samples = first_samples.to(device)
    model.eval()
    model.sqer_enabled = False
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        base = model(first_samples)
        model.sqer_enabled = True
        enabled = model(first_samples)
        model.sqer_bypass = True
        bypass = model(first_samples)
    identity = {
        "enabled_logits_max_abs": float(
            (enabled["pred_logits"] - base["pred_logits"]).abs().max()
        ),
        "enabled_boxes_max_abs": float(
            (enabled["pred_boxes"] - base["pred_boxes"]).abs().max()
        ),
        "bypass_logits_max_abs": float(
            (bypass["pred_logits"] - base["pred_logits"]).abs().max()
        ),
        "bypass_boxes_max_abs": float(
            (bypass["pred_boxes"] - base["pred_boxes"]).abs().max()
        ),
        "inference_without_targets": True,
    }
    if any(value != 0 for key, value in identity.items() if key.endswith("max_abs")):
        raise RuntimeError(f"S-QER identity/bypass contract failed: {identity}")
    model.sqer_bypass = False
    paired_config["identity_checks"] = identity

    timing = inference_timing(model, first_samples, device)
    paired_config["inference_timing"] = timing
    paired_config["sqer_parameters"] = sum(p.numel() for p in model.sqer.parameters())
    paired_config["sqer_estimated_macs_per_image"] = sqer_mac_estimate(model.sqer)

    coverage = measure_query_coverage(model, criterion.matcher, loader, device)
    paired_config["topk_coverage"] = coverage
    failed_gates = []
    if coverage["all"]["coverage"] < 0.95:
        failed_gates.append("matched-target top64 coverage < 95%")
    if coverage["small_lt32"]["coverage"] < 0.90:
        failed_gates.append("small-object matched-target top64 coverage < 90%")
    if failed_gates:
        paired_config["status"] = "GATE_FAIL"
        paired_config["failed_gates"] = failed_gates
        paired_config["official_training_started"] = False
        (args.output_dir / "preflight.json").write_text(
            json.dumps(paired_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({
            "status": "GATE_FAIL",
            "failed_gates": failed_gates,
            "topk_coverage": coverage,
            "preflight": str(args.output_dir / "preflight.json"),
        }, ensure_ascii=False), flush=True)
        return

    # Two actual optimizer updates, each with four physical micro-batches.
    accumulation = int(sam_cfg.yaml_cfg.get("gradient_accumulation_steps", 1))
    if loader.batch_size != 8 or accumulation != 4:
        raise RuntimeError(
            f"protocol drift: batch={loader.batch_size}, accumulation={accumulation}"
        )
    model.train()
    criterion.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    update_rows = []
    first_gradient_ratio = None
    for update_index in range(2):
        micro_losses = []
        for micro_index in range(accumulation):
            samples, targets = next(iterator)
            samples = samples.to(device, non_blocking=False)
            targets = move_targets(targets, device)
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(samples, targets=targets)
            with torch.autocast("cuda", enabled=False):
                loss_dict = criterion(
                    outputs,
                    targets,
                    epoch=0,
                    step=update_index * accumulation + micro_index,
                    global_step=update_index * accumulation + micro_index,
                    epoch_step=len(loader),
                )
                total = sum(loss_dict.values())
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite SQER preflight loss: {loss_dict}")
            if update_index == 0 and micro_index == 0:
                sqer_params = [p for p in model.sqer.parameters() if p.requires_grad]
                detection_loss = total - loss_dict.get("loss_sqer_shape", total.new_zeros(()))
                sqer_loss = loss_dict["loss_sqer_shape"]
                detection_norm = loss_gradient_norm(detection_loss, sqer_params)
                shape_norm = loss_gradient_norm(sqer_loss, sqer_params)
                first_gradient_ratio = {
                    "shape_loss_sqer_gradient_norm": shape_norm,
                    "detection_sqer_gradient_norm": detection_norm,
                    "ratio": shape_norm / max(detection_norm, 1e-12),
                }
                # Keep the ratio as a diagnostic, not a hard gate.  S-QER2
                # deliberately routes shape supervision into attention while
                # detection gradients initially reach the zero-initialized
                # residual head; their aggregate norms are not comparable at
                # step zero.  The hard gates below verify nonzero detector and
                # evidence-path gradients after actual optimizer updates.
            scaler.scale(total / accumulation).backward()
            micro_losses.append({
                "loss_total": float(total.detach()),
                "loss_sqer_shape": float(loss_dict["loss_sqer_shape"].detach()),
                "sqer_supervised_queries": float(
                    outputs["sqer_supervised_queries"].detach()
                ),
                "teacher_map_difference": float(
                    outputs["sqer_teacher_map_difference"].detach()
                ),
            })
            del samples, targets, outputs, loss_dict, total

        scaler.unscale_(optimizer)
        gradients = {
            "sqer_output": gradient_bucket_norm(model, ("sqer.quality_head.5.",)),
            "sqer_local_encoder": gradient_bucket_norm(model, ("sqer.s4_projection.",)),
            "sqer_cross_attention": gradient_bucket_norm(model, ("sqer.query_blocks.",)),
            "rgb_s4_backbone": gradient_bucket_norm(model, ("backbone.stages.0.",)),
            "m_conditioner": gradient_bucket_norm(model, ("sd2_conditioner.",)),
            "detector_decoder": gradient_bucket_norm(model, ("decoder.",)),
        }
        max_norm = float(sam_cfg.yaml_cfg.get("clip_max_norm", 0.1))
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)
        if warmup is not None:
            warmup.step()
        torch.cuda.synchronize()
        update_rows.append({
            "optimizer_step": update_index + 1,
            "micro_batches": micro_losses,
            "gradients_before_clip": gradients,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        })
        print(json.dumps(update_rows[-1], ensure_ascii=False), flush=True)
    paired_config["first_batch_gradient_comparison"] = first_gradient_ratio
    paired_config["optimizer_updates"] = update_rows
    required_nonzero = (
        "sqer_output",
        "sqer_local_encoder",
        "sqer_cross_attention",
        "rgb_s4_backbone",
        "m_conditioner",
        "detector_decoder",
    )
    for key in required_nonzero:
        if update_rows[-1]["gradients_before_clip"][key]["norm"] <= 0:
            raise RuntimeError(f"required gradient bucket is zero after two steps: {key}")
    if update_rows[0]["gradients_before_clip"]["sqer_output"]["norm"] <= 0:
        raise RuntimeError("zero-initialized S-QER final head got no first-step gradient")
    paired_config["peak_allocated_gib"] = max(
        row["peak_allocated_gib"] for row in update_rows
    )
    paired_config["peak_reserved_gib"] = max(
        row["peak_reserved_gib"] for row in update_rows
    )
    paired_config["status"] = "PASS"
    paired_config["official_training_started"] = False
    (args.output_dir / "preflight.json").write_text(
        json.dumps(paired_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "preflight": str(args.output_dir / 'preflight.json')}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
