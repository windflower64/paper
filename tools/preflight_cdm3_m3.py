"""M3目标候选无坐标校准的严格CUDA预检。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def move_targets(targets, device):
    return [
        {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = YAMLConfig(str(args.config.resolve()))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["ema"]["module"], strict=True)

    decoder = model.decoder
    if decoder.sdtec_coupler_mode != "candidate_logit_calibration":
        raise RuntimeError("configured model is not M3 candidate calibration")
    tokenizer = decoder.sdtec_candidate_tokenizer
    calibrator = decoder.sdtec_logit_calibrator
    if tokenizer is None or calibrator is None:
        raise RuntimeError("M3 tokenizer or calibrator was not constructed")
    if tokenizer.num_tokens != 8 or calibrator.protected_tail_queries != 50:
        raise RuntimeError("M3 candidate count or HRQS protection is incorrect")

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    invalid_trainable = [
        name for name in trainable_names if not name.startswith("decoder.sdtec_")
    ]
    frozen_target_head = [
        name
        for name, parameter in model.named_parameters()
        if name.startswith("decoder.sdtec_candidate_tokenizer.target_")
        and not parameter.requires_grad
    ]
    all_target_head = [
        name
        for name, _ in model.named_parameters()
        if name.startswith("decoder.sdtec_candidate_tokenizer.target_")
    ]
    if invalid_trainable:
        raise RuntimeError(f"non-M3 trainable parameters: {invalid_trainable[:10]}")
    if len(frozen_target_head) != len(all_target_head) or not all_target_head:
        raise RuntimeError("the imported thermal candidate head is not fully frozen")

    frozen_before = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key not in trainable_names
    }
    device = torch.device("cuda")
    model.to(device)
    criterion = cfg.criterion.to(device)
    optimizer = cfg.optimizer
    # A short four-step preflight cannot wait for the default 65536 scale to
    # back off after overflow.  A conservative finite scale still exercises
    # the same AMP unscale/step path used by formal training.
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=128.0)
    loader = cfg.train_dataloader
    samples, targets = next(iter(loader))
    samples = samples.to(device)
    targets = move_targets(targets, device)
    if samples.shape[0] != 32 or samples.shape[1] != 6:
        raise RuntimeError(f"unexpected M3 input shape {tuple(samples.shape)}")

    captured = {}

    def capture_candidates(_module, _inputs, output):
        captured["scores"] = output["candidate_scores"].detach().cpu()
        captured["tokens"] = output["tokens"].detach().cpu()

    hook = tokenizer.register_forward_hook(capture_candidates)
    model.eval()
    decoder.set_training_epoch(20)
    model.rgbt_thermal_intervention = "normal"
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        initial_normal = model(samples)
    initial_scores = captured["scores"].clone()
    initial_tokens = captured["tokens"].clone()
    model.rgbt_thermal_intervention = "zero"
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        initial_zero = model(samples)
    if not torch.equal(initial_normal["pred_logits"], initial_zero["pred_logits"]):
        raise RuntimeError("zero-initialized M3 changed logits")
    if not torch.equal(initial_normal["pred_boxes"], initial_zero["pred_boxes"]):
        raise RuntimeError("zero-initialized M3 changed boxes")
    if not torch.isfinite(initial_scores).all():
        raise RuntimeError("M3 candidate scores contain non-finite values")
    if not torch.all(initial_scores[:, :-1] >= initial_scores[:, 1:]):
        raise RuntimeError("M3 candidate scores are not rank sorted")

    model.rgbt_thermal_intervention = "feature_permute"
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        initial_permuted = model(samples)
    permuted_scores = captured["scores"].clone()
    permuted_tokens = captured["tokens"].clone()
    score_permutation_error = float(
        (initial_scores - permuted_scores).abs().max()
    )
    token_permutation_error = float(
        (initial_tokens - permuted_tokens).abs().max()
    )
    if score_permutation_error > 1e-6 or token_permutation_error > 1e-6:
        raise RuntimeError(
            "M3 exported coordinate-free evidence changed after spatial permutation: "
            f"scores={score_permutation_error}, tokens={token_permutation_error}"
        )
    if not torch.equal(initial_normal["pred_boxes"], initial_permuted["pred_boxes"]):
        raise RuntimeError("M3 feature permutation changed initial boxes")

    records = []
    torch.cuda.reset_peak_memory_stats(device)
    model.rgbt_thermal_intervention = "normal"
    for step in range(4):
        decoder.set_training_epoch(1)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(samples, targets=targets)
        with torch.autocast(device_type="cuda", enabled=False):
            losses = criterion(
                outputs,
                targets,
                epoch=1,
                step=step,
                global_step=step,
                epoch_step=len(loader),
            )
            total_loss = sum(losses.values())
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite M3 loss at step {step}")
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        delta = outputs["sdtec_logit_delta"].detach()
        if torch.count_nonzero(delta[:, -50:]):
            raise RuntimeError("M3 changed a protected HRQS classification logit")
        if "sdtec_mismatch_logits" not in outputs:
            raise RuntimeError("M3 training omitted paired mismatch output")
        target_head_gradients = [
            name
            for name, parameter in model.named_parameters()
            if name.startswith("decoder.sdtec_candidate_tokenizer.target_")
            and parameter.grad is not None
        ]
        if target_head_gradients:
            raise RuntimeError(
                f"frozen thermal candidate head received gradients: {target_head_gradients}"
            )
        nonzero_gradient = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad.detach()))
        ]
        scaler.step(optimizer)
        scaler.update()
        records.append(
            {
                "step": step,
                "loss": float(total_loss.detach().cpu()),
                "loss_sdtec_pair_rank": float(
                    losses["loss_sdtec_pair_rank"].detach().cpu()
                ),
                "ordinary_mean_abs_logit_delta": float(
                    delta[:, :-50].abs().mean().cpu()
                ),
                "candidate_score_mean": float(
                    outputs["sdtec_candidate_scores"].detach().mean().cpu()
                ),
                "trainable_with_nonzero_gradient": len(nonzero_gradient),
                "nonzero_gradient_names": nonzero_gradient,
            }
        )

    model.eval()
    decoder.set_training_epoch(20)
    model.rgbt_thermal_intervention = "normal"
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        opened_normal = model(samples)
    model.rgbt_thermal_intervention = "zero"
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        opened_zero = model(samples)
    if not torch.equal(opened_normal["pred_boxes"], opened_zero["pred_boxes"]):
        raise RuntimeError("opened M3 changed box geometry")
    if torch.equal(opened_normal["pred_logits"], opened_zero["pred_logits"]):
        raise RuntimeError("opened M3 did not causally use thermal candidates")
    hook.remove()

    model.cpu()
    frozen_changes = []
    for key, value in model.state_dict().items():
        if key in frozen_before and not torch.equal(
            value.detach().cpu(), frozen_before[key]
        ):
            frozen_changes.append(key)
    if frozen_changes:
        raise RuntimeError(f"M3 training changed frozen tensors: {frozen_changes[:10]}")

    result = {
        "status": "PASS",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": args.seed,
        "batch_size": int(loader.batch_size),
        "input_shape": list(samples.shape),
        "trainable_tensor_count": len(trainable_names),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "frozen_candidate_head_tensors": len(frozen_target_head),
        "candidate_tokens": tokenizer.num_tokens,
        "protected_hrqs_queries": 50,
        "max_logit_delta": decoder.sdtec_max_logit_delta,
        "initial_normal_equals_zero_logits": True,
        "initial_normal_equals_zero_boxes": True,
        "opened_normal_equals_zero_boxes": True,
        "opened_normal_differs_from_zero_logits": True,
        "feature_permutation_max_candidate_score_error": score_permutation_error,
        "feature_permutation_max_candidate_token_error": token_permutation_error,
        "initial_candidate_score_mean": float(initial_scores.mean()),
        "initial_candidate_quality_mean": float(initial_scores.sigmoid().mean()),
        "steps": records,
        "frozen_tensor_changes": frozen_changes,
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
