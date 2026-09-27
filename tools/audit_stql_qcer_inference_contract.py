"""Verify the trained QCER inference contract on the real validation loader."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def maximum_difference(left, right) -> float:
    return float((left - right).abs().max())

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig
    cfg = YAMLConfig(str(args.config.resolve()))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model.cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    loader = cfg.val_dataloader
    if getattr(loader.dataset, "sam_mask_root", None) is not None:
        raise RuntimeError("validation loader unexpectedly depends on a SAM cache")
    samples, _ = next(iter(loader))
    samples = samples.cuda()
    unavailable = torch.zeros(samples.shape[0], device=samples.device, dtype=torch.bool)
    with torch.no_grad():
        model.rgbt_thermal_intervention = "normal"
        model.qcer_bypass = False
        normal = model(samples)
        model.rgbt_thermal_intervention = "batch_shuffle"
        shuffled = model(samples)
        model.rgbt_thermal_intervention = "normal"
        model.qcer_bypass = True
        bypass = model(samples)
        model.qcer_bypass = False
        missing = model(samples, qcer_availability=unavailable)
    result = {
        "schema": "stql_qcer_inference_contract_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "validation_sam_cache": None,
        "batch_shape": list(samples.shape),
        "normal_box_vs_shuffled_max_abs": maximum_difference(normal["pred_boxes"], shuffled["pred_boxes"]),
        "normal_base_logit_vs_shuffled_max_abs": maximum_difference(normal["base_pred_logits"], shuffled["base_pred_logits"]),
        "normal_final_logit_vs_shuffled_max_abs": maximum_difference(normal["pred_logits"], shuffled["pred_logits"]),
        "normal_delta_vs_shuffled_max_abs": maximum_difference(normal["qcer_delta_logits"], shuffled["qcer_delta_logits"]),
        "bypass_final_vs_base_max_abs": maximum_difference(bypass["pred_logits"], bypass["base_pred_logits"]),
        "bypass_delta_max_abs": float(bypass["qcer_delta_logits"].abs().max()),
        "missing_final_vs_base_max_abs": maximum_difference(missing["pred_logits"], missing["base_pred_logits"]),
        "missing_delta_max_abs": float(missing["qcer_delta_logits"].abs().max()),
        "normal_delta_mean_abs": float(normal["qcer_delta_logits"].abs().mean()),
    }
    exact_zero_fields = (
        "normal_box_vs_shuffled_max_abs",
        "normal_base_logit_vs_shuffled_max_abs",
        "bypass_final_vs_base_max_abs",
        "bypass_delta_max_abs",
        "missing_final_vs_base_max_abs",
        "missing_delta_max_abs",
    )
    if any(result[name] != 0.0 for name in exact_zero_fields):
        raise RuntimeError(f"inference isolation failed: {result}")
    if result["normal_final_logit_vs_shuffled_max_abs"] <= 0.0:
        raise RuntimeError("trained QCER is insensitive to changed thermal content")
    result["status"] = "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()