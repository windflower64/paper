"""Audit the corrected layout, shared start and one real training batch."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "D-FINE"
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig
from src.misc import dist_utils
from src.zoo.dfine.target_evidence_thermal_fusion import TargetEvidenceThermalFusion

OUT = ROOT / "reports/M_LAYOUT_RECOVERY_20E_TESTDEV/preflight.json"
INIT = ROOT / "outputs/M_LAYOUT_RECOVERY_20E_TESTDEV/common_init.pth"
CONFIGS = {
    "R1": REPO / "experiments/phase_m/m_layout_recovery_r1_warmup.yml",
    "R2": REPO / "experiments/phase_m/m_layout_recovery_r2_warmup.yml",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def legacy_layout_contract():
    source = subprocess.check_output(
        ["git", "show", "e875e5e3c9927e49393256798a2d9bac2e2a06c1:src/zoo/dfine/target_evidence_thermal_fusion.py"],
        cwd=REPO, text=True,
    )
    namespace = {"__name__": "historical_target_evidence_thermal_fusion"}
    exec(compile(source, "<historical_e2_module>", "exec"), namespace)
    historical = namespace["TargetEvidenceThermalFusion"](8, 12, 8, embed_dim=8).eval()
    current = TargetEvidenceThermalFusion(8, 12, 8, embed_dim=8, layout_fix=False).eval()
    corrected = TargetEvidenceThermalFusion(8, 12, 8, embed_dim=8, layout_fix=True).eval()
    with torch.no_grad():
        historical.residual_generator[-1].weight.normal_(0, 0.1)
        historical.residual_generator[-1].bias.normal_(0, 0.1)
    current.load_state_dict(historical.state_dict(), strict=True)
    corrected.load_state_dict(historical.state_dict(), strict=True)
    rgb = torch.randn(2, 8, 4, 5)
    ir8, ir16 = torch.randn(2, 12, 8, 10), torch.randn(2, 8, 4, 5)
    with torch.inference_mode():
        historical_out = historical(rgb, ir8, ir16)[0]
        legacy_out = current(rgb, ir8, ir16)[0]
        aligned_out = corrected(rgb, ir8, ir16)[0]
        shifted = rgb.roll(1, dims=2)
        corrected_shifted = corrected(shifted, ir8, ir16)[0]
        legacy_shifted = current(shifted, ir8, ir16)[0]
    historical_max_error = float((historical_out - legacy_out).abs().max())
    aligned_shift_error = float(((corrected_shifted - shifted) - (aligned_out - rgb).roll(1, dims=2)).norm()
                                / (aligned_out - rgb).norm().clamp_min(1e-12))
    legacy_shift_error = float(((legacy_shifted - shifted) - (legacy_out - rgb).roll(1, dims=2)).norm()
                               / (legacy_out - rgb).norm().clamp_min(1e-12))
    if historical_max_error != 0 or aligned_shift_error > 1e-4 or legacy_shift_error < 0.1:
        raise RuntimeError("Layout compatibility or corrected spatial contract failed")
    return {"historical_model_max_error": historical_max_error,
            "corrected_rgb_shift_relative_error": aligned_shift_error,
            "historical_rgb_shift_relative_error": legacy_shift_error,
            "nonzero_generator_tested": True}


def build_model(arm, common):
    cfg = YAMLConfig(str(CONFIGS[arm]))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    dist_utils.setup_seed(0)
    model = cfg.model
    expected = model.state_dict()
    missing = sorted(set(expected) - set(common))
    expected_missing = "sd2_conditioner." if arm == "R1" else "mote_fusion."
    if any(not key.startswith(expected_missing) for key in missing):
        raise RuntimeError(f"Unexpected missing initialization keys in {arm}: {missing[:12]}")
    incompatible = [key for key, value in common.items() if key not in expected or expected[key].shape != value.shape]
    if incompatible:
        raise RuntimeError(f"Incompatible common init for {arm}: {incompatible[:12]}")
    model.load_state_dict(common, strict=False)
    if arm == "R2" and not model.mote_fusion.layout_fix:
        raise RuntimeError("R2 is not using corrected layout")
    if cfg.yaml_cfg["train_dataloader"]["total_batch_size"] != 8 or cfg.yaml_cfg["gradient_accumulation_steps"] != 4:
        raise RuntimeError("Wrong physical or effective training batch")
    optimizer = cfg.optimizer
    ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if ids != trainable or len(ids) != sum(len(group["params"]) for group in optimizer.param_groups):
        raise RuntimeError("Optimizer trainable coverage is not exact")
    return cfg, model, missing


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if not INIT.is_file():
        raise FileNotFoundError(INIT)
    torch.multiprocessing.set_sharing_strategy("file_system")
    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)
    try:
        torch.manual_seed(7)
        layout = legacy_layout_contract()
        common = torch.load(INIT, map_location="cpu", weights_only=False)["model"]
        metrics = {}
        reference = None
        for arm in ("R1", "R2"):
            cfg, model, missing = build_model(arm, common)
            if any(key.startswith("thermal_") and param.requires_grad
                   for key, param in model.named_parameters()):
                raise RuntimeError("Thermal stream is not frozen")
            model.to("cuda").eval()
            loader = cfg.val_dataloader
            batch, targets = next(iter(loader))
            if batch.shape[0] != 8 or batch.shape[1:] != (6, 512, 640):
                raise RuntimeError(f"Unexpected real RGB-T batch shape: {tuple(batch.shape)}")
            model.set_training_epoch(0)
            with torch.inference_mode():
                outputs = model(batch.cuda())
            values = (outputs["pred_logits"].cpu(), outputs["pred_boxes"].cpu())
            if reference is None:
                reference = values
                difference = 0.0
            else:
                difference = max(float((a-b).abs().max()) for a,b in zip(reference,values))
            if difference != 0.0:
                raise RuntimeError(f"R1/R2 start outputs differ: {difference}")
            optimizer_groups = [{"lr": group["lr"], "parameters": sum(p.numel() for p in group["params"])}
                                for group in cfg.optimizer.param_groups]
            metrics[arm] = {"initial_prediction_max_error_from_R1": difference,
                            "method_unique_missing_keys": len(missing),
                            "optimizer_groups": optimizer_groups,
                            "validation_image_count": len(loader.dataset),
                            "batch_shape": list(batch.shape),
                            "rgb_box_counts_first_batch": [len(t["boxes"]) for t in targets],
                            "ir_box_counts_first_batch": [len(t["infrared_boxes"]) for t in targets]}
            del model, cfg
            torch.cuda.empty_cache()
        result = {"schema": "m_layout_recovery_preflight_v1", "status": "PASS",
                  "common_init": str(INIT), "common_init_sha256": sha256(INIT),
                  "layout": layout, "arms": metrics,
                  "model_training_performed": False, "validation_gt_in_forward": False}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        dist_utils.cleanup()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
