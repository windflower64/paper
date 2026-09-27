"""Two scratch optimizer updates per recovery arm on real RGB-T batches."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "D-FINE"
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig
from src.misc import dist_utils

INIT = ROOT / "outputs/M_LAYOUT_RECOVERY_20E_TESTDEV/common_init.pth"
OUT = ROOT / "reports/M_LAYOUT_RECOVERY_20E_TESTDEV/training_path_preflight.json"


def digest_tensors(state):
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        if key.startswith(("thermal_backbone.", "thermal_encoder.")):
            result.update(key.encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def run_arm(arm):
    dist_utils.setup_seed(0)
    config = REPO / f"experiments/phase_m/m_layout_recovery_{arm.lower()}_warmup.yml"
    cfg = YAMLConfig(str(config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model.to("cuda")
    state = torch.load(INIT, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state, strict=False)
    criterion = cfg.criterion.to("cuda")
    optimizer = cfg.optimizer
    if hasattr(model, "set_training_epoch"):
        model.set_training_epoch(0)
    if hasattr(criterion, "training_epoch"):
        criterion.training_epoch = 0
    model.train()
    criterion.train()
    thermal_before = digest_tensors(model.state_dict())
    dist_utils.setup_seed(int(cfg.yaml_cfg["recovery_data_seed"]))
    loader = cfg.train_dataloader
    loader.set_epoch(0)
    stream = iter(loader)
    records = []
    signature = []
    for step in range(2):
        samples, targets = next(stream)
        signature.append({"image_ids": [int(t["image_id"]) for t in targets],
                          "sample_sha256": hashlib.sha256(samples.contiguous().numpy().tobytes()).hexdigest()})
        if samples.shape != (8, 6, 512, 640):
            raise RuntimeError(f"Unexpected {arm} training batch shape: {tuple(samples.shape)}")
        for target in targets:
            if "infrared_boxes" not in target or "masks" not in target:
                raise RuntimeError("RGB, corrected IR and SAM teacher fields must all be present")
            boxes = target["infrared_boxes"]
            if len(boxes) and (boxes.abs().max() > 1.0 or boxes.min() < 0.0 or boxes.shape[-1] != 4):
                raise RuntimeError("Training IR boxes must be normalized 4-coordinate tensors")
        samples = samples.cuda()
        targets = [{k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                   for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=True):
            outputs = model(samples, targets=targets)
        with torch.autocast(device_type="cuda", enabled=False):
            losses = criterion(outputs, targets, epoch=0, step=step,
                               global_step=step, epoch_step=len(loader))
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {arm} detector loss")
        loss.backward()
        prefixes = ("sd2_conditioner.",) if arm == "R1" else (
            "mote_fusion.objectness_head.", "mote_fusion.object_token_s8.",
            "mote_fusion.object_token_s16.", "mote_fusion.visible_query.",
            "mote_fusion.evidence_query.", "mote_fusion.evidence_key.",
            "mote_fusion.residual_generator.", "mote_fusion.reliability_gate.",
        )
        gradients = {}
        for prefix in prefixes:
            values = [parameter.grad.detach().float().abs().max().item()
                      for name, parameter in model.named_parameters()
                      if name.startswith(prefix) and parameter.grad is not None]
            gradients[prefix] = max(values, default=0.0)
        if not any(value > 0 for value in gradients.values()):
            raise RuntimeError(f"No M gradients in {arm} step {step}")
        if arm == "R2" and step == 1 and any(value <= 0 for value in gradients.values()):
            raise RuntimeError(f"R2 branch does not receive complete gradients: {gradients}")
        optimizer.step()
        records.append({"step": step, "total_loss": float(loss.detach()),
                        "ir_box_counts": [len(t["infrared_boxes"]) for t in targets],
                        "sam_teacher_valid": sum(float(t["sam_quality"]) > 0 for t in targets),
                        "m_grad_max_by_prefix": gradients,
                        "finite_prediction": bool(torch.isfinite(outputs["pred_boxes"]).all())})
    thermal_after = digest_tensors(model.state_dict())
    if thermal_before != thermal_after:
        raise RuntimeError(f"Frozen IR state changed in {arm}")
    return {"config": str(config), "image_signatures": signature,
            "scratch_steps": records, "frozen_ir_unchanged": True,
            "optimizer_updates": 2, "checkpoint_written": False}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if not INIT.is_file():
        raise FileNotFoundError(INIT)
    torch.multiprocessing.set_sharing_strategy("file_system")
    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)
    try:
        arms = {}
        for arm in ("R1", "R2"):
            arms[arm] = run_arm(arm)
            torch.cuda.empty_cache()
        if arms["R1"]["image_signatures"] != arms["R2"]["image_signatures"]:
            raise RuntimeError("Paired data/augmentation order differs between R1 and R2")
        result = {"schema": "m_layout_recovery_training_path_v1", "status": "PASS",
                  "real_data_signature_matched": True, "arms": arms,
                  "note": "Scratch optimizer steps only; no production training state saved."}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        dist_utils.cleanup()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
