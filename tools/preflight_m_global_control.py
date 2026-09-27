"""Check global-statistics dual-modal control against the matched recovery run."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "D-FINE"
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig
from src.misc import dist_utils

INIT = ROOT / "outputs/M_LAYOUT_RECOVERY_20E_TESTDEV/common_init.pth"
SOURCE = ROOT / "reports/M_LAYOUT_RECOVERY_20E_TESTDEV/training_path_preflight.json"
OUT = ROOT / "reports/M_GLOBAL_CONTROL_20E_TESTDEV/preflight.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.multiprocessing.set_sharing_strategy("file_system")
    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)
    cfg = YAMLConfig(str(REPO / "experiments/phase_m/m_global_control_warmup.yml"))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model.cuda()
    state = torch.load(INIT, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state, strict=False)
    branch = model.sd2_conditioner
    if type(branch).__name__ != "SpatiallyDecoupledThermalConditioner":
        raise RuntimeError("Expected global-statistics control, not token M")
    criterion = cfg.criterion.cuda()
    optimizer = cfg.optimizer
    model.train()
    model.set_training_epoch(0)
    criterion.training_epoch = 0
    dist_utils.setup_seed(int(cfg.yaml_cfg["recovery_data_seed"]))
    loader = cfg.train_dataloader
    loader.set_epoch(0)
    iterator = iter(loader)
    signatures = []
    steps = []
    for step in range(2):
        samples, targets = next(iterator)
        signatures.append({"image_ids": [int(t["image_id"]) for t in targets],
                           "sample_sha256": hashlib.sha256(samples.contiguous().numpy().tobytes()).hexdigest()})
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
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        grads = [float(p.grad.detach().abs().max()) for p in branch.parameters()
                 if p.grad is not None]
        if not grads or max(grads) <= 0:
            raise RuntimeError(f"No global fusion gradients at step {step}")
        optimizer.step()
        steps.append({"step": step, "loss": float(loss.detach()),
                      "fusion_grad_max": max(grads),
                      "finite_boxes": bool(torch.isfinite(outputs["pred_boxes"]).all())})
    expected = json.loads(SOURCE.read_text(encoding="utf-8"))["arms"]["R1"]["image_signatures"]
    if signatures != expected:
        raise RuntimeError("Training image/augmentation sequence differs from R1/R2")
    result = {"schema": "m_global_control_preflight_v1", "status": "PASS",
              "common_init": str(INIT), "first_two_batches_match_recovery": True,
              "fusion_class": type(branch).__name__, "steps": steps,
              "note": "Scratch updates only, no checkpoint saved."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
