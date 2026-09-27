"""Two real-batch scratch updates for the R2-without-SAM training control."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig
from src.misc import dist_utils

CONTROL = REPO / "experiments/phase_m/m_r2_no_s_warmup.yml"
REFERENCE = REPO / "experiments/phase_m/m_layout_recovery_r2_warmup.yml"
INIT = ROOT / "outputs/M_LAYOUT_RECOVERY_20E_TESTDEV/common_init.pth"
REFERENCE_AUDIT = ROOT / "reports/M_LAYOUT_RECOVERY_20E_TESTDEV/training_path_preflight.json"
OUT = ROOT / "reports/M_R2_SAM_FACTORIAL_20E_TESTDEV/no_s_preflight.json"


def sample_signature(samples, targets):
    return {"image_ids": [int(target["image_id"]) for target in targets],
            "sample_sha256": hashlib.sha256(samples.contiguous().numpy().tobytes()).hexdigest()}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.multiprocessing.set_sharing_strategy("file_system")
    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)
    try:
        base = YAMLConfig(str(REFERENCE))
        cfg = YAMLConfig(str(CONTROL))
        excluded = {"__include__", "output_dir", "DFINE", "DFINECriterion"}
        for key in set(base.yaml_cfg) | set(cfg.yaml_cfg):
            if key not in excluded and base.yaml_cfg.get(key) != cfg.yaml_cfg.get(key):
                raise RuntimeError(f"Unexpected configuration difference: {key}")
        for key in set(base.yaml_cfg["DFINE"]) | set(cfg.yaml_cfg["DFINE"]):
            if key not in {"sgc_enabled", "sgc_aux_weight"} and base.yaml_cfg["DFINE"].get(key) != cfg.yaml_cfg["DFINE"].get(key):
                raise RuntimeError(f"Unexpected DFINE difference: {key}")
        for key in set(base.yaml_cfg["DFINECriterion"]) | set(cfg.yaml_cfg["DFINECriterion"]):
            if base.yaml_cfg["DFINECriterion"].get(key) != cfg.yaml_cfg["DFINECriterion"].get(key):
                raise RuntimeError(f"Unexpected criterion difference: {key}")
        if cfg.yaml_cfg["DFINE"]["sgc_enabled"] or cfg.yaml_cfg["DFINE"]["sqer_bypass"] is not True:
            raise RuntimeError("SAM guidance is not fully disabled")
        if cfg.yaml_cfg["DFINECriterion"]["sqer_shape_weight"] != 0:
            raise RuntimeError("SQER shape loss is not zero")

        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        model = cfg.model.cuda().train()
        state = torch.load(INIT, map_location="cpu", weights_only=False)["model"]
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not key.startswith("mote_fusion.") for key in result.missing_keys):
            raise RuntimeError(f"Unexpected state loading result: {result}")
        criterion = cfg.criterion.cuda().train()
        optimizer = cfg.optimizer
        model.set_training_epoch(0)
        criterion.training_epoch = 0
        dist_utils.setup_seed(int(cfg.yaml_cfg["recovery_data_seed"]))
        loader = cfg.train_dataloader
        loader.set_epoch(0)
        stream = iter(loader)
        expected = json.loads(REFERENCE_AUDIT.read_text(encoding="utf-8"))["arms"]["R2"]["image_signatures"]
        records = []
        for step in range(2):
            samples, targets = next(stream)
            signature = sample_signature(samples, targets)
            if signature != expected[step]:
                raise RuntimeError(f"Training samples differ at step {step}: {signature}")
            if samples.shape != (8, 6, 512, 640):
                raise RuntimeError(f"Unexpected sample shape {tuple(samples.shape)}")
            samples = samples.cuda()
            targets = [{k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=True):
                outputs = model(samples, targets=targets)
            with torch.autocast(device_type="cuda", enabled=False):
                losses = criterion(outputs, targets, epoch=0, step=step,
                                   global_step=step, epoch_step=len(loader))
                loss = sum(losses.values())
            if not torch.isfinite(loss) or not torch.isfinite(outputs["pred_boxes"]).all():
                raise RuntimeError("Non-finite detector loss or prediction")
            if any("sgc" in key or "sqer_shape" in key for key in losses):
                raise RuntimeError(f"SAM loss active: {list(losses)}")
            loss.backward()
            gradients = [float(param.grad.detach().abs().max()) for name, param in model.named_parameters()
                         if name.startswith("mote_fusion.") and param.grad is not None]
            if not gradients or max(gradients) <= 0:
                raise RuntimeError("M branch has no detection gradient")
            optimizer.step()
            records.append({"step": step, "signature": signature, "loss": float(loss.detach()),
                            "m_gradient_max": max(gradients)})
        report = {"schema": "m_r2_no_s_preflight_v1", "status": "PASS",
                  "common_init": str(INIT), "reference": str(REFERENCE),
                  "control": str(CONTROL), "scratch_optimizer_updates": 2,
                  "sam_losses_disabled": True, "real_batch_signatures_match_r2": True,
                  "records": records, "production_checkpoint_written": False}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        dist_utils.cleanup()


if __name__ == "__main__":
    main()
