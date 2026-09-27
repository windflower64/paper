"""Read-only same-EMA inference bypass for the R2 SQER2 branch."""

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
from src.solver import TASKS
from src.solver.det_engine import evaluate

CHECKPOINT = ROOT / "outputs/M_LAYOUT_RECOVERY_20E_TESTDEV/R2_OTE2_LAYOUT/seed0/best_stg1.pth"
CONFIG = REPO / "experiments/phase_m/m_layout_recovery_r2_main.yml"
DESTINATION = ROOT / "reports/M_R2_SAM_FACTORIAL_20E_TESTDEV/sqer_contribution/full_best_ema_sqer_inference_bypass.json"


def main():
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    torch.multiprocessing.set_sharing_strategy("file_system")
    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)
    try:
        cfg = YAMLConfig(str(CONFIG), resume=str(CHECKPOINT), tuning=None, use_amp=True,
                         output_dir=str(DESTINATION.parent / "sqer_bypass_eval_runtime"))
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver.eval()
        model = solver.ema.module if solver.ema else solver.model
        if model.sqer_bypass:
            raise RuntimeError("The full R2 SQER branch is already bypassed")
        normal, _ = evaluate(model, solver.criterion, solver.postprocessor,
                             solver.val_dataloader, solver.evaluator, solver.device,
                             epoch=-1, use_wandb=False)
        model.sqer_bypass = True
        try:
            bypass, _ = evaluate(model, solver.criterion, solver.postprocessor,
                                 solver.val_dataloader, solver.evaluator, solver.device,
                                 epoch=-1, use_wandb=False)
        finally:
            model.sqer_bypass = False
        normal_metrics = normal["coco_eval_bbox"]
        bypass_metrics = bypass["coco_eval_bbox"]
        expected = max((json.loads(line) for line in (CHECKPOINT.parent / "log.txt").read_text(
            encoding="utf-8").splitlines()), key=lambda x: x["test_coco_eval_bbox"][0])["test_coco_eval_bbox"]
        error = max(abs(float(x) - float(y)) for x, y in zip(normal_metrics, expected))
        if error > 2e-4:
            raise RuntimeError(f"Normal best EMA mismatch: {error}")
        result = {"schema": "r2_sqer_same_weight_inference_bypass_v1", "status": "PASS",
                  "checkpoint": str(CHECKPOINT),
                  "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                  "normal_metrics": normal_metrics, "sqer_bypassed_metrics": bypass_metrics,
                  "normal_minus_bypass_pp": [100 * (float(x) - float(y))
                                             for x, y in zip(normal_metrics, bypass_metrics)],
                  "normal_reproduction_max_abs_error": error,
                  "note": "Same trained EMA; SQER2 bypassed at inference only. This measures immediate dependence, not retrained module contribution."}
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        DESTINATION.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        dist_utils.cleanup()


if __name__ == "__main__":
    main()
