"""Best-EMA normal/M-bypassed testdev comparison for the R1/R2 recovery run."""

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

REPORT = ROOT / "reports/M_LAYOUT_RECOVERY_20E_TESTDEV"
RUN = ROOT / "outputs/M_LAYOUT_RECOVERY_20E_TESTDEV"
ARMS = {"R1": ("r1_main", RUN / "R1_SD22/seed0"),
        "R2": ("r2_main", RUN / "R2_OTE2_LAYOUT/seed0"),
        "GLOBAL": ("global", ROOT / "outputs/M_GLOBAL_CONTROL_20E_TESTDEV/seed0"),
        "R2_NO_S": ("r2_no_s", ROOT / "outputs/M_R2_SAM_FACTORIAL_20E_TESTDEV/no_s_seed0"),
        "R2_SGC_ONLY": ("r2_sgc_only", ROOT / "outputs/M_R2_SAM_FACTORIAL_20E_TESTDEV/sgc_only_seed0")}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ARMS:
        raise SystemExit("usage: evaluate_m_layout_recovery.py R1|R2|GLOBAL|R2_NO_S|R2_SGC_ONLY")
    torch.multiprocessing.set_sharing_strategy("file_system")
    arm = sys.argv[1]
    config_name, run = ARMS[arm]
    report = (ROOT / "reports/M_GLOBAL_CONTROL_20E_TESTDEV" if arm == "GLOBAL"
              else ROOT / "reports/M_R2_SAM_FACTORIAL_20E_TESTDEV" if arm in ("R2_NO_S", "R2_SGC_ONLY")
              else REPORT)
    output = report / f"{arm.lower()}_best_ema_m_ablation.json"
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(line) for line in (run / "log.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 18:
        raise RuntimeError(f"Expected 18 main-run log rows, got {len(rows)}")
    best = max(rows, key=lambda row: row["test_coco_eval_bbox"][0])
    checkpoint = run / "best_stg1.pth"
    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)
    config_path = (REPO / "experiments/phase_m/m_global_control_main.yml" if arm == "GLOBAL"
                   else REPO / "experiments/phase_m/m_r2_no_s_main.yml" if arm == "R2_NO_S"
                   else REPO / "experiments/phase_m/m_r2_sgc_only_main.yml" if arm == "R2_SGC_ONLY"
                   else REPO / f"experiments/phase_m/m_layout_recovery_{config_name}.yml")
    cfg = YAMLConfig(str(config_path),
                     resume=str(checkpoint), tuning=None, use_amp=True,
                     output_dir=str(report / f"{arm.lower()}_eval_runtime"))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    model = solver.ema.module if solver.ema else solver.model
    normal, _ = evaluate(model, solver.criterion, solver.postprocessor,
                         solver.val_dataloader, solver.evaluator, solver.device,
                         epoch=-1, use_wandb=False)
    normal_metrics = normal["coco_eval_bbox"]
    error = max(abs(float(x) - float(y)) for x, y in zip(normal_metrics,
                                                         best["test_coco_eval_bbox"]))
    if error > 0.0002:
        raise RuntimeError(f"Best EMA re-evaluation mismatch: {error}")
    if arm in ("R1", "GLOBAL"):
        handle = model.sd2_conditioner.register_forward_hook(
            lambda module, inputs, result: inputs[0])
    else:
        handle = model.mote_fusion.register_forward_hook(
            lambda module, inputs, result: (inputs[0], result[1]))
    try:
        off, _ = evaluate(model, solver.criterion, solver.postprocessor,
                          solver.val_dataloader, solver.evaluator, solver.device,
                          epoch=-1, use_wandb=False)
    finally:
        handle.remove()
    off_metrics = off["coco_eval_bbox"]
    result = {"schema": "m_layout_recovery_best_ema_ablation_v1", "status": "PASS",
              "arm": arm, "best_epoch_zero_based": best["epoch"], "images": 1820,
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
              "normal_metrics": normal_metrics, "m_bypassed_metrics": off_metrics,
              "normal_minus_m_bypassed_ap_pp": 100.0 * (normal_metrics[0] - off_metrics[0]),
              "normal_reproduction_max_abs_error": error,
              "note": "Same best EMA; M output bypassed by forward hook; no retraining."}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
