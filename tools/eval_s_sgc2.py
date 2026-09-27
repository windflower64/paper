"""Independently re-evaluate the best SGC2 EMA checkpoint on all 1820 images."""
import hashlib
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
REPORT = ROOT / "reports/109_sgc2_teacher_retirement/evaluation"
sys.path.insert(0, str(REPO))

from src.core import YAMLConfig
from src.solver import TASKS
from src.solver.det_engine import evaluate


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "sam"
    assert arm in ("sam", "box")
    run = ROOT / f"outputs/S_SGC2_{arm.upper()}_DECAY9_14_B8A4_20E_TESTDEV/seed0"
    torch.set_num_threads(4)
    for archived in (run / "artifacts/src").rglob("*.py"):
        current = REPO / archived.relative_to(run / "artifacts")
        assert archived.read_bytes() == current.read_bytes(), str(current)

    checkpoint = run / "best_stg1.pth"
    checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = REPO / f"experiments/phase_s/s_sgc2_{arm}_decay9_14_b8a4_20e.yml"
    cfg = YAMLConfig(
        str(config),
        resume=str(checkpoint),
        output_dir=str(REPORT / "runtime"),
    )
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    assert solver.val_dataloader.dataset.sam_mask_root is None
    assert len(solver.val_dataloader.dataset) == 1820
    assert solver.ema is not None

    metrics, _ = evaluate(
        solver.ema.module,
        solver.criterion,
        solver.postprocessor,
        solver.val_dataloader,
        solver.evaluator,
        solver.device,
        epoch=-1,
        use_wandb=False,
    )
    values = metrics["coco_eval_bbox"]
    rows = [
        json.loads(line)
        for line in (run / "log.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    best = max(rows, key=lambda row: row["test_coco_eval_bbox"][0])
    max_error = max(
        abs(actual - expected)
        for actual, expected in zip(values, best["test_coco_eval_bbox"])
    )
    assert max_error < 0.0002
    result = {
        "status": "PASS_CLOSE_REPRODUCTION",
        "arm": arm,
        "checkpoint_sha256": checkpoint_digest,
        "weight_source": "ema",
        "best_epoch": best["epoch"],
        "images": 1820,
        "coco_eval_bbox": values,
        "logged_coco_eval_bbox": best["test_coco_eval_bbox"],
        "max_abs_error": max_error,
        "validation_sam_cache": None,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    destination = REPORT / ("best_ema.json" if arm == "sam" else "best_ema_box.json")
    destination.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("RESULT", result, flush=True)


if __name__ == "__main__":
    main()
