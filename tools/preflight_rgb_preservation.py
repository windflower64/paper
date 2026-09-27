"""Real-batch forward/backward checks before P1 training is permitted."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from src.core import YAMLConfig
from src.misc import dist_utils
from src.solver import TASKS
from src.solver.rgb_preservation import RGBPreserver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["c", "cm"], required=True)
    args = parser.parse_args()
    report = REPO.parent / "reports/97_rgb_preservation" / f"preflight_{args.arm}.json"
    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)
    cfg = YAMLConfig(str(REPO / f"experiments/phase_p/p1_{args.arm}_preserve_b8a4_20e.yml"),
                     tuning=str(REPO.parent / "weights/m_sd2_joint_coco_thermal_identity_init.pth"),
                     output_dir=str(report.parent / f"preflight_runtime_{args.arm}"))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.train()
    model = solver.model
    samples, targets = next(iter(solver.train_dataloader))
    samples = samples.to(solver.device)
    targets = [{k: v.to(solver.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
    model.eval()
    with torch.no_grad():
        before = model(samples)
    keys = tuple(model.state_dict())
    preserver = RGBPreserver(model, cfg.yaml_cfg["rgb_preservation"], solver.device)
    with torch.no_grad():
        after = model(samples)
    diffs = {k: float((before[k] - after[k]).abs().max()) for k in ("pred_logits", "pred_boxes")}
    assert max(diffs.values()) == 0 and tuple(model.state_dict()) == keys
    assert not any(p.requires_grad for p in preserver.teacher.parameters())
    teacher_before = {k: v.detach().clone() for k, v in preserver.teacher.state_dict().items()}
    del before, after
    records = []
    model.train()
    for step in range(2):
        solver.optimizer.zero_grad(set_to_none=True)
        # Reuse a fixed real batch for repeatable engineering checks, not training.
        for micro in range(4):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(samples, targets=targets)
            with torch.autocast(device_type="cuda", enabled=False):
                losses = solver.criterion(outputs, targets, epoch=0, step=step, global_step=step, epoch_step=400)
                kd = preserver(samples, targets)
                detection = sum(losses.values())
                loss = detection + kd
            assert torch.isfinite(loss)
            solver.scaler.scale(loss / 4).backward()
        solver.scaler.unscale_(solver.optimizer)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        encoder_grad = sum(float(p.grad.float().square().sum()) for p in model.encoder.parameters() if p.grad is not None) ** .5
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
        solver.scaler.step(solver.optimizer)
        solver.scaler.update()
        records.append({"step": step, "detection_loss": float(detection.detach()),
                        "preservation_loss": float(kd.detach()), "encoder_gradient_norm": encoder_grad})
    assert all(torch.equal(teacher_before[k], v) for k, v in preserver.teacher.state_dict().items())
    assert all(p.grad is None for p in preserver.teacher.parameters())
    preserver.close()
    report.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status="passed", arm=args.arm, physical_batch=len(samples), accumulation=4,
                  inference_max_abs_diff=diffs, model_state_keys_unchanged=True,
                  teacher_parameters_and_buffers_unchanged=True, steps=records,
                  peak_memory_allocated=torch.cuda.max_memory_allocated(), **preserver.describe())
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("PREFLIGHT_PASSED", str(report), flush=True)


if __name__ == "__main__":
    main()
