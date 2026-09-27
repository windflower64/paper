"""Preflight the scheduled SGC2-SAM experiment before a full run."""
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO))

from src.core import YAMLConfig
from src.solver import BaseSolver
from tools.preflight_q_rank1 import move_targets


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "sam"
    assert arm in ("sam", "box")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    config = REPO / f"experiments/phase_s/s_sgc2_{arm}_decay9_14_b8a4_20e.yml"
    cfg = YAMLConfig(str(config))
    model = cfg.model
    shim = BaseSolver.__new__(BaseSolver)
    shim.model = model
    checkpoint = ROOT / "weights/m_sd2_joint_coco_thermal_identity_init.pth"
    shim.load_tuning_state(str(checkpoint))
    initial_digest = model_digest(model)

    assert cfg.val_dataloader.dataset.sam_mask_root is None
    assert model._sgc_supervision_scale() == 1.0
    schedule = {}
    for epoch in range(20):
        model.set_training_epoch(epoch)
        schedule[str(epoch)] = model._sgc_supervision_scale()
    assert schedule == {
        str(epoch): value
        for epoch, value in enumerate(
            [1.0] * 10 + [0.8, 0.6, 0.4, 0.2] + [0.0] * 6
        )
    }

    model = model.cuda()
    criterion = cfg.criterion.cuda()
    optimizer = cfg.optimizer
    samples, targets = next(iter(cfg.train_dataloader))
    assert tuple(samples.shape) == (8, 3, 512, 640)
    samples = samples.cuda()
    targets = move_targets(targets, "cuda")

    model.eval()
    with torch.no_grad():
        scheduled = model(samples)
        model.sgc_enabled = False
        model.backbone.return_idx = [2, 3]
        control = model(samples)
        for key in ("pred_logits", "pred_boxes"):
            torch.testing.assert_close(scheduled[key], control[key], rtol=0, atol=0)
    model.sgc_enabled = True
    model.backbone.return_idx = [1, 2, 3]
    del scheduled, control

    model.train()
    model.set_training_epoch(12)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        output = model(samples, targets)
    assert "sgc_group_loss" in output and output["sgc_group_loss"].item() > 0
    weighted_epoch12 = float(output["sgc_group_loss"])
    model.set_training_epoch(14)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        output = model(samples, targets)
    assert "sgc_group_loss" not in output
    del output

    model.set_training_epoch(0)
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler(init_scale=128)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    parameter = model.backbone.stages[1].blocks[0].layers[0].conv.weight
    before = parameter.detach().clone()
    for step, (batch, batch_targets) in enumerate(cfg.train_dataloader):
        if step == 4:
            break
        batch = batch.cuda()
        batch_targets = move_targets(batch_targets, "cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(batch, batch_targets)
        losses = criterion(
            output,
            batch_targets,
            epoch=0,
            step=step,
            global_step=step,
            epoch_step=len(cfg.train_dataloader),
        )
        assert all(torch.isfinite(value) for value in losses.values())
        assert losses["loss_sgc_group"].item() > 0
        rows.append(float(losses["loss_sgc_group"].detach()))
        scaler.scale(sum(losses.values()) / 4).backward()
    scaler.unscale_(optimizer)
    assert all(
        torch.isfinite(item.grad).all()
        for item in model.parameters()
        if item.grad is not None
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    scaler.step(optimizer)
    scaler.update()
    parameter_update = (parameter - before).abs().max().item()
    assert parameter_update > 0

    result = {
        "status": "PASS",
        "arm": arm,
        "config": str(config),
        "initial_model_sha256": initial_digest,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "inference_exact_C_equivalence": True,
        "physical_batch": 8,
        "accumulation": 4,
        "schedule": schedule,
        "weighted_aux_epoch12": weighted_epoch12,
        "weighted_aux_epoch0_steps": rows,
        "parameter_update": parameter_update,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    report = ROOT / "reports/109_sgc2_teacher_retirement"
    report.mkdir(parents=True, exist_ok=True)
    destination = report / ("preflight.json" if arm == "sam" else "preflight_box.json")
    destination.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
