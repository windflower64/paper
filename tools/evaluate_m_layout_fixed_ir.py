"""Replace all per-image M evidence with its training-set mean at fixed EMA weights."""

from __future__ import annotations

import hashlib
import json
import sys
import types
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
        "R2": ("r2_main", RUN / "R2_OTE2_LAYOUT/seed0")}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ARMS:
        raise SystemExit("usage: evaluate_m_layout_fixed_ir.py R1|R2")
    arm = sys.argv[1]
    config_name, run = ARMS[arm]
    result_file = REPORT / f"{arm.lower()}_fixed_train_mean_ir.json"
    if result_file.exists():
        raise FileExistsError(result_file)
    reference = json.loads((REPORT / f"{arm.lower()}_best_ema_m_ablation.json").read_text(encoding="utf-8"))
    torch.multiprocessing.set_sharing_strategy("file_system")
    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)
    checkpoint = run / "best_stg1.pth"
    cfg = YAMLConfig(str(REPO / f"experiments/phase_m/m_layout_recovery_{config_name}.yml"),
                     resume=str(checkpoint), tuning=None, use_amp=True,
                     output_dir=str(REPORT / f"{arm.lower()}_fixed_eval_runtime"))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    model = solver.ema.module if solver.ema else solver.model
    branch = model.sd2_conditioner if arm == "R1" else model.mote_fusion
    method_name = "_thermal_tokens" if arm == "R1" else "_candidate_tokens"
    original = getattr(branch, method_name)

    # Use the validation transform on training images. No training labels or
    # test images enter the prototype. This is evidence, not a learned module.
    train_cfg = YAMLConfig(str(REPO / f"experiments/phase_m/m_layout_recovery_{config_name}.yml"))
    train_cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    ds = train_cfg.yaml_cfg["val_dataloader"]["dataset"]
    ds["img_folder"] = str(ROOT / "data/antiuav6k_common/images/train")
    ds["ann_file"] = str(ROOT / "data/antiuav6k_common/annotations/instances_visible_common_train.json")
    ds["infrared_folder"] = "F:/data/Anti-UAV/Anti_UAV_6K/train/infrared/images"
    ds["infrared_label_folder"] = str(ROOT / "data/antiuav6k_ir_raw_verified/train/labels")
    loader = train_cfg.val_dataloader
    sums = None
    count, ids = 0, []
    with torch.inference_mode():
        for step, (samples, targets) in enumerate(loader):
            ir = samples[:, 3:6].to(solver.device)
            chunk_size = model.rgbt_thermal_forward_chunk_size
            if 0 < chunk_size < len(ir):
                pieces = [model.thermal_backbone(part) for part in ir.split(chunk_size, 0)]
                features = [torch.cat([part[level] for part in pieces], 0)
                            for level in range(len(pieces[0]))]
            else:
                features = model.thermal_backbone(ir)
            encoded = model.thermal_encoder(features[1:])
            if arm == "R1":
                values = (original(encoded),)
            else:
                values = original(features[0], encoded[0],
                                  branch.objectness_head(features[0]))[:3]
            if sums is None:
                sums = [torch.zeros_like(value[0], dtype=torch.float64) for value in values]
            for total, value in zip(sums, values):
                total.add_(value.detach().double().sum(0))
            count += len(samples)
            ids.extend(int(t["image_id"]) for t in targets)
            if step % 80 == 0:
                print(f"{arm} train_mean {count}", flush=True)
    if count != 3200 or len(set(ids)) != 3200:
        raise RuntimeError(f"Incomplete train prototype: {count} images")
    means = [(total / count).float().to(solver.device) for total in sums]

    # Confirm standalone extraction matches the detector's actual IR inputs.
    captured = {}
    def capture(this, *args):
        result = original(*args)
        captured["values"] = tuple(value.detach().clone() for value in
                                   ((result,) if arm == "R1" else result[:3]))
        return result
    setattr(branch, method_name, types.MethodType(capture, branch))
    with torch.inference_mode():
        samples, _ = next(iter(solver.val_dataloader))
        model(samples.to(solver.device))
        ir = samples[:, 3:6].to(solver.device)
        pieces = [model.thermal_backbone(part) for part in ir.split(model.rgbt_thermal_forward_chunk_size, 0)]
        features = [torch.cat([part[level] for part in pieces], 0)
                    for level in range(len(pieces[0]))]
        encoded = model.thermal_encoder(features[1:])
        extracted = (original(encoded),) if arm == "R1" else original(
            features[0], encoded[0], branch.objectness_head(features[0]))[:3]
        max_error = max(float((x-y).abs().max()) for x, y in zip(captured["values"], extracted))
    if max_error > 1e-5:
        raise RuntimeError(f"Standalone IR evidence does not match detector: {max_error}")

    if arm == "R1":
        def constant(this, encoded):
            return means[0].unsqueeze(0).expand(encoded[0].shape[0], -1, -1)
    else:
        def constant(this, thermal_s8, thermal_s16, objectness_logits):
            batch = thermal_s8.shape[0]
            tokens = means[0].unsqueeze(0).expand(batch, -1, -1)
            scores = means[1].unsqueeze(0).expand(batch, -1)
            background = means[2].unsqueeze(0).expand(batch, -1, -1)
            probability = objectness_logits.sigmoid()
            return tokens, scores, background, probability
    setattr(branch, method_name, types.MethodType(constant, branch))
    try:
        fixed, _ = evaluate(model, solver.criterion, solver.postprocessor,
                            solver.val_dataloader, solver.evaluator, solver.device,
                            epoch=-1, use_wandb=False)
    finally:
        setattr(branch, method_name, original)
    metrics = fixed["coco_eval_bbox"]
    result = {"schema": "m_layout_fixed_train_mean_ir_v1", "status": "PASS", "arm": arm,
              "checkpoint_sha256": sha(checkpoint), "training_images": count,
              "test_images": 1820, "train_ids_unique": True,
              "prototype_source": "All 3200 training images with eval transforms, no labels",
              "evidence_fields_fixed": ["IR tokens"] if arm == "R1" else
                  ["object tokens", "candidate scores", "background token"],
              "standalone_extraction_max_error": max_error,
              "fixed_metrics": metrics,
              "normal_metrics": reference["normal_metrics"],
              "m_bypassed_metrics": reference["m_bypassed_metrics"],
              "normal_minus_fixed_ap_pp": 100*(reference["normal_metrics"][0]-metrics[0]),
              "fixed_minus_m_bypassed_ap_pp": 100*(metrics[0]-reference["m_bypassed_metrics"][0]),
              "note": "Same weights. R2 objectness logits still computed but no per-image scores or tokens enter fusion."}
    result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
