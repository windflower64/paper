"""Read-only, same-checkpoint M-strength audit on the RGB testdev set.

Modes are off, trained strength, and twice trained strength. The latter is a
diagnostic intervention on a fixed checkpoint, not a newly trained model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "D-FINE"
RUNS = {
    "A5": (
        "experiments/phase_s/sqer2_sam_b8a4_20e.yml",
        "outputs/SQER2_SAM_B8A4_20E_TESTDEV/seed0/best_stg1.pth",
        0.537878469924924,
    ),
    "E1": (
        "experiments/phase_m/mote2_e1_sd22_correct_ir_b8a4_20e.yml",
        "outputs/M_OTE2_C_S_PAIR_20E_TESTDEV/E1/seed0/best_stg1.pth",
        0.523423555672387,
    ),
    "E2": (
        "experiments/phase_m/mote2_e2_ote2_b8a4_20e.yml",
        "outputs/M_OTE2_C_S_PAIR_20E_TESTDEV/E2/seed0/best_stg1.pth",
        0.5205378610069922,
    ),
}
MODES = {"off": 0.0, "trained": 1.0, "double": 2.0}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def model_digest(model) -> str:
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode("utf-8"))
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=RUNS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("testdev", "train"), default="testdev")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.glob(f"{args.arm}_*")):
        raise FileExistsError(f"Existing results for {args.arm}: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    from src.core import YAMLConfig
    from src.misc import dist_utils
    from src.solver import TASKS

    config_rel, checkpoint_rel, expected_ap = RUNS[args.arm]
    config = REPO / config_rel
    checkpoint = ROOT / checkpoint_rel
    for path in (config, checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)
    cfg = YAMLConfig(
        str(config), resume=str(checkpoint),
        output_dir=str(output_dir / f"runtime_{args.arm}"),
    )
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    if args.split == "train":
        dataset = cfg.yaml_cfg["val_dataloader"]["dataset"]
        dataset["img_folder"] = str(ROOT / "data/antiuav6k_common/images/train")
        dataset["ann_file"] = str(ROOT / "data/antiuav6k_common/annotations/instances_visible_common_train.json")
        dataset["infrared_folder"] = "F:/data/Anti-UAV/Anti_UAV_6K/train/infrared/images"
        dataset["infrared_label_folder"] = str(ROOT / "data/antiuav6k_ir_raw_verified/train/labels")
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    model = solver.ema.module if solver.ema else solver.model
    model.eval()
    epoch = int(solver.last_epoch)
    if hasattr(model, "set_training_epoch"):
        model.set_training_epoch(epoch)
    branch = model.mote_fusion if args.arm == "E2" else model.sd2_conditioner
    if branch is None:
        raise RuntimeError("Expected M branch is absent")
    original_strength = float(branch.final_residual_scale)
    if abs(original_strength - 0.5) > 1e-8:
        raise RuntimeError(f"Expected trained M strength 0.5, got {original_strength}")
    original_digest = model_digest(model)
    summary = {
        "schema": "m_fusion_influence_v1",
        "arm": args.arm,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": epoch,
        "weight_source": "ema" if solver.ema else "model",
        "sqer_bypass": bool(model.sqer_bypass),
        "trained_strength": original_strength,
        "split": args.split,
        "validation_images": len(solver.val_dataloader.dataset),
        "test_gt_used_in_forward": False,
        "modes": {},
    }
    expected_images = 1820 if args.split == "testdev" else 3200
    if summary["validation_images"] != expected_images:
        raise RuntimeError(f"Expected {expected_images} images in {args.split}")

    for mode in args.modes:
        multiplier = MODES[mode]
        branch.final_residual_scale = original_strength * multiplier
        evaluator = solver.evaluator
        evaluator.cleanup()
        ids, boxes, scores, labels, visible_counts, ir_counts = [], [], [], [], [], []
        with torch.inference_mode():
            for step, (samples, targets) in enumerate(solver.val_dataloader):
                samples = samples.to(solver.device)
                outputs = model(samples)
                sizes = torch.stack([target["orig_size"] for target in targets]).to(solver.device)
                results = solver.postprocessor(outputs, sizes)
                evaluator.update({int(t["image_id"]): r for t, r in zip(targets, results)})
                for target, result in zip(targets, results):
                    ids.append(int(target["image_id"]))
                    boxes.append(result["boxes"].detach().cpu().float().numpy())
                    scores.append(result["scores"].detach().cpu().float().numpy())
                    labels.append(result["labels"].detach().cpu().numpy())
                    visible_counts.append(len(target["boxes"]))
                    ir_counts.append(len(target["infrared_boxes"]))
                if step % 50 == 0:
                    print(args.arm, mode, step, len(ids), flush=True)
        if len(ids) != expected_images or len(set(ids)) != expected_images:
            raise RuntimeError("Validation image IDs are not complete and unique")
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        metrics = evaluator.coco_eval["bbox"].stats.tolist()
        summary["modes"][mode] = {
            "strength": branch.final_residual_scale,
            "coco_eval_bbox": metrics,
            "AP_percentage": metrics[0] * 100,
        }
        np.savez_compressed(
            output_dir / f"{args.arm}_{mode}_predictions.npz",
            image_ids=np.asarray(ids, dtype=np.int64),
            boxes=np.stack(boxes),
            scores=np.stack(scores),
            labels=np.stack(labels),
            visible_counts=np.asarray(visible_counts, dtype=np.int16),
            ir_counts=np.asarray(ir_counts, dtype=np.int16),
        )
        print(args.arm, mode, "AP", metrics[0] * 100, flush=True)

    branch.final_residual_scale = original_strength
    summary["weights_unchanged"] = model_digest(model) == original_digest
    if not summary["weights_unchanged"]:
        raise RuntimeError("Model weights changed during read-only evaluation")
    if args.split == "testdev" and "trained" in summary["modes"] and abs(summary["modes"]["trained"]["coco_eval_bbox"][0] - expected_ap) > 1e-6:
        raise RuntimeError("Trained-strength AP did not reproduce historical result")
    (output_dir / f"{args.arm}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    dist_utils.cleanup()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
