"""Matched nonlinear decoder readout; deliberately NOT called a linear probe."""
import argparse
import hashlib
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
from src.solver.rgb_preservation import load_rgb_features


def tensor_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if (output / "log.txt").exists():
        raise FileExistsError("refusing to overwrite an existing probe run")
    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)
    cfg = YAMLConfig(
        str(REPO / "experiments/phase_m/c_only_gq1_b8a4_20e_testdev_local.yml"),
        output_dir=str(output), epochs=args.epochs, use_ema=False,
        tuning=None, resume=None,
    )
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["lr_warmup_scheduler"]["warmup_duration"] = 100
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    # Same seed and generic initialization for both readout heads. Never import
    # a trained decoder from either source checkpoint.
    solver.model = cfg.model
    solver.load_tuning_state(str(REPO.parent / "weights/m_sd2_joint_coco_thermal_identity_init.pth"))
    load_rgb_features(solver.model, args.source)
    hooks = []
    for name in ("backbone", "encoder"):
        module = getattr(solver.model, name)
        module.eval().requires_grad_(False)
        hooks.append(module.register_forward_pre_hook(lambda module, inputs: module.eval() and None))
    initial = {name: tensor_hash(getattr(solver.model, name)) for name in ("backbone", "encoder", "decoder")}
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "frozen_RGB_features_common_nonlinear_decoder_readout",
        "source": args.source, "source_sha256": hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
        "epochs": args.epochs, "seed": 0, "batch": 8, "accumulation": 4,
        "ema": False, "initial_hashes": initial,
        "note": "Diagnostic only. Finite readout optimization is not a proof of representation quality.",
    }
    (output / "probe_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    solver.fit()
    final = {name: tensor_hash(getattr(solver.model, name)) for name in ("backbone", "encoder")}
    if any(final[name] != initial[name] for name in final):
        raise RuntimeError("frozen feature tensors or BN buffers changed")
    records = [json.loads(line) for line in (output / "log.txt").read_text().splitlines() if line.strip()]
    metrics = [r["test_coco_eval_bbox"][0] for r in records]
    manifest.update(status="completed", frozen_exact=True, AP_curve=metrics,
                    best_AP=max(metrics), last_AP=metrics[-1], last3_AP=sum(metrics[-3:]) / len(metrics[-3:]))
    (output / "probe_result.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for hook in hooks:
        hook.remove()
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
