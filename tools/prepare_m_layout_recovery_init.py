"""Construct one deterministic COCO-start RGB/S + verified-IR initializer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "D-FINE"
DEFAULT_OLD = ROOT / "weights/m_sd2_joint_coco_thermal_identity_init.pth"
DEFAULT_IR = ROOT / "outputs/M_IR_RAW_VERIFIED_GQ1_30E_TESTDEV/seed0/best_stg1.pth"
DEFAULT_OUTPUT = ROOT / "outputs/M_LAYOUT_RECOVERY_20E_TESTDEV/common_init.pth"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ema_state(payload):
    return payload["ema"]["module"] if "ema" in payload else payload["model"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in (args.old, args.ir):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256(args.ir) != "9416a3c4734e7b15e86889ae6bc3161f10051efc1717c5800b3bb10bc234f0a0":
        raise RuntimeError("Corrected IR source has an unexpected SHA256")

    sys.path.insert(0, str(REPO))
    from src.core import YAMLConfig
    from src.misc import dist_utils

    dist_utils.setup_seed(0)
    r1_path = REPO / "experiments/phase_m/m_layout_recovery_r1_warmup.yml"
    cfg = YAMLConfig(str(r1_path))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    expected = cfg.model.state_dict()
    old_payload = torch.load(args.old, map_location="cpu", weights_only=False)
    ir_payload = torch.load(args.ir, map_location="cpu", weights_only=False)
    if "msd2_joint_init" not in old_payload:
        raise RuntimeError("Old source is not the audited COCO-start initializer")
    old = ema_state(old_payload)
    ir = ema_state(ir_payload)
    common = {}
    loaded_rgb, loaded_ir, fresh = [], [], []
    for key, template in expected.items():
        if key.startswith("sd2_conditioner."):
            continue
        if key.startswith(("thermal_backbone.", "thermal_encoder.")):
            source_key = key.replace("thermal_", "", 1)
            source = ir.get(source_key)
            if source is None or source.shape != template.shape:
                raise RuntimeError(f"Verified IR mapping missing or incompatible: {key} <- {source_key}")
            common[key] = source.detach().cpu().clone()
            loaded_ir.append(key)
        elif key.startswith(("backbone.", "encoder.", "decoder.")):
            source = old.get(key)
            if source is not None and source.shape == template.shape:
                common[key] = source.detach().cpu().clone()
                loaded_rgb.append(key)
            else:
                common[key] = template.detach().cpu().clone()
                fresh.append(key)
        elif key.startswith("sqer."):
            common[key] = template.detach().cpu().clone()
            fresh.append(key)
        else:
            raise RuntimeError(f"Unexpected shared tensor key: {key}")
    if len(loaded_ir) != 564 or not any("pat_sf" in key for key in fresh):
        raise RuntimeError("Unexpected IR or C initialization key counts")
    if not any("score_head" in key for key in fresh):
        raise RuntimeError("Expected fresh task-specific score heads")

    dist_utils.setup_seed(0)
    r2_path = REPO / "experiments/phase_m/m_layout_recovery_r2_warmup.yml"
    r2 = YAMLConfig(str(r2_path))
    r2.yaml_cfg["HGNetv2"]["pretrained"] = False
    r2_state = r2.model.state_dict()
    if set(common) != {key for key in r2_state if not key.startswith("mote_fusion.")}:
        raise RuntimeError("R1 and R2 shared key sets differ")
    for key, value in common.items():
        if r2_state[key].shape != value.shape:
            raise RuntimeError(f"R2 shared tensor shape differs: {key}")

    metadata = {
        "schema": "m_layout_recovery_common_init_v1",
        "old_coco_source": str(args.old.resolve()), "old_source_sha256": sha256(args.old),
        "original_coco_visible_sha256": old_payload["msd2_joint_init"]["visible_sha256"],
        "corrected_ir_source": str(args.ir.resolve()), "corrected_ir_sha256": sha256(args.ir),
        "seed": 0, "shared_key_count": len(common),
        "loaded_rgb_keys": len(loaded_rgb), "loaded_ir_keys": len(loaded_ir),
        "fresh_task_keys": fresh, "fresh_task_key_count": len(fresh),
        "method_unique_parameters": "constructed by each model from seed 0; no pretrained M key copied",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": common, "ema": {"module": common, "updates": 0},
                "m_layout_recovery_init": metadata}, args.output)
    metadata["output"] = str(args.output.resolve())
    metadata["output_sha256"] = sha256(args.output)
    audit = args.output.parent / "common_init_audit.json"
    audit.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "fresh_task_keys"}, indent=2))


if __name__ == "__main__":
    main()
