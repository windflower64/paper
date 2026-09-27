"""Audit the paired S-QER2 configs and initialization before warmup training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def digest_state(state, prefix: str | None = None) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        if prefix is not None and not name.startswith(prefix):
            continue
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--sam-config", type=Path, required=True)
    parser.add_argument("--box-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from src.core import YAMLConfig

    arm_rows = {}
    resolved = {}
    model_hashes = {}
    sqer_hashes = {}
    sqer_optimizer_groups = {}
    for name, path in (("sam", args.sam_config), ("box", args.box_config)):
        seed_all(0)
        cfg = YAMLConfig(str(path.resolve()))
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        model = cfg.model
        optimizer = cfg.optimizer
        names_by_id = {id(param): key for key, param in model.named_parameters()}
        groups = []
        for index, group in enumerate(optimizer.param_groups):
            group_names = [names_by_id.get(id(param), "<unknown>") for param in group["params"]]
            sqer_names = [key for key in group_names if key.startswith("sqer.")]
            if sqer_names:
                groups.append({
                    "group_index": index,
                    "lr": float(group["lr"]),
                    "parameter_tensors": len(sqer_names),
                    "parameter_count": sum(dict(model.named_parameters())[key].numel() for key in sqer_names),
                    "all_names_are_sqer": len(sqer_names) == len(group_names),
                })
        if len(groups) != 1 or not groups[0]["all_names_are_sqer"]:
            raise RuntimeError(f"{name}: S-QER parameters are not isolated in one optimizer group: {groups}")
        arm_rows[name] = {
            "resolved": copy.deepcopy(cfg.yaml_cfg),
            "model_state_sha256": digest_state(model.state_dict()),
            "sqer_state_sha256": digest_state(model.state_dict(), "sqer."),
            "sqer_parameters": sum(p.numel() for p in model.sqer.parameters()),
            "sqer_optimizer_groups": groups,
            "resume_path": cfg.yaml_cfg.get("resume"),
            "epochs": int(cfg.yaml_cfg["epochs"]),
            "sqer_bypass": bool(model.sqer_bypass),
            "sqer_evidence_only": bool(model.sqer_evidence_only),
            "sqer_shape_weight": float(cfg.criterion.sqer_shape_weight),
            "sqer_supervision": str(cfg.criterion.sqer_supervision),
        }

    sam_cfg = copy.deepcopy(arm_rows["sam"]["resolved"])
    box_cfg = copy.deepcopy(arm_rows["box"]["resolved"])
    for cfg in (sam_cfg, box_cfg):
        cfg["output_dir"] = "<paired-output>"
        cfg["resume"] = "<shared-warmup>"
        cfg["DFINECriterion"]["sqer_supervision"] = "<paired-teacher>"
    if sam_cfg != box_cfg:
        raise RuntimeError("SAM/BOX configs differ beyond output, shared resume, and teacher type")
    if arm_rows["sam"]["model_state_sha256"] != arm_rows["box"]["model_state_sha256"]:
        raise RuntimeError("SAM/BOX constructed model initializations differ")
    if arm_rows["sam"]["sqer_optimizer_groups"] != arm_rows["box"]["sqer_optimizer_groups"]:
        raise RuntimeError("SAM/BOX S-QER optimizer groups differ")
    if any(row["epochs"] != 20 or row["sqer_bypass"] or not row["sqer_evidence_only"] for row in arm_rows.values()):
        raise RuntimeError("formal arm configuration violates S-QER2 screening contract")
    if any(row["sqer_shape_weight"] != 0.05 for row in arm_rows.values()):
        raise RuntimeError("formal arm S-QER shape weight must be 0.05")
    if arm_rows["sam"]["sqer_supervision"] != "sam" or arm_rows["box"]["sqer_supervision"] != "box":
        raise RuntimeError("teacher supervision selectors are not SAM/BOX paired")

    report = {
        "status": "PASS",
        "allowed_config_differences": ["output_dir", "resume (same future warmup checkpoint)", "DFINECriterion.sqer_supervision"],
        "common_model_initialization_sha256": arm_rows["sam"]["model_state_sha256"],
        "common_sqer_initialization_sha256": arm_rows["sam"]["sqer_state_sha256"],
        "sam": {key: value for key, value in arm_rows["sam"].items() if key != "resolved"},
        "box": {key: value for key, value in arm_rows["box"].items() if key != "resolved"},
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
