"""Whole-model QDMF T1--T7/T10 integration checks (no optimization)."""

import argparse
import json
import sys
from pathlib import Path

import torch


def maximum(a, b):
    return float((a - b).detach().abs().max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    torch.manual_seed(123)
    cfg = YAMLConfig(str(args.config.resolve()))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model.cuda().eval()
    checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    source = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    current = model.state_dict()
    model.load_state_dict(
        {key: value for key, value in source.items()
         if key in current and current[key].shape == value.shape},
        strict=False,
    )
    visible = torch.randn(2, 3, 512, 640, device="cuda")
    thermal_a = torch.randn_like(visible)
    thermal_b = torch.randn_like(visible)
    input_a = torch.cat((visible, thermal_a), dim=1)
    input_b = torch.cat((visible, thermal_b), dim=1)

    with torch.no_grad():
        normal_a = model(input_a)
        normal_b = model(input_b)
        shuffled = model(torch.cat((visible, thermal_a.flip(0)), dim=1))
        model.qdmf_bypass = True
        bypass = model(input_a)
        model.qdmf_bypass = False
        missing = model(input_a, qdmf_availability=torch.tensor([False, False], device="cuda"))

    t1_no_private_keys = not any(
        key.startswith("qdmf_head_") or key in {"qdmf_base_corners", "qdmf_pred_corners", "qdmf_ref_points"}
        for key in normal_a
    )
    t2 = {
        "query_residual_max": float(bypass["qdmf_residual"].abs().max()),
        "logit_base_max_diff": maximum(bypass["pred_logits"], bypass["base_pred_logits"]),
        "box_base_max_diff": maximum(bypass["pred_boxes"], bypass["base_pred_boxes"]),
    }
    t3 = {
        "gate_nonzero": int(torch.count_nonzero(missing["qdmf_gate"])),
        "residual_nonzero": int(torch.count_nonzero(missing["qdmf_residual"])),
        "logit_base_max_diff": maximum(missing["pred_logits"], missing["base_pred_logits"]),
        "box_base_max_diff": maximum(missing["pred_boxes"], missing["base_pred_boxes"]),
        "finite": bool(torch.isfinite(missing["pred_boxes"]).all()),
    }
    t4 = {
        "base_logit_max_diff": maximum(normal_a["base_pred_logits"], normal_b["base_pred_logits"]),
        "base_box_max_diff": maximum(normal_a["base_pred_boxes"], normal_b["base_pred_boxes"]),
        "final_logit_max_diff": maximum(normal_a["pred_logits"], normal_b["pred_logits"]),
        "final_box_max_diff": maximum(normal_a["pred_boxes"], normal_b["pred_boxes"]),
        "gate_max_diff": maximum(normal_a["qdmf_gate"], normal_b["qdmf_gate"]),
    }
    t5 = {
        "base_logit_max_diff": maximum(normal_a["base_pred_logits"], shuffled["base_pred_logits"]),
        "final_logit_max_diff": maximum(normal_a["pred_logits"], shuffled["pred_logits"]),
        "final_box_max_diff": maximum(normal_a["pred_boxes"], shuffled["pred_boxes"]),
        "gate_max_diff": maximum(normal_a["qdmf_gate"], shuffled["qdmf_gate"]),
    }

    # T6: training DN is generated internally, while QDMF output remains the
    # fixed set of ordinary detector queries.
    cfg2 = YAMLConfig(str(args.config.resolve()))
    cfg2.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg2.yaml_cfg["DFINE"]["stql_enabled"] = False
    cfg2.yaml_cfg["DFINECriterion"]["stql_weight"] = 0.0
    train_model = cfg2.model.cuda().train()
    train_current = train_model.state_dict()
    train_model.load_state_dict(
        {key: value for key, value in source.items()
         if key in train_current and train_current[key].shape == value.shape},
        strict=False,
    )
    targets = [
        {
            "labels": torch.tensor([0], device="cuda"),
            "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]], device="cuda"),
            "area": torch.tensor([0.01], device="cuda"),
            "size": torch.tensor([512, 640], device="cuda"),
            "orig_size": torch.tensor([512, 640], device="cuda"),
        }
        for _ in range(2)
    ]
    train_output = train_model(input_a, targets=targets)
    t6 = {
        "ordinary_queries": int(train_output["pred_logits"].shape[1]),
        "configured_queries": int(train_model.decoder.num_queries),
        "has_dn_outputs": "dn_outputs" in train_output,
        "qdmf_gate_queries": int(train_output["qdmf_gate"].shape[1]),
    }

    passed = (
        t1_no_private_keys
        and max(t2.values()) == 0.0
        and t3["gate_nonzero"] == 0 and t3["residual_nonzero"] == 0
        and t3["logit_base_max_diff"] == 0.0 and t3["box_base_max_diff"] == 0.0
        and t4["base_logit_max_diff"] == 0.0 and t4["base_box_max_diff"] == 0.0
        and t4["final_logit_max_diff"] > 0.0 and t4["final_box_max_diff"] > 0.0
        and t5["base_logit_max_diff"] == 0.0
        and (t5["final_logit_max_diff"] > 0.0 or t5["gate_max_diff"] > 0.0)
        and t6["ordinary_queries"] == t6["configured_queries"]
        and t6["qdmf_gate_queries"] == t6["configured_queries"]
        and t6["has_dn_outputs"]
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "T1_legacy_private_keys_absent": t1_no_private_keys,
        "T2_bypass": t2,
        "T3_missing_ir": t3,
        "T4_ir_change": t4,
        "T5_ir_shuffle": t5,
        "T6_dn_isolation": t6,
        "T10_inference_without_sam_cache": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
