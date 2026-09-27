"""Read-only M-series mechanism audit. Never updates detector parameters.

The layout task instruments a private in-memory copy of M-OTE2.forward.
The prototype task replaces per-image A5 IR tokens by their training-set mean.
Neither intervention changes production code or any saved checkpoint.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import textwrap
import types
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "D-FINE"
sys.path.insert(0, str(REPO))
from audit_m_fusion_influence import model_digest, sha256


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_solver(config, checkpoint, output):
    from src.core import YAMLConfig
    from src.solver import TASKS

    cfg = YAMLConfig(str(REPO / config), resume=str(checkpoint), output_dir=str(output))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    model = solver.ema.module if solver.ema else solver.model
    model.eval()
    model.set_training_epoch(int(solver.last_epoch))
    return solver, model


def instrumented_forward(branch, fix_layout=False):
    source = textwrap.dedent(inspect.getsource(type(branch).forward))
    bad = "raw_update = (full_response - null_response).transpose(1, 2).reshape(batch, self.visible_channels, height, width)"
    if source.count(bad) != 1:
        raise RuntimeError("Unexpected source: layout audit no longer applies")
    source = source.replace(bad, "intended_update = full_response - null_response\n    " + (
        "raw_update = intended_update" if fix_layout else bad
    ))
    source = source.replace("raw_update = raw_update * confidence * gate_map", "permuted_update = raw_update\n    raw_update = raw_update * confidence * gate_map")
    source = source.replace("return output, objectness_loss", "self._mechanism_audit = {\n        'intended': intended_update.detach(), 'permuted': permuted_update.detach(),\n        'gated': raw_update.detach(), 'update': update.detach(),\n        'gate': gate_map.detach(), 'confidence': confidence.detach(),\n        'cap': cap.detach(), 'limiter': limiter.detach(),\n        'attention': attention.detach(),\n    }\n    return output, objectness_loss")
    namespace = dict(type(branch).forward.__globals__)
    exec(compile(source, "<read_only_mote_layout_audit>", "exec"), namespace)
    return types.MethodType(namespace["forward"], branch)


def rms(x):
    return x.float().square().mean(dim=(1, 2, 3)).sqrt()


def layout_audit(output):
    from src.zoo.dfine.target_evidence_thermal_fusion import TargetEvidenceThermalFusion

    cases = []
    for c, h, w in ((128, 32, 40), (128, 40, 40), (8, 4, 5)):
        indices = torch.arange(c * h * w).reshape(1, c, h, w)
        permuted = indices.transpose(1, 2).reshape_as(indices)
        cc = torch.arange(c).reshape(1, c, 1, 1).expand_as(indices)
        hh = torch.arange(h).reshape(1, 1, h, 1).expand_as(indices)
        cases.append({"shape": [1, c, h, w], "same_element_fraction": (indices == permuted).float().mean().item(),
                      "same_channel_fraction": (permuted // (h * w) == cc).float().mean().item(),
                      "same_row_fraction": ((permuted // w) % h == hh).float().mean().item()})
    # The branch uses global normalization and pointwise RGB queries, so a
    # cyclic shift of RGB alone must shift its residual by the same amount.
    torch.manual_seed(42)
    branch = TargetEvidenceThermalFusion(8, 12, 8, embed_dim=8).eval()
    with torch.no_grad():
        branch.residual_generator[-1].weight.normal_(0, 0.1)
    rgb = torch.randn(2, 8, 4, 5)
    ir8, ir16 = torch.randn(2, 12, 8, 10), torch.randn(2, 8, 4, 5)
    equivariance = {}
    with torch.inference_mode():
        for mode, fixed in (("as_trained", False), ("layout_corrected_in_memory", True)):
            branch.forward = instrumented_forward(branch, fixed)
            a = branch(rgb, ir8, ir16)[0] - rgb
            shifted = rgb.roll(1, dims=2)
            b = branch(shifted, ir8, ir16)[0] - shifted
            equivariance[mode] = float((b - a.roll(1, dims=2)).norm() / a.norm().clamp_min(1e-12))
    if equivariance["as_trained"] < 0.1 or equivariance["layout_corrected_in_memory"] > 1e-4:
        raise RuntimeError(f"Layout regression check failed: {equivariance}")

    result = {"schema": "mote_layout_and_gate_audit_v1", "index_mapping": cases,
              "synthetic_rgb_row_shift_relative_error": equivariance,
              "production_code_modified": False, "training_performed": False, "checkpoints": {}}
    for label, filename in (("best", "best_stg1.pth"), ("last", "last.pth")):
        checkpoint = ROOT / "outputs/M_OTE2_C_S_PAIR_20E_TESTDEV/E2/seed0" / filename
        solver, model = load_solver("experiments/phase_m/mote2_e2_ote2_b8a4_20e.yml", checkpoint, output / f"layout_{label}_runtime")
        digest = model_digest(model)
        original_forward = model.mote_fusion.forward
        model.mote_fusion.forward = instrumented_forward(model.mote_fusion)
        rows, exact_error = [], []
        with torch.inference_mode():
            for step, (samples, _) in enumerate(solver.val_dataloader):
                if step == 8:
                    break
                samples = samples.to(solver.device)
                prediction = model(samples)
                trace = model.mote_fusion._mechanism_audit
                if step == 0:
                    model.mote_fusion.forward = original_forward
                    reference = model(samples)
                    exact_error.append(max(float((prediction[k] - reference[k]).abs().max()) for k in ("pred_logits", "pred_boxes")))
                    model.mote_fusion.forward = instrumented_forward(model.mote_fusion)
                cosine = F.cosine_similarity(trace["intended"].flatten(1), trace["permuted"].flatten(1), dim=1)
                entropy = -(trace["attention"].clamp_min(1e-8) * trace["attention"].clamp_min(1e-8).log()).sum(-1).mean(-1)
                cap = trace["cap"].flatten()
                s = rms(trace["gated"])
                # Halving the complete gate field while holding generated
                # content fixed. Reports attenuation after the RMS limiter.
                ratio_half = 0.5 * ((s.square() + cap.square() + 2e-6) / (0.25 * s.square() + cap.square() + 2e-6)).sqrt()
                for i in range(len(samples)):
                    rows.append({"cosine_intended_vs_written_layout": float(cosine[i]),
                                 "raw_rms": float(rms(trace["intended"])[i]),
                                 "gated_rms": float(s[i]), "cap_rms": float(cap[i]),
                                 "gate_mean": float(trace["gate"][i].mean()),
                                 "gate_min": float(trace["gate"][i].min()),
                                 "gate_max": float(trace["gate"][i].max()),
                                 "limiter": float(trace["limiter"].flatten()[i]),
                                 "halved_gate_final_norm_ratio": float(ratio_half[i]),
                                 "attention_entropy": float(entropy[i]),
                                 "s16_shape": list(trace["update"].shape[1:])})
        model.mote_fusion.forward = original_forward
        if model_digest(model) != digest or max(exact_error) != 0.0:
            raise RuntimeError("Read-only instrumentation changed weights or predictions")
        result["checkpoints"][label] = {"checkpoint": str(checkpoint), "sha256": sha256(checkpoint),
            "epoch": int(solver.last_epoch), "images": len(rows), "instrumentation_prediction_max_error": max(exact_error),
            "weights_unchanged": True, "means": {k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if k != "s16_shape"},
            "rows": rows}
        del solver, model
        torch.cuda.empty_cache()
    dump(output / "layout_and_gate_audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "checkpoints"}, indent=2), flush=True)
    for label, data in result["checkpoints"].items():
        print(label, json.dumps(data["means"], indent=2), flush=True)


def prototype_audit(output):
    from src.core import YAMLConfig

    config = "experiments/phase_s/sqer2_sam_b8a4_20e.yml"
    checkpoint = ROOT / "outputs/SQER2_SAM_B8A4_20E_TESTDEV/seed0/best_stg1.pth"
    solver, model = load_solver(config, checkpoint, output / "prototype_runtime")
    digest = model_digest(model)
    branch = model.sd2_conditioner
    original_tokenizer = branch._thermal_tokens
    def extract_tokens(ir):
        chunk_size = model.rgbt_thermal_forward_chunk_size
        if 0 < chunk_size < ir.shape[0]:
            chunks = [model.thermal_backbone(part) for part in ir.split(chunk_size, 0)]
            features = [torch.cat([part[level] for part in chunks], 0) for level in range(len(chunks[0]))]
        else:
            features = model.thermal_backbone(ir)
        encoded = model.thermal_encoder(features[-len(model.thermal_encoder.in_channels):])
        return original_tokenizer(encoded)
    train_cfg = YAMLConfig(str(REPO / config))
    train_cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    ds = train_cfg.yaml_cfg["val_dataloader"]["dataset"]
    ds["img_folder"] = str(ROOT / "data/antiuav6k_common/images/train")
    ds["ann_file"] = str(ROOT / "data/antiuav6k_common/annotations/instances_visible_common_train.json")
    ds["infrared_folder"] = "F:/data/Anti-UAV/Anti_UAV_6K/train/infrared/images"
    ds["infrared_label_folder"] = str(ROOT / "data/antiuav6k_ir_raw_verified/train/labels")
    train_loader = train_cfg.val_dataloader
    total, square_total, count, ids = None, None, 0, []
    with torch.inference_mode():
        for step, (samples, targets) in enumerate(train_loader):
            ir = samples[:, 3:6].to(solver.device)
            tokens = extract_tokens(ir).double()
            total = tokens.sum(0) if total is None else total + tokens.sum(0)
            square_total = tokens.square().sum(0) if square_total is None else square_total + tokens.square().sum(0)
            count += len(samples)
            ids.extend(int(t["image_id"]) for t in targets)
            if step % 80 == 0:
                print("training_mean_tokens", step, count, flush=True)
    if count != 3200 or len(set(ids)) != 3200:
        raise RuntimeError("Incomplete train-only token prototype")
    prototype = (total / count).float().unsqueeze(0)
    variance = (square_total / count - (total / count).square()).clamp_min(0)
    torch.save({"prototype": prototype.cpu(), "training_images": count,
                "checkpoint_sha256": sha256(checkpoint)}, output / "a5_train_mean_tokens.pth")
    prototype_result = {"schema": "a5_train_mean_ir_tokens_v1", "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256(checkpoint), "epoch": int(solver.last_epoch),
                        "prototype_source": "all 3200 training images, evaluation transforms, no labels used",
                        "prototype_shape": list(prototype.shape),
                        "token_across_image_std_rms": float(variance.mean().sqrt()),
                        "token_mean_rms": float(prototype.square().mean().sqrt()),
                        "uses_test_gt_in_forward": False, "training_performed": False,
                        "interpretation": "Fixed-checkpoint diagnostic, not a trained RGB-only control or formal ablation."}
    # Validate the extracted tokens against the actual detector's IR path.
    captured = {}
    def capture_tokens(this, features):
        value = original_tokenizer(features)
        captured["value"] = value.detach()
        return value
    branch._thermal_tokens = types.MethodType(capture_tokens, branch)
    with torch.inference_mode():
        samples, _ = next(iter(solver.val_dataloader))
        samples = samples.to(solver.device)
        model(samples)
        prototype_result["standalone_token_extraction_max_error"] = float((captured["value"] - extract_tokens(samples[:, 3:6])).abs().max())
    if prototype_result["standalone_token_extraction_max_error"] > 1e-5:
        dump(output / "INVALID_token_extraction_mismatch.json", prototype_result)
        raise RuntimeError("Token extraction differs from actual detector path")
    def constant_tokens(this, features):
        return prototype.expand(features[0].shape[0], -1, -1)
    branch._thermal_tokens = types.MethodType(constant_tokens, branch)
    evaluator = solver.evaluator
    evaluator.cleanup()
    evaluated_ids = []
    with torch.inference_mode():
        for step, (samples, targets) in enumerate(solver.val_dataloader):
            outputs = model(samples.to(solver.device))
            sizes = torch.stack([t["orig_size"] for t in targets]).to(solver.device)
            results = solver.postprocessor(outputs, sizes)
            evaluator.update({int(t["image_id"]): r for t, r in zip(targets, results)})
            evaluated_ids.extend(int(t["image_id"]) for t in targets)
            if step % 50 == 0:
                print("constant_IR_prototype_testdev", step, len(evaluated_ids), flush=True)
    if len(evaluated_ids) != 1820 or len(set(evaluated_ids)) != 1820:
        raise RuntimeError("Incomplete testdev evaluation")
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    branch._thermal_tokens = original_tokenizer
    prototype_result["weights_unchanged"] = model_digest(model) == digest
    if not prototype_result["weights_unchanged"]:
        raise RuntimeError("Checkpoint tensors changed")
    historical = json.loads((ROOT / "reports/M_FUSION_INFLUENCE_AUDIT_V2/A5_summary.json").read_text(encoding="utf-8"))
    if historical["checkpoint_sha256"] != prototype_result["checkpoint_sha256"]:
        raise RuntimeError("Historical baseline belongs to a different checkpoint")
    prototype_result["metrics"] = {"constant_train_mean_IR": evaluator.coco_eval["bbox"].stats.tolist(),
        "original_real_IR_reference": historical["modes"]["trained"]["coco_eval_bbox"],
        "M_off_reference": historical["modes"]["off"]["coco_eval_bbox"]}
    dump(output / "a5_prototype_result.json", prototype_result)
    print(json.dumps(prototype_result, indent=2), flush=True)


def gradient_audit(output):
    from src.zoo.dfine.target_evidence_thermal_fusion import TargetEvidenceThermalFusion

    torch.manual_seed(42)
    branch = TargetEvidenceThermalFusion(8, 12, 8, embed_dim=8).train()
    targets = [{"infrared_boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])},
               {"infrared_boxes": torch.empty(0, 4)}]
    _, loss = branch(torch.randn(2, 8, 4, 5), torch.randn(2, 12, 8, 10),
                     torch.randn(2, 8, 4, 5), targets=targets)
    loss.backward()
    groups = {}
    for name, parameter in branch.named_parameters():
        group = groups.setdefault(name.split(".")[0], {"tensor_count": 0, "nonzero_grad_count": 0})
        group["tensor_count"] += 1
        group["nonzero_grad_count"] += int(parameter.grad is not None and bool(parameter.grad.abs().max() > 0))
    result = {"schema": "mote_ir_objectness_only_gradient_v1", "parameter_groups": groups,
              "uses_synthetic_tensors": True, "checkpoint_modified": False,
              "optimizer_steps": 0, "loss": float(loss.detach()),
              "interpretation": "IR localization supervision directly updates the objectness head only; cross-modal reader and generator depend on detection gradients."}
    dump(output / "objectness_gradient_audit.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("layout", "prototype", "gradient"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    expected = {"layout": "layout_and_gate_audit.json", "prototype": "a5_prototype_result.json",
                "gradient": "objectness_gradient_audit.json"}[args.task]
    if (output / expected).exists():
        raise FileExistsError(output / expected)
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    from src.misc import dist_utils
    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)
    try:
        {"layout": layout_audit, "prototype": prototype_audit, "gradient": gradient_audit}[args.task](output)
    finally:
        dist_utils.cleanup()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
