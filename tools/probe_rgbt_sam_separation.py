"""Frozen feature probe on shared training-side inputs; no optimizer updates."""
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig


def regions(mask, height, width):
    weights = F.interpolate(mask[None, None].float(), size=(height, width), mode="area")[0, 0]
    support = (weights > 0).float()
    ring = (F.max_pool2d(support[None, None], 5, stride=1, padding=2)[0, 0] - support).clamp_min(0)
    return weights, ring


def separation(feature, inside, outside):
    # Mean squared prototype gap divided by within-region variance. Invariant
    # to shared scalar scaling; not a trained classifier or boundary accuracy.
    pixels = feature.float().flatten(1)
    moments = []
    for weight in (inside, outside):
        weight = weight.flatten()
        mass = weight.sum()
        mean = (pixels * weight).sum(1) / mass
        variance = ((pixels - mean[:, None]).square() * weight).sum() / mass
        moments.append((mean, variance))
    gap = (moments[0][0] - moments[1][0]).square().sum()
    return float(gap / (moments[0][1] + moments[1][1]).clamp_min(1e-8))


def main():
    torch.manual_seed(0)
    torch.set_num_threads(4)
    definitions = {
        "SAM": ("s_sgc2_sam_c_plus_m_sd22_half_b8a4_20e.yml", "C_PLUS_M_SD22_HALF_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV"),
        "NONE": ("c_plus_m_sd22_half_sgc0_b8a4_20e.yml", "C_PLUS_M_SD22_HALF_SGC0_B8A4_20E_TESTDEV"),
    }
    models, configs, metadata, captured, handles = {}, {}, {}, {}, []
    for name, (config_name, run_name) in definitions.items():
        cfg = YAMLConfig(str(REPO / "experiments/phase_s" / config_name))
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = 16
        cfg.yaml_cfg["train_dataloader"]["num_workers"] = 0
        cfg.yaml_cfg["train_dataloader"]["persistent_workers"] = False
        checkpoint = ROOT / "outputs" / run_name / "seed0/best_stg1.pth"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = cfg.model.cuda().eval().requires_grad_(False)
        model.load_state_dict(state["ema"]["module"], strict=True)
        models[name], configs[name] = model, cfg
        metadata[name] = {"checkpoint": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "epoch": state["last_epoch"]}
        del state
        def hook(module, args, output, label=name):
            captured[label] = ([x.detach() for x in args[0]], [x.detach() for x in output])
        handles.append(model.sd2_conditioner.register_forward_hook(hook))
    rows, ids, failures = [], [], {"no_mask": 0, "insufficient_common_mass": 0}
    # Build one loader only; both checkpoints receive exactly the same tensors.
    torch.manual_seed(0)
    loader = configs["SAM"].train_dataloader
    with torch.no_grad():
        for batch_index, (samples, targets) in enumerate(loader):
            if batch_index >= 4:
                break
            samples = samples.cuda()
            ids.extend(int(t["image_id"]) for t in targets)
            for model in models.values():
                model(samples)
            for image_index, target in enumerate(targets):
                masks = target.get("masks")
                if masks is None:
                    raise RuntimeError("Training loader omitted SAM masks")
                for target_index, mask in enumerate(masks):
                    if not bool(mask.any()):
                        failures["no_mask"] += 1
                        continue
                    mask = mask.cuda()
                    # Transformed training boxes use normalized cxcywh.
                    box = target["boxes"][target_index].float().cuda()
                    ih, iw = mask.shape[-2:]
                    assert bool((box >= 0).all()) and bool((box <= 1).all()), "Unexpected training box format"
                    xx = (torch.arange(iw, device="cuda") + .5) / iw
                    yy = (torch.arange(ih, device="cuda") + .5) / ih
                    rectangle = ((xx[None] >= box[0] - box[2]/2) & (xx[None] <= box[0] + box[2]/2)
                                 & (yy[:, None] >= box[1] - box[3]/2) & (yy[:, None] <= box[1] + box[3]/2))
                    area = float(box[2]*iw*box[3]*ih)
                    for level, feature in enumerate(captured["SAM"][0]):
                        h, w = feature.shape[-2:]
                        definitions_regions = {"SAM_region": regions(mask, h, w), "BOX_region": regions(rectangle, h, w)}
                        if any(float(a.sum()) < 4 or float(b.sum()) < 4 for a, b in definitions_regions.values()):
                            failures["insufficient_common_mass"] += 1
                            continue
                        record = {"image_id": int(target["image_id"]), "target_index": target_index, "level": level,
                                  "shape": [h, w], "transformed_area": area, "scale": "small" if area < 1024 else "other", "scores": {}}
                        for name, (before, after) in captured.items():
                            record["scores"][name] = {region: {"before": separation(before[level][image_index], a, b),
                                                              "after": separation(after[level][image_index], a, b)}
                                                       for region, (a, b) in definitions_regions.items()}
                        rows.append(record)
            print(f"batch={batch_index} images={len(ids)} valid_target_levels={len(rows)}", flush=True)
    for handle in handles:
        handle.remove()
    summary = {}
    for name in models:
        summary[name] = {}
        for level in (0, 1):
            for scale in ("small", "other"):
                selected = [r for r in rows if r["level"] == level and r["scale"] == scale]
                for region in ("SAM_region", "BOX_region"):
                    values = [r["scores"][name][region] for r in selected]
                    summary[name][f"level{level}_{scale}_{region}"] = {"n": len(values),
                        "mean_before": sum(v["before"] for v in values)/len(values) if values else None,
                        "mean_after": sum(v["after"] for v in values)/len(values) if values else None,
                        "increased": sum(v["after"] > v["before"] for v in values)}
    result = {"metadata": metadata, "image_ids": ids, "failures": failures, "summary": summary, "rows": rows,
              "caveats": ["64 augmented training inputs, shared between models; not held-out evaluation",
                          "best epochs differ; cross-model differences do not isolate SAM causally",
                          "region-dependent feature diagnostic, not true edge recovery or detection accuracy",
                          "common mass >=4 selects larger supported targets; excluded tiny targets not measured",
                          "scale uses transformed training box area, not original COCO evaluation area"]}
    dest = ROOT / "reports/147_core_claim_audit/feature_probe.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
