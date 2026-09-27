"""Measure how M1 perturbs ordinary and HRQS query groups in CDM1.

The diagnostic is read-only with respect to the checkpoint.  It runs the same
validation samples once with the learned thermal residual and once with the
residual disabled, then reports decoder-layer and prediction changes for the
first 250 ordinary queries and the final 50 HRQS queries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ordinary-queries", type=int, default=250)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = cfg.model
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    state = checkpoint["ema"]["module"]
    model.load_state_dict(state, strict=True)
    model.eval().cuda()
    model.rgbt_thermal_intervention = "normal"

    couplers = list(model.decoder.sdtec_couplers)
    if len(couplers) != model.decoder.num_layers:
        raise RuntimeError("diagnostic requires independent per-layer couplers")
    original_max_scales = [coupler.max_residual_scale for coupler in couplers]

    mode = {"name": "normal"}
    group_records = {
        "normal": [
            {
                "sample_sum_ordinary_delta_norm": 0.0,
                "sample_sum_hrqs_delta_norm": 0.0,
                "sample_sum_ordinary_relative_delta": 0.0,
                "sample_sum_hrqs_relative_delta": 0.0,
                "samples": 0,
            }
            for _ in couplers
        ],
        "disabled": [
            {
                "sample_sum_ordinary_delta_norm": 0.0,
                "sample_sum_hrqs_delta_norm": 0.0,
                "sample_sum_ordinary_relative_delta": 0.0,
                "sample_sum_hrqs_relative_delta": 0.0,
                "samples": 0,
            }
            for _ in couplers
        ],
    }

    def make_hook(layer_index: int):
        def hook(_module, inputs, outputs):
            before = inputs[0].detach().float()
            after = outputs[0].detach().float()
            query_count = before.shape[1]
            split = args.ordinary_queries
            if query_count <= split:
                raise RuntimeError(
                    f"expected HRQS queries after index {split}, got {query_count}"
                )
            delta = (after - before).norm(dim=-1)
            relative = delta / before.norm(dim=-1).clamp_min(1e-8)
            record = group_records[mode["name"]][layer_index]
            batch_size = before.shape[0]
            record["sample_sum_ordinary_delta_norm"] += float(
                delta[:, :split].mean(dim=1).sum().cpu()
            )
            record["sample_sum_hrqs_delta_norm"] += float(
                delta[:, split:].mean(dim=1).sum().cpu()
            )
            record["sample_sum_ordinary_relative_delta"] += float(
                relative[:, :split].mean(dim=1).sum().cpu()
            )
            record["sample_sum_hrqs_relative_delta"] += float(
                relative[:, split:].mean(dim=1).sum().cpu()
            )
            record["samples"] += batch_size

        return hook

    handles = [
        coupler.register_forward_hook(make_hook(index))
        for index, coupler in enumerate(couplers)
    ]
    prediction_sums = {
        "ordinary_logit_abs": 0.0,
        "hrqs_logit_abs": 0.0,
        "ordinary_box_l1": 0.0,
        "hrqs_box_l1": 0.0,
        "samples": 0,
    }

    with torch.inference_mode():
        for batch_index, (samples, _targets) in enumerate(cfg.val_dataloader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            samples = samples.cuda(non_blocking=True)
            mode["name"] = "normal"
            for coupler, max_scale in zip(couplers, original_max_scales):
                coupler.max_residual_scale = max_scale
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                normal = model(samples)

            mode["name"] = "disabled"
            for coupler in couplers:
                coupler.max_residual_scale = 0.0
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                disabled = model(samples)

            split = args.ordinary_queries
            logit_delta = (
                normal["pred_logits"].float() - disabled["pred_logits"].float()
            ).abs().mean(dim=-1)
            box_delta = (
                normal["pred_boxes"].float() - disabled["pred_boxes"].float()
            ).abs().mean(dim=-1)
            prediction_sums["ordinary_logit_abs"] += float(
                logit_delta[:, :split].mean(dim=1).sum().cpu()
            )
            prediction_sums["hrqs_logit_abs"] += float(
                logit_delta[:, split:].mean(dim=1).sum().cpu()
            )
            prediction_sums["ordinary_box_l1"] += float(
                box_delta[:, :split].mean(dim=1).sum().cpu()
            )
            prediction_sums["hrqs_box_l1"] += float(
                box_delta[:, split:].mean(dim=1).sum().cpu()
            )
            prediction_sums["samples"] += int(samples.shape[0])

    for handle in handles:
        handle.remove()
    for coupler, max_scale in zip(couplers, original_max_scales):
        coupler.max_residual_scale = max_scale

    def finalize(record):
        count = record.pop("samples")
        return {
            "samples": count,
            "ordinary_delta_norm": record[
                "sample_sum_ordinary_delta_norm"
            ]
            / count,
            "hrqs_delta_norm": record["sample_sum_hrqs_delta_norm"] / count,
            "hrqs_over_ordinary_delta_norm": record[
                "sample_sum_hrqs_delta_norm"
            ]
            / max(record["sample_sum_ordinary_delta_norm"], 1e-12),
            "ordinary_relative_delta": record[
                "sample_sum_ordinary_relative_delta"
            ]
            / count,
            "hrqs_relative_delta": record[
                "sample_sum_hrqs_relative_delta"
            ]
            / count,
        }

    sample_count = prediction_sums.pop("samples")
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_state": "ema.module",
        "ordinary_query_range": [0, args.ordinary_queries - 1],
        "hrqs_query_range": [args.ordinary_queries, model.decoder.num_queries - 1],
        "max_residual_scales": original_max_scales,
        "layers": {
            key: [finalize(record) for record in records]
            for key, records in group_records.items()
        },
        "prediction_change_normal_vs_disabled": {
            key: value / sample_count for key, value in prediction_sums.items()
        },
        "samples": sample_count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
