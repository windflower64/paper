#!/usr/bin/env python3
"""Run the established custom-size evaluator with PAT_sf branch interventions.

The interventions are evaluation-only causal probes. They do not modify the
checkpoint or retrain the model. Spatial-mean modes preserve each branch's
per-channel mean while removing its position-specific representation.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "learned",
            "local_mean",
            "attention_mean",
            "attention_identity",
            "attention_norm",
            "global_query",
            "value_mean",
            "zero_local",
            "zero_attention",
        ),
        default="learned",
    )
    return parser.parse_args()


args = parse_args()
sys.path.insert(0, str(args.repo))

from src.nn.backbone.partialnet_pat_sf import PartialSelfAttentionConv  # noqa: E402


mode = args.mode
stats = {
    "mode": mode,
    "calls": 0,
    "local_sum_sq": 0.0,
    "local_elements": 0,
    "attention_sum_sq": 0.0,
    "attention_elements": 0,
    "local_spatial_residual_sum_sq": 0.0,
    "attention_spatial_residual_sum_sq": 0.0,
}


def intervened_forward(self, x):
    local_x, attention_input = torch.split(
        x,
        [self.dim_conv3, self.dim_untouched],
        dim=1,
    )
    local_x = self.partial_conv3(local_x)
    attention_norm = self.norm(attention_input)
    if mode in {"global_query", "value_mean"}:
        batch, channels, height, width = attention_norm.shape
        tokens = attention_norm.flatten(2).transpose(1, 2)
        token_count = tokens.shape[1]
        heads = self.attn.num_heads
        head_dim = channels // heads
        qkv = (
            self.attn.qkv(tokens)
            .reshape(batch, token_count, 3, heads, head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        if mode == "global_query":
            global_q = q.mean(dim=2, keepdim=True) * self.attn.scale
            weights = (global_q @ k.transpose(-2, -1)).softmax(dim=-1)
            global_out = weights @ v
        else:
            global_out = v.mean(dim=2, keepdim=True)
        global_out = (
            global_out.transpose(1, 2)
            .reshape(batch, 1, channels)
        )
        global_out = self.attn.proj(global_out)
        attention_x = (
            global_out.transpose(1, 2)
            .reshape(batch, channels, 1, 1)
            .expand(batch, channels, height, width)
        )
    else:
        attention_x = self.attn(attention_norm)

    stats["calls"] += 1
    stats["local_sum_sq"] += float(local_x.detach().float().square().sum())
    stats["local_elements"] += local_x.numel()
    stats["attention_sum_sq"] += float(attention_x.detach().float().square().sum())
    stats["attention_elements"] += attention_x.numel()
    local_float = local_x.detach().float()
    attention_float = attention_x.detach().float()
    stats["local_spatial_residual_sum_sq"] += float(
        (local_float - local_float.mean(dim=(-2, -1), keepdim=True)).square().sum()
    )
    stats["attention_spatial_residual_sum_sq"] += float(
        (
            attention_float
            - attention_float.mean(dim=(-2, -1), keepdim=True)
        ).square().sum()
    )

    if mode == "local_mean":
        local_x = local_x.mean(dim=(-2, -1), keepdim=True).expand_as(local_x)
    elif mode == "attention_mean":
        attention_x = attention_x.mean(dim=(-2, -1), keepdim=True).expand_as(attention_x)
    elif mode == "attention_identity":
        attention_x = attention_input
    elif mode == "attention_norm":
        attention_x = attention_norm
    elif mode == "zero_local":
        local_x = torch.zeros_like(local_x)
    elif mode == "zero_attention":
        attention_x = torch.zeros_like(attention_x)

    return torch.cat((local_x, attention_x), dim=1)


PartialSelfAttentionConv.forward = intervened_forward

evaluator = args.repo / "evaluate_custom_sizes_and_importance.py"
sys.argv = [
    str(evaluator),
    "--repo", str(args.repo),
    "--config", str(args.config),
    "--checkpoint", str(args.checkpoint),
    "--output-dir", str(args.output_dir),
    "--weight-source", "ema",
    "--skip-importance",
]

runpy.run_path(str(evaluator), run_name="__main__")

stats["local_rms"] = (
    stats["local_sum_sq"] / max(stats["local_elements"], 1)
) ** 0.5
stats["attention_rms"] = (
    stats["attention_sum_sq"] / max(stats["attention_elements"], 1)
) ** 0.5
stats["local_spatial_residual_rms"] = (
    stats["local_spatial_residual_sum_sq"]
    / max(stats["local_elements"], 1)
) ** 0.5
stats["attention_spatial_residual_rms"] = (
    stats["attention_spatial_residual_sum_sq"]
    / max(stats["attention_elements"], 1)
) ** 0.5
stats["local_spatial_fraction"] = (
    stats["local_spatial_residual_rms"] / max(stats["local_rms"], 1e-12)
)
stats["attention_spatial_fraction"] = (
    stats["attention_spatial_residual_rms"]
    / max(stats["attention_rms"], 1e-12)
)
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / "pat_sf_branch_stats.json").write_text(
    json.dumps(stats, indent=2), encoding="utf-8"
)
print(json.dumps(stats, indent=2))
