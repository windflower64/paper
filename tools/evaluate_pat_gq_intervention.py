#!/usr/bin/env python3
"""Evaluate causal interventions on a trained PartialGlobalQueryConv model."""

from __future__ import annotations

import argparse
import json
import math
import runpy
import sys
from pathlib import Path

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument(
    "--mode",
    choices=("learned", "uniform", "zero_local", "zero_global"),
    required=True,
)
args = parser.parse_args()
sys.path.insert(0, str(args.repo))

from src.nn.backbone.partialnet_pat_sf import PartialGlobalQueryConv


stats = {
    "mode": args.mode,
    "calls": 0,
    "attention_rows": 0,
    "normalized_entropy_sum": 0.0,
    "effective_token_ratio_sum": 0.0,
    "max_weight_sum": 0.0,
    "top10_mass_sum": 0.0,
    "relative_mad_from_uniform_sum": 0.0,
    "local_sum_sq": 0.0,
    "local_elements": 0,
    "global_sum_sq": 0.0,
    "global_elements": 0,
}


def intervened_forward(self, x):
    local_x, global_x = torch.split(
        x,
        [self.dim_conv3, self.dim_untouched],
        dim=1,
    )
    local_x = self.partial_conv3(local_x)
    global_x = self.norm(global_x)

    batch, channels, height, width = global_x.shape
    tokens = global_x.flatten(2).transpose(1, 2)
    token_count = tokens.shape[1]
    heads = self.attn.num_heads
    head_dim = self.attn.head_dim

    global_token = tokens.mean(dim=1, keepdim=True)
    q = self.attn.q(global_token).reshape(
        batch, 1, heads, head_dim
    ).transpose(1, 2)
    q = q * self.attn.scale
    kv = self.attn.kv(tokens).reshape(
        batch,
        token_count,
        2,
        heads,
        head_dim,
    ).permute(2, 0, 3, 1, 4)
    k, v = kv[0], kv[1]
    learned_weights = (q @ k.transpose(-2, -1)).softmax(dim=-1)

    detached = learned_weights.detach().float().flatten(0, 2)
    row_count = detached.shape[0]
    entropy = -(detached * detached.clamp_min(1e-12).log()).sum(dim=-1)
    normalized_entropy = entropy / math.log(token_count)
    effective_ratio = entropy.exp() / token_count
    top_count = max(1, int(math.ceil(token_count * 0.1)))
    top10_mass = detached.topk(top_count, dim=-1).values.sum(dim=-1)
    relative_mad = (detached - 1.0 / token_count).abs().mean(dim=-1) * token_count

    stats["calls"] += 1
    stats["attention_rows"] += row_count
    stats["normalized_entropy_sum"] += float(normalized_entropy.sum())
    stats["effective_token_ratio_sum"] += float(effective_ratio.sum())
    stats["max_weight_sum"] += float(detached.max(dim=-1).values.sum())
    stats["top10_mass_sum"] += float(top10_mass.sum())
    stats["relative_mad_from_uniform_sum"] += float(relative_mad.sum())

    if args.mode == "uniform":
        weights = torch.full_like(learned_weights, 1.0 / token_count)
    else:
        weights = learned_weights
    context = weights @ v
    context = context.transpose(1, 2).reshape(batch, 1, channels)
    context = self.attn.proj(context)
    context = context.transpose(1, 2).reshape(batch, channels, 1, 1)
    global_out = context.expand(batch, channels, height, width)

    stats["local_sum_sq"] += float(local_x.detach().float().square().sum())
    stats["local_elements"] += local_x.numel()
    stats["global_sum_sq"] += float(global_out.detach().float().square().sum())
    stats["global_elements"] += global_out.numel()

    if args.mode == "zero_local":
        local_x = torch.zeros_like(local_x)
    elif args.mode == "zero_global":
        global_out = torch.zeros_like(global_out)
    return torch.cat((local_x, global_out), dim=1)


PartialGlobalQueryConv.forward = intervened_forward

evaluator = args.repo / "tools" / "evaluate_custom_sizes_and_importance.py"
sys.argv = [
    str(evaluator),
    "--repo", str(args.repo),
    "--config", str(args.config),
    "--checkpoint", str(args.checkpoint),
    "--output-dir", str(args.output_dir),
    "--weight-source", "ema",
]
runpy.run_path(str(evaluator), run_name="__main__")

rows = max(stats["attention_rows"], 1)
stats["normalized_entropy_mean"] = stats["normalized_entropy_sum"] / rows
stats["effective_token_ratio_mean"] = stats["effective_token_ratio_sum"] / rows
stats["max_weight_mean"] = stats["max_weight_sum"] / rows
stats["top10_mass_mean"] = stats["top10_mass_sum"] / rows
stats["relative_mad_from_uniform_mean"] = (
    stats["relative_mad_from_uniform_sum"] / rows
)
stats["local_rms"] = (
    stats["local_sum_sq"] / max(stats["local_elements"], 1)
) ** 0.5
stats["global_rms"] = (
    stats["global_sum_sq"] / max(stats["global_elements"], 1)
) ** 0.5
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / "pat_gq_intervention_stats.json").write_text(
    json.dumps(stats, indent=2), encoding="utf-8"
)
print(json.dumps(stats, indent=2))
