#!/usr/bin/env python3
"""CPU smoke test for the official PAT_ch same-location control."""

import sys
from pathlib import Path

import torch


repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

from src.core import YAMLConfig
from src.nn.backbone.hgnetv2 import PartialChannelAttentionConv


cfg = YAMLConfig(str(repo / "experiments/phase_c/c_pat_ch_s32_r4.yml"))
cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
model = cfg.model.backbone.cpu().eval()

modules = [m for m in model.modules() if isinstance(m, PartialChannelAttentionConv)]
assert len(modules) == 3, f"expected three PAT_ch modules, got {len(modules)}"
assert all(m.conv_channels == 32 and m.attention_channels == 96 for m in modules)

x = torch.randn(1, 3, 128, 160)
out = model(x)
assert len(out) >= 2
assert all(torch.isfinite(t).all() for t in out)
spatial = [tuple(t.shape[-2:]) for t in out]
assert all(
    spatial[index + 1][0] < spatial[index][0]
    and spatial[index + 1][1] < spatial[index][1]
    for index in range(len(spatial) - 1)
), spatial
loss = sum(t.float().square().mean() for t in out)
loss.backward()

for index, module in enumerate(modules):
    grads = (
        module.partial_conv.weight.grad,
        module.channel_attention.stat_fusion.weight.grad,
        module.attention_norm.weight.grad,
    )
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert all(g.abs().sum() > 0 for g in grads), f"zero PAT_ch gradient at {index}"

print("C-PAT-CH1 CPU smoke passed")
print(f"modules={len(modules)} split=32/96 backbone_spatial={spatial}")
