#!/usr/bin/env python3
"""CPU smoke test for the official PartialConv-only PAT_sf control."""

import sys
from pathlib import Path

import torch


repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

from src.core import YAMLConfig
from src.nn.backbone.partialnet_pat_sf import PartialConvOnly


cfg = YAMLConfig(str(repo / "experiments/phase_c/c_pat_pc_s32_r4.yml"))
cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
model = cfg.model.backbone.cpu().eval()

modules = [m for m in model.modules() if isinstance(m, PartialConvOnly)]
assert len(modules) == 3, f"expected three PartialConv modules, got {len(modules)}"
assert all(m.conv_channels == 32 for m in modules)

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
    grad = module.partial_conv3.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0, f"zero PartialConv gradient at module {index}"

print("C-PAT-PC1 CPU smoke passed")
print(f"modules={len(modules)} conv_channels={modules[0].conv_channels}")
print(f"backbone_spatial={spatial}")
