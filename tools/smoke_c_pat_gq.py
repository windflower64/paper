"""CPU structural and gradient smoke test for C-PAT-GQ1."""

from pathlib import Path
import sys

import torch


repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
torch.set_num_threads(1)

from src.core import YAMLConfig
from src.nn.backbone.hgnetv2 import HGNetv2
from src.nn.backbone.partialnet_pat_sf import PartialGlobalQueryConv


model = HGNetv2(
    name="B0",
    pretrained=False,
    freeze_at=-1,
    freeze_norm=False,
    pat_sf_stage=3,
    pat_sf_n_div=4,
    pat_sf_variant="global_query",
)
modules = [m for m in model.modules() if isinstance(m, PartialGlobalQueryConv)]
assert len(modules) == 3, len(modules)
assert all(m.conv_channels == 32 for m in modules)
assert all(m.attention_channels == 96 for m in modules)

model.train()
x = torch.randn(2, 3, 128, 160, requires_grad=True)
outputs = model(x)
sum(y.square().mean() for y in outputs).backward()
for module in modules:
    assert module.partial_conv3.weight.grad is not None
    assert module.attn.q.weight.grad is not None
    assert module.attn.kv.weight.grad is not None
    assert module.attn.proj.weight.grad is not None

config = YAMLConfig(str(repo / "experiments/phase_c/c_pat_gq_s32_r4.yml"))
configured = [
    m for m in config.model.modules() if isinstance(m, PartialGlobalQueryConv)
]
assert len(configured) == 3, len(configured)
config.model.eval()
with torch.no_grad():
    prediction = config.model(torch.randn(1, 3, 512, 640))

def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True

assert finite(prediction)
print("gq_modules=3")
print("split=32_local+96_global_query")
print("backward=finite")
print("configured_forward=finite")
