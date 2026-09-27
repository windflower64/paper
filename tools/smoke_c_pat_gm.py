"""CPU structural/gradient smoke test for the global-mean control."""

from pathlib import Path
import sys

import torch

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
torch.set_num_threads(1)

from src.core import YAMLConfig
from src.nn.backbone.hgnetv2 import HGNetv2
from src.nn.backbone.partialnet_pat_sf import PartialGlobalMeanConv

model = HGNetv2(
    name="B0",
    pretrained=False,
    freeze_at=-1,
    freeze_norm=False,
    pat_sf_stage=3,
    pat_sf_n_div=4,
    pat_sf_variant="global_mean",
)
modules = [m for m in model.modules() if isinstance(m, PartialGlobalMeanConv)]
assert len(modules) == 3, len(modules)
model.train()
x = torch.randn(2, 3, 128, 160, requires_grad=True)
prediction = model(x)
loss = sum(v.square().mean() for v in prediction)
loss.backward()
for module in modules:
    assert module.partial_conv3.weight.grad is not None
    assert module.attn.kv.weight.grad is not None
    assert module.attn.proj.weight.grad is not None

config = YAMLConfig(str(repo / "experiments/phase_c/c_pat_gm_s32_r4.yml"))
configured = [
    m for m in config.model.modules() if isinstance(m, PartialGlobalMeanConv)
]
assert len(configured) == 3, len(configured)
config.model.eval()
with torch.no_grad():
    output = config.model(torch.randn(1, 3, 512, 640))
assert all(
    torch.isfinite(value).all()
    for value in output.values()
    if torch.is_tensor(value)
)
print("global_mean_modules=3")
print("active_local_v_proj_gradients=True")
print("forward_backward=finite")
