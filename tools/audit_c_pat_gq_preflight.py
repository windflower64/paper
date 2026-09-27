"""Static optimizer and operation audit for C-PAT-GQ1; never trains."""

import sys
from collections import Counter
from pathlib import Path


repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

from src.core import YAMLConfig
from src.nn.backbone.partialnet_pat_sf import PartialGlobalQueryConv


config = YAMLConfig(str(repo / "experiments/phase_c/c_pat_gq_s32_r4.yml"))
model = config.model
optimizer = config.optimizer

trainable = {
    id(parameter): name
    for name, parameter in model.named_parameters()
    if parameter.requires_grad
}
assigned = [
    id(parameter)
    for group in optimizer.param_groups
    for parameter in group["params"]
]
counts = Counter(assigned)
missing = [name for identity, name in trainable.items() if identity not in counts]
duplicated = [
    trainable[identity]
    for identity, count in counts.items()
    if identity in trainable and count != 1
]
foreign = [identity for identity in counts if identity not in trainable]
assert not missing and not duplicated and not foreign, (
    missing,
    duplicated,
    len(foreign),
)

modules = [m for m in model.modules() if isinstance(m, PartialGlobalQueryConv)]
assert len(modules) == 3, len(modules)
gq_parameters = sum(
    parameter.numel()
    for module in modules
    for parameter in module.parameters()
)

# Exact MAC count for the replaced Stage4 S32 mixer at 512x640.
tokens = 16 * 20
channels = 128
local_channels = 32
global_channels = 96
baseline_per_module = tokens * channels * 5 * 5
local_conv = tokens * local_channels * local_channels * 3 * 3
query_projection = global_channels * global_channels
key_value_projection = tokens * global_channels * global_channels * 2
context_projection = global_channels * global_channels
attention_matmuls = 2 * tokens * global_channels
gq_per_module = (
    local_conv
    + query_projection
    + key_value_projection
    + context_projection
    + attention_matmuls
)
delta_total = len(modules) * (gq_per_module - baseline_per_module)

print("optimizer_assigned_once=True")
print(f"modules={len(modules)}")
print(f"gq_parameters={gq_parameters}")
print(f"baseline_macs_per_module={baseline_per_module}")
print(f"gq_macs_per_module={gq_per_module}")
print(f"replacement_macs_delta_total={delta_total}")
print(f"attention_score_elements_per_module_per_sample={4 * tokens}")
