#!/usr/bin/env python3
import sys


repo = sys.argv[1]
sys.path.insert(0, repo)
from src.core import YAMLConfig

for config_path in sys.argv[2:]:
    cfg = YAMLConfig(config_path)
    criterion = cfg.criterion
    print(
        config_path,
        "fgl=", criterion.fgl_edge_weight_mode,
        "ddf=", criterion.ddf_edge_reliability_mode,
        "reg_max=", criterion.reg_max,
    )
