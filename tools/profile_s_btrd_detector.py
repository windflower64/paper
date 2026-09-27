#!/usr/bin/env python3
"""Profile A00 and S-BTRD1 at the actual 512x640 experiment resolution."""

from pathlib import Path
import sys

import torch.nn as nn
from calflops import calculate_flops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


class DetectorForProfile(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = config.model.deploy()

    def forward(self, images):
        return self.model(images)


def main():
    for relative_path in (
        "experiments/phase_s/visible_60e_base.yml",
        "experiments/phase_s/s_btrd1_act_trans_s8_s16.yml",
    ):
        config = YAMLConfig(str(ROOT / relative_path))
        model = DetectorForProfile(config).eval()
        flops, macs, parameters = calculate_flops(
            model=model,
            input_shape=(1, 3, 512, 640),
            output_as_string=False,
            print_results=False,
        )
        print(
            {
                "config": relative_path,
                "flops": flops,
                "macs": macs,
                "parameters_from_profiler": parameters,
                "parameters_exact": sum(p.numel() for p in model.parameters()),
            }
        )


if __name__ == "__main__":
    main()

