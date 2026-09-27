#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("configs", nargs="+")
    a = p.parse_args()
    sys.path.insert(0, str(a.repo))
    from src.core import YAMLConfig
    from src.misc import stats
    for item in a.configs:
        cfg = YAMLConfig(str(a.repo / item))
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        params, result = stats(cfg)
        print(Path(item).name, params, result, flush=True)


if __name__ == "__main__":
    main()
