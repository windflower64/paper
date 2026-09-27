#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from src.core import YAMLConfig
c=YAMLConfig(sys.argv[2])
for split, loader in (("train", c.train_dataloader), ("val", c.val_dataloader)):
    samples, targets=next(iter(loader))
    print(split, samples.shape, targets[0].keys(), targets[0]["boxes"][:5], targets[0].get("size"), targets[0].get("orig_size"))
