#!/usr/bin/env python3
"""CPU smoke test for D3 mask projection and spatial controls."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from diagnose_true_contour_causality import mask_modes


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "mask.png"
        image = Image.new("L", (1920, 1080), 0)
        draw = ImageDraw.Draw(image)
        # A deliberately non-rectangular, aircraft-like silhouette.
        draw.polygon([(760, 525), (900, 500), (965, 455), (1000, 500), (1160, 525), (1010, 550), (970, 610), (930, 550)], fill=255)
        image.save(path)
        record = {"mask_path": str(path), "bbox_xyxy": [760, 455, 1160, 610]}
        masks = mask_modes(record, 512, 640, 64, 80, "cpu")

        def cos(left: str, right: str) -> float:
            return float(F.cosine_similarity(masks[left].flatten(), masks[right].flatten(), dim=0, eps=1e-12))

        report = {
            "shapes": {name: list(value.shape) for name, value in masks.items()},
            "finite": all(bool(torch.isfinite(value).all()) for value in masks.values()),
            "mass": {name: float(value.sum()) for name, value in masks.items()},
            "cosine": {
                "contour_box": cos("true_contour", "box_edge"),
                "contour_interior": cos("true_contour", "target_interior"),
                "contour_shifted": cos("true_contour", "shifted_contour"),
                "contour_background": cos("true_contour", "background"),
            },
        }
        if not report["finite"] or any(value <= 0 for value in report["mass"].values()):
            raise RuntimeError(report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
