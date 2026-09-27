#!/usr/bin/env python3
"""Audit whether target contours remain spatially distinguishable at each stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import math
import statistics


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--input-height", type=int, default=512)
    args = parser.parse_args()

    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in data["images"]}
    sizes = []
    for ann in data["annotations"]:
        image = images[int(ann["image_id"])]
        _, _, width, height = ann["bbox"]
        resized_width = float(width) * args.input_width / float(image["width"])
        resized_height = float(height) * args.input_height / float(image["height"])
        sizes.append((resized_width, resized_height))

    widths = [item[0] for item in sizes]
    heights = [item[1] for item in sizes]
    max_sides = [max(item) for item in sizes]
    report = {
        "annotations": len(sizes),
        "detector_input": [args.input_height, args.input_width],
        "resized_box_pixels": {},
        "feature_cells": {},
    }
    for name, values in (("width", widths), ("height", heights), ("max_side", max_sides)):
        report["resized_box_pixels"][name] = {
            key: float(value)
            for key, value in zip(
                ("q05", "q25", "median", "q75", "q95", "mean"),
                (*[quantile(values, q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)], statistics.fmean(values)),
            )
        }
    for stride in (4, 8, 16, 32):
        cell_widths = [value / stride for value in widths]
        cell_heights = [value / stride for value in heights]
        report["feature_cells"][f"S{stride}"] = {
            "median_width": statistics.median(cell_widths),
            "median_height": statistics.median(cell_heights),
            "q05_width": quantile(cell_widths, 0.05),
            "q05_height": quantile(cell_heights, 0.05),
            "fraction_both_sides_ge_3_cells": sum(w >= 3 and h >= 3 for w, h in zip(cell_widths, cell_heights)) / len(sizes),
            "fraction_both_sides_ge_5_cells": sum(w >= 5 and h >= 5 for w, h in zip(cell_widths, cell_heights)) / len(sizes),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
