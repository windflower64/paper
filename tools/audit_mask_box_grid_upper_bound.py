#!/usr/bin/env python3
"""Grid-quantization upper bound for boxes recovered from binary feature masks."""

import argparse
import itertools
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "E:/two_paper/data/antiuav6k_common/annotations/instances_visible_common_test.json"
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--strides", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("E:/two_paper/reports/145_sqmi1_result/grid_upper_bound.json"),
    )
    return parser.parse_args()


def iou(left, right):
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-12)


def candidates(value, cells):
    scaled = value * cells
    return sorted({max(0, min(cells, math.floor(scaled))), max(0, min(cells, math.ceil(scaled)))})


def oracle_grid_iou(box, grid_width, grid_height):
    xs0 = candidates(box[0], grid_width)
    ys0 = candidates(box[1], grid_height)
    xs1 = candidates(box[2], grid_width)
    ys1 = candidates(box[3], grid_height)
    best = 0.0
    for x0, y0, x1, y1 in itertools.product(xs0, ys0, xs1, ys1):
        if x1 <= x0 or y1 <= y0:
            continue
        quantized = (x0 / grid_width, y0 / grid_height, x1 / grid_width, y1 / grid_height)
        best = max(best, iou(box, quantized))
    return best


def summarize(values):
    ordered = sorted(values)
    count = len(ordered)
    percentile = lambda p: ordered[min(count - 1, round((count - 1) * p))]
    return {
        "count": count,
        "mean_iou": sum(ordered) / count,
        "median_iou": percentile(0.5),
        "p10_iou": percentile(0.1),
        "fraction_iou_ge_050": sum(value >= 0.50 for value in ordered) / count,
        "fraction_iou_ge_075": sum(value >= 0.75 for value in ordered) / count,
    }


def main():
    args = parse_args()
    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    image_by_id = {image["id"]: image for image in coco["images"]}
    boxes = []
    for annotation in coco["annotations"]:
        image = image_by_id[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        boxes.append(
            (
                x / image["width"],
                y / image["height"],
                (x + width) / image["width"],
                (y + height) / image["height"],
            )
        )
    result = {"annotations": str(args.annotations), "objects": len(boxes), "strides": {}}
    for stride in args.strides:
        grid_width = args.width // stride
        grid_height = args.height // stride
        values = [oracle_grid_iou(box, grid_width, grid_height) for box in boxes]
        result["strides"][str(stride)] = {
            "grid": [grid_height, grid_width],
            **summarize(values),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
