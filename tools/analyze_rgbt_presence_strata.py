"""Stratify frozen RGB/thermal detector confidence by annotation availability."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def thermal_positive(path: Path) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            _, x, y, width, height = map(float, fields[:5])
        except ValueError:
            continue
        if 0 <= x <= 1 and 0 <= y <= 1 and width > 0 and height > 0:
            return True
    return False


def summarize(values: list[float], threshold: float) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        f"fraction_ge_{threshold:g}": float((array >= threshold).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--visible-coco", type=Path, required=True)
    parser.add_argument("--thermal-labels", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    coco = json.loads(args.visible_coco.read_text(encoding="utf-8"))
    visible_positive = {
        int(ann["image_id"])
        for ann in coco["annotations"]
        if not ann.get("iscrowd", 0)
    }
    name_by_id = {int(item["id"]): item["file_name"] for item in coco["images"]}
    visible_scores = cache["visible_logits"].squeeze(-1).float().sigmoid().amax(1)
    thermal_scores = cache["thermal_logits"].squeeze(-1).float().sigmoid().amax(1)
    groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"visible": [], "thermal": []}
    )
    for row, image_id in enumerate(cache["image_ids"].tolist()):
        visible = int(image_id) in visible_positive
        thermal = thermal_positive(
            args.thermal_labels / f"{Path(name_by_id[int(image_id)]).stem}.txt"
        )
        group = (
            "both_positive"
            if visible and thermal
            else "visible_only"
            if visible
            else "thermal_only"
            if thermal
            else "both_negative"
        )
        groups[group]["visible"].append(float(visible_scores[row]))
        groups[group]["thermal"].append(float(thermal_scores[row]))

    report = {
        "schema": "rgbt_presence_strata_audit_v1",
        "purpose_zh": "区分红外无信息与可见光单标注协议惩罚红外独有目标。",
        "threshold": args.threshold,
        "groups": {
            group: {
                "count": len(values["visible"]),
                "visible_top1": summarize(values["visible"], args.threshold),
                "thermal_top1": summarize(values["thermal"], args.threshold),
            }
            for group, values in sorted(groups.items())
        },
        "interpretation_zh": (
            "若thermal_only组的红外高分比例远高于可见光，而both_negative组保持低分，"
            "则红外确有业务互补信息；仅按可见光框计算AP会把该信息计为误报。"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
