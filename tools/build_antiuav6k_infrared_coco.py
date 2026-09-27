"""Convert local Anti-UAV-6K infrared YOLO labels to correct COCO boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def valid_yolo_rows(path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if len(fields) < 5:
            continue
        try:
            cls, cx, cy, width, height = map(float, fields[:5])
        except ValueError:
            continue
        if width > 0 and height > 0:
            rows.append((int(cls), cx, cy, width, height))
    return rows


def convert_split(dataset_root, split):
    image_root = dataset_root / split / "infrared" / "images"
    label_root = dataset_root / split / "infrared" / "labels"
    image_paths = sorted(
        path
        for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    images, annotations = [], []
    annotation_id = 1
    for image_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        for _, cx, cy, box_width, box_height in valid_yolo_rows(
            label_root / f"{image_path.stem}.txt"
        ):
            x0 = max(0.0, min(float(width), (cx - box_width / 2.0) * width))
            y0 = max(0.0, min(float(height), (cy - box_height / 2.0) * height))
            x1 = max(0.0, min(float(width), (cx + box_width / 2.0) * width))
            y1 = max(0.0, min(float(height), (cy + box_height / 2.0) * height))
            pixel_width = x1 - x0
            pixel_height = y1 - y0
            if pixel_width <= 0 or pixel_height <= 0:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    # D-FINE uses category ids directly when
                    # remap_mscoco_category=False.  The established visible
                    # Anti-UAV protocol therefore uses the sole class id 0.
                    "category_id": 0,
                    "bbox": [x0, y0, pixel_width, pixel_height],
                    "area": pixel_width * pixel_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "info": {
            "description": "Anti-UAV-6K infrared boxes converted from local YOLO labels",
            "split": split,
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "UAV", "supercategory": "object"}],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"F:\data\Anti-UAV\Anti_UAV_6K"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:\two_paper\data\antiuav6k_ir\annotations"),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        coco = convert_split(args.dataset_root, split)
        output = args.output_root / f"instances_infrared_{split}.json"
        output.write_text(
            json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"{split}: images={len(coco['images'])} "
            f"annotations={len(coco['annotations'])} output={output}"
        )


if __name__ == "__main__":
    main()
