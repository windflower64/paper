"""审计Anti-UAV RGB-T中红外候选监督是否足以支撑M3前提实验。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible-coco", type=Path, required=True)
    parser.add_argument("--infrared-images", type=Path, required=True)
    parser.add_argument("--infrared-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", required=True)
    return parser.parse_args()


def read_yolo_labels(path: Path) -> tuple[list[dict], list[str]]:
    boxes: list[dict] = []
    errors: list[str] = []
    if not path.is_file():
        return boxes, ["missing_label_file"]
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 5:
            errors.append(f"line_{line_number}:field_count_{len(fields)}")
            continue
        try:
            class_id = int(float(fields[0]))
            x, y, width, height = (float(value) for value in fields[1:5])
        except ValueError:
            errors.append(f"line_{line_number}:parse_error")
            continue
        valid = (
            0.0 <= x <= 1.0
            and 0.0 <= y <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
            and x - width / 2.0 >= -1e-6
            and y - height / 2.0 >= -1e-6
            and x + width / 2.0 <= 1.0 + 1e-6
            and y + height / 2.0 <= 1.0 + 1e-6
        )
        if not valid:
            errors.append(f"line_{line_number}:invalid_normalized_box")
            continue
        boxes.append(
            {
                "class_id": class_id,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "relative_area": width * height,
            }
        )
    return boxes, errors


def describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def quantile(q: float) -> float:
        index = round((len(ordered) - 1) * q)
        return float(ordered[index])

    return {
        "count": len(ordered),
        "min": float(ordered[0]),
        "p25": quantile(0.25),
        "p50": quantile(0.50),
        "p75": quantile(0.75),
        "p95": quantile(0.95),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / len(ordered)),
    }


def main() -> None:
    args = parse_args()
    coco = json.loads(args.visible_coco.read_text(encoding="utf-8"))
    images = coco["images"]
    annotations = coco["annotations"]
    annotations_by_image: Counter[int] = Counter()
    for annotation in annotations:
        if float(annotation.get("area", 0.0)) > 0.0:
            annotations_by_image[int(annotation["image_id"])] += 1

    rows = []
    all_ir_boxes: list[dict] = []
    error_examples = []
    missing_ir_images = []
    for image in images:
        file_name = str(image["file_name"])
        stem = Path(file_name).stem
        label_path = args.infrared_labels / f"{stem}.txt"
        boxes, errors = read_yolo_labels(label_path)
        all_ir_boxes.extend(boxes)
        if errors and len(error_examples) < 30:
            error_examples.append({"file": label_path.name, "errors": errors})
        ir_image = args.infrared_images / file_name
        if not ir_image.is_file() and len(missing_ir_images) < 30:
            missing_ir_images.append(file_name)
        rows.append(
            {
                "file_name": file_name,
                "visible_positive": annotations_by_image[int(image["id"])] > 0,
                "visible_objects": annotations_by_image[int(image["id"])],
                "infrared_positive": len(boxes) > 0,
                "infrared_objects": len(boxes),
                "label_errors": len(errors),
                "infrared_image_exists": ir_image.is_file(),
            }
        )

    visible_positive = sum(row["visible_positive"] for row in rows)
    infrared_positive = sum(row["infrared_positive"] for row in rows)
    both_positive = sum(
        row["visible_positive"] and row["infrared_positive"] for row in rows
    )
    both_negative = sum(
        not row["visible_positive"] and not row["infrared_positive"] for row in rows
    )
    visible_only = sum(
        row["visible_positive"] and not row["infrared_positive"] for row in rows
    )
    infrared_only = sum(
        not row["visible_positive"] and row["infrared_positive"] for row in rows
    )
    class_counts = Counter(box["class_id"] for box in all_ir_boxes)
    relative_areas = [box["relative_area"] for box in all_ir_boxes]
    widths = [box["width"] for box in all_ir_boxes]
    heights = [box["height"] for box in all_ir_boxes]
    invalid_label_files = sum(row["label_errors"] > 0 for row in rows)
    agreement = both_positive + both_negative

    report = {
        "status": "PASS" if not missing_ir_images and invalid_label_files == 0 else "WARN",
        "split": args.split,
        "visible_coco": str(args.visible_coco),
        "infrared_images": str(args.infrared_images),
        "infrared_labels": str(args.infrared_labels),
        "images": len(rows),
        "visible_annotations": len(annotations),
        "infrared_valid_boxes": len(all_ir_boxes),
        "missing_infrared_images_count": sum(
            not row["infrared_image_exists"] for row in rows
        ),
        "invalid_label_files_count": invalid_label_files,
        "presence": {
            "visible_positive": visible_positive,
            "infrared_positive": infrared_positive,
            "both_positive": both_positive,
            "both_negative": both_negative,
            "visible_only": visible_only,
            "infrared_only": infrared_only,
            "agreement_fraction": agreement / len(rows) if rows else 0.0,
        },
        "infrared_class_counts": {str(key): value for key, value in class_counts.items()},
        "infrared_box_width": describe(widths),
        "infrared_box_height": describe(heights),
        "infrared_box_relative_area": describe(relative_areas),
        "error_examples": error_examples,
        "missing_infrared_image_examples": missing_ir_images,
        "presence_disagreement_examples": [
            row
            for row in rows
            if row["visible_positive"] != row["infrared_positive"]
        ][:30],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
