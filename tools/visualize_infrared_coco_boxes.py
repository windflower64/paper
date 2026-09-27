"""Create a compact infrared COCO box audit sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()

    coco = json.loads(args.annotation.read_text(encoding="utf-8"))
    annotations = {}
    for annotation in coco["annotations"]:
        annotations.setdefault(annotation["image_id"], []).append(annotation)
    positive = [image for image in coco["images"] if image["id"] in annotations]
    if not positive:
        raise RuntimeError("No positive infrared samples found")
    step = max(1, len(positive) // args.count)
    selected = positive[::step][: args.count]

    tile_width, tile_height = 320, 280
    columns = 4
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    for index, record in enumerate(selected):
        image = Image.open(args.image_root / record["file_name"]).convert("RGB")
        original_width, original_height = image.size
        image.thumbnail((tile_width, tile_height - 24), Image.Resampling.BILINEAR)
        scale_x = image.width / original_width
        scale_y = image.height / original_height
        draw = ImageDraw.Draw(image)
        for annotation in annotations[record["id"]]:
            x, y, width, height = annotation["bbox"]
            draw.rectangle(
                (
                    x * scale_x,
                    y * scale_y,
                    (x + width) * scale_x,
                    (y + height) * scale_y,
                ),
                outline=(255, 40, 40),
                width=3,
            )
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        tile.paste(image, ((tile_width - image.width) // 2, 0))
        ImageDraw.Draw(tile).text((4, tile_height - 20), record["file_name"], fill="black")
        sheet.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=95)
    print(args.output)


if __name__ == "__main__":
    main()

