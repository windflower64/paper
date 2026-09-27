#!/usr/bin/env python3
"""Assemble SAM2 contour audit overlays into compact review sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = {int(item["image_id"]): item for item in json.loads(args.records.read_text(encoding="utf-8"))}
    paths = sorted(args.overlay_dir.glob("*.webp"))
    per_sheet = args.columns * args.rows
    tile_width, tile_height, label_height = 320, 320, 28
    for page, offset in enumerate(range(0, len(paths), per_sheet), start=1):
        page_paths = paths[offset : offset + per_sheet]
        canvas = Image.new("RGB", (args.columns * tile_width, args.rows * (tile_height + label_height)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, path in enumerate(page_paths):
            image_id = int(path.stem)
            record = records[image_id]
            tile = Image.open(path).convert("RGB").resize((tile_width, tile_height))
            column, row = index % args.columns, index // args.columns
            x, y = column * tile_width, row * (tile_height + label_height)
            canvas.paste(tile, (x, y))
            reasons = ",".join(record["rejection_reasons"]) or "accepted"
            label = f"id={image_id} {reasons[:38]}"
            draw.text((x + 3, y + tile_height + 5), label, fill=(0, 0, 0))
        canvas.save(args.output_dir / f"contact_sheet_{page:02d}.jpg", quality=92)
    print(json.dumps({"overlays": len(paths), "sheets": (len(paths) + per_sheet - 1) // per_sheet}, indent=2))


if __name__ == "__main__":
    main()
