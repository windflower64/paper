import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "export_random_multistage_overlays.py"
SPEC = importlib.util.spec_from_file_location("export_random_multistage_overlays", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RandomMultistageOverlayTest(unittest.TestCase):
    def test_zero_activation_keeps_original_background(self):
        image = np.full((4, 5, 3), 127, dtype=np.uint8)
        heat = np.zeros((4, 5), dtype=np.float32)
        overlay = MODULE.activation_overlay(image, heat)
        self.assertTrue(np.array_equal(overlay, image))

    def test_random_choice_is_reproducible_and_uses_distinct_images(self):
        records = [{"image_id": i, "object_index": 0} for i in range(20)]
        first = MODULE.random_distinct_records(records, count=5, seed=42)
        second = MODULE.random_distinct_records(records, count=5, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len({row["image_id"] for row in first}), 5)

    def test_smooth_heatmap_resize_creates_intermediate_values(self):
        heat = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        resized = MODULE.resize_heatmap_smooth(heat, width=32, height=24)
        self.assertEqual(resized.shape, (24, 32))
        self.assertTrue(np.any((resized > 0.0) & (resized < 1.0)))

    def test_full_context_keeps_resolution_and_marks_crop_boundary(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        context = MODULE.full_context_with_crop_box(image, crop=(20, 10, 60, 50))
        self.assertEqual(context.size, (100, 80))
        self.assertNotEqual(context.getpixel((20, 10)), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
