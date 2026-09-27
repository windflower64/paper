import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "export_sam_feature_comparison.py"
SPEC = importlib.util.spec_from_file_location("export_sam_feature_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SamFeatureComparisonTest(unittest.TestCase):
    def test_empty_mask_leaves_image_unchanged(self):
        image = np.full((12, 16, 3), 100, dtype=np.uint8)
        mask = np.zeros((12, 16), dtype=np.uint8)
        self.assertTrue(np.array_equal(MODULE.sam_mask_overlay(image, mask), image))

    def test_identical_feature_maps_leave_difference_background_unchanged(self):
        image = np.full((10, 14, 3), 80, dtype=np.uint8)
        values = np.ones((10, 14), dtype=np.float32)
        self.assertTrue(np.array_equal(MODULE.difference_overlay(image, values, values), image))

    def test_random_records_are_accepted_reproducible_and_distinct(self):
        records = [
            {"image_id": i // 2, "annotation_id": i, "accepted": i % 3 != 0}
            for i in range(30)
        ]
        first = MODULE.random_accepted_records(records, count=3, seed=7)
        second = MODULE.random_accepted_records(records, count=3, seed=7)
        self.assertEqual(first, second)
        self.assertTrue(all(row["accepted"] for row in first))
        self.assertEqual(len({row["image_id"] for row in first}), 3)

    def test_mask_filename_uses_recorded_image_key_not_annotation_id(self):
        record = {
            "image_id": 1847,
            "annotation_id": 1792,
            "mask_path": r"..\reports\masks\001847.png",
        }
        self.assertEqual(MODULE.mask_filename(record), "001847.png")


if __name__ == "__main__":
    unittest.main()
