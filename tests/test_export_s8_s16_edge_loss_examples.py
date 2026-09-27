import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "export_s8_s16_edge_loss_examples.py"
SPEC = importlib.util.spec_from_file_location("export_s8_s16_edge_loss_examples", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RawExampleHelpersTest(unittest.TestCase):
    def test_joint_normalization_uses_one_scale_for_both_maps(self):
        first = np.array([[0.0, 1.0]], dtype=np.float32)
        second = np.array([[2.0, 4.0]], dtype=np.float32)
        first_norm, second_norm = MODULE.joint_normalize(first, second, lower=0, upper=100)
        self.assertEqual(first_norm[0, 0], 0.0)
        self.assertEqual(second_norm[0, 1], 1.0)
        self.assertLess(first_norm.max(), second_norm.max())

    def test_feature_crop_maps_image_coordinates_to_native_grid(self):
        feature_map = np.arange(8 * 10, dtype=np.float32).reshape(8, 10)
        crop = MODULE.feature_crop(feature_map, (20, 16, 60, 48), image_width=100, image_height=80)
        self.assertEqual(crop.shape, (4, 4))
        self.assertEqual(crop[0, 0], feature_map[1, 2])
        self.assertEqual(crop[-1, -1], feature_map[4, 5])


if __name__ == "__main__":
    unittest.main()
