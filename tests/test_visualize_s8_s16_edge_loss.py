import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "visualize_s8_s16_edge_loss.py"
SPEC = importlib.util.spec_from_file_location("visualize_s8_s16_edge_loss", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VisualizationHelpersTest(unittest.TestCase):
    def test_robust_unit_interval_handles_outlier_and_constant_map(self):
        values = np.array([[0.0, 1.0], [2.0, 100.0]], dtype=np.float32)
        normalized = MODULE.robust_unit_interval(values, lower=0, upper=75)
        self.assertEqual(normalized.min(), 0.0)
        self.assertEqual(normalized.max(), 1.0)
        self.assertEqual(normalized[1, 0], 1.0)

        constant = MODULE.robust_unit_interval(np.ones((2, 2), dtype=np.float32))
        self.assertTrue(np.array_equal(constant, np.zeros((2, 2), dtype=np.float32)))

    def test_edge_energy_returns_one_spatial_map_and_highlights_step(self):
        feature = torch.zeros(1, 2, 5, 7)
        feature[:, :, :, 3:] = 4.0
        energy = MODULE.edge_energy(feature)
        self.assertEqual(energy.shape, (5, 7))
        self.assertTrue(np.isfinite(energy).all())
        self.assertGreater(energy[:, 2:4].mean(), energy[:, :2].mean())

    def test_square_crop_stays_in_image_and_contains_box(self):
        crop = MODULE.square_crop((90, 40, 100, 50), image_width=120, image_height=80, scale=4.0)
        left, top, right, bottom = crop
        self.assertTrue(0 <= left < right <= 120)
        self.assertTrue(0 <= top < bottom <= 80)
        self.assertTrue(left <= 90 and top <= 40 and right >= 100 and bottom >= 50)
        self.assertEqual(right - left, bottom - top)


if __name__ == "__main__":
    unittest.main()
