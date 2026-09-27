import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location(
    "rgb_preservation", Path(__file__).resolve().parents[1] / "src/solver/rgb_preservation.py"
)
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)


class PreservationTests(unittest.TestCase):
    def test_tiny_box_area_and_empty_image(self):
        box = torch.tensor([[0.501, 0.499, 0.001, 0.002]])
        mask = rp.box_occupancy(box, 32, 40, "cpu")
        self.assertGreater(mask.sum().item(), 0)
        self.assertAlmostEqual(mask.sum().item() / (32 * 40), 0.000002, places=9)
        self.assertEqual(rp.box_occupancy(torch.empty(0, 4), 2, 2, "cpu").sum(), 0)

    def test_identical_and_positive_scale_have_zero_loss(self):
        x = torch.randn(2, 8, 4, 5)
        targets = [{"boxes": torch.tensor([[.5, .5, .2, .2]])}, {"boxes": torch.empty(0, 4)}]
        self.assertLess(rp.preservation_loss([x], [x * 3], targets).item(), 1e-10)

    def test_gradients_only_enter_student(self):
        x = torch.randn(1, 8, 4, 5, requires_grad=True)
        teacher = torch.randn_like(x, requires_grad=True)
        loss = rp.preservation_loss([x], [teacher], [{"boxes": torch.empty(0, 4)}])
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertIsNone(teacher.grad)


if __name__ == "__main__":
    unittest.main()
