import sys
import unittest
from pathlib import Path

import torch


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from s_scbv import SCBV_MODES, SemanticConditionedBoundaryVolume, gradient_l2


class SemanticConditionedBoundaryVolumeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260822)
        self.reader = SemanticConditionedBoundaryVolume(
            s8_channels=8,
            s16_channels=16,
            detail_channels=4,
            tangent_points=5,
            offset_bins=9,
        )
        self.s8 = torch.randn(2, 8, 16, 20)
        self.s16 = torch.randn(2, 16, 8, 10)
        self.boxes = torch.tensor(
            [
                [0.30, 0.30, 0.20, 0.16],
                [0.72, 0.55, 0.18, 0.22],
                [0.45, 0.70, 0.30, 0.20],
            ],
            dtype=torch.float32,
        )
        self.batch_indices = torch.tensor([0, 0, 1], dtype=torch.long)

    def test_zero_update_is_exact_identity(self):
        actual = self.reader(
            self.s8, self.s16, self.boxes, self.batch_indices, "zero_update"
        )
        self.assertTrue(torch.equal(actual, self.boxes))

    def test_zero_initialized_reader_is_near_identity(self):
        actual = self.reader(
            self.s8, self.s16, self.boxes, self.batch_indices, "aligned"
        )
        self.assertLess(float((actual - self.boxes).abs().max()), 1e-6)

    def test_all_modes_have_finite_box_outputs(self):
        for mode in SCBV_MODES:
            actual = self.reader(
                self.s8, self.s16, self.boxes, self.batch_indices, mode
            )
            self.assertEqual(actual.shape, self.boxes.shape)
            self.assertTrue(torch.isfinite(actual).all(), msg=mode)

    def test_candidate_volume_shape_and_backward(self):
        truth = self.boxes.clone()
        truth[:, 0] += 0.01
        logits = self.reader.candidate_logits(
            self.s8, self.s16, self.boxes, self.batch_indices, "aligned"
        )
        self.assertEqual(tuple(logits.shape), (3, 4, 9))
        loss, accuracy = self.reader.bin_loss_and_accuracy(logits, self.boxes, truth)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(accuracy))
        loss.backward()
        self.assertGreater(gradient_l2(self.reader.parameters()), 0.0)

    def test_shifted_detail_preserves_global_detail_energy(self):
        self.assertLess(self.reader.detail_energy_control_error(self.s8), 1e-6)


if __name__ == "__main__":
    unittest.main()

