"""Focused tests for the standalone grouping pilot."""
import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.pilot_s_sgc1 import selection, contrast
from src.zoo.dfine.sam_group_contrast import (
    group_loss,
    contrast as production_contrast,
    supervision_scale,
)
from src.zoo.dfine.dfine import DFINE


class GroupContrastTest(unittest.TestCase):
    def test_supervision_scale_keeps_then_linearly_retires_teacher(self):
        expected = {
            0: 1.0,
            9: 1.0,
            10: 0.8,
            11: 0.6,
            12: 0.4,
            13: 0.2,
            14: 0.0,
            19: 0.0,
        }
        for epoch, value in expected.items():
            self.assertAlmostEqual(supervision_scale(epoch, 9, 14), value)

    def test_supervision_scale_can_be_disabled_for_sgc1_compatibility(self):
        for epoch in (0, 9, 19, 100):
            self.assertEqual(supervision_scale(epoch, -1, -1), 1.0)

    def test_supervision_scale_rejects_partial_or_reversed_schedule(self):
        for start, end in ((-1, 14), (9, -1), (14, 9), (9, 9)):
            with self.assertRaises(ValueError):
                supervision_scale(0, start, end)

    def test_dfine_epoch_hook_drives_sgc_supervision_scale(self):
        model = object.__new__(DFINE)
        torch.nn.Module.__init__(model)
        model.backbone = torch.nn.Identity()
        model.encoder = torch.nn.Identity()
        model.decoder = torch.nn.Identity()
        model.thermal_backbone = None
        model.sgc_decay_start = 9
        model.sgc_decay_end = 14
        model.set_training_epoch(12)
        self.assertEqual(model.sgc_training_epoch, 12)
        self.assertAlmostEqual(model._sgc_supervision_scale(), 0.4)

    def test_production_matches_pilot(self):
        x = torch.randn(4, 8, 8, requires_grad=True)
        ids = (torch.arange(8), torch.arange(8, 32))
        a, _ = contrast(x, ids); b, _ = production_contrast(x, ids)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(a, x)[0], torch.autograd.grad(b, x)[0])

    def test_group_loss_skips_rejected(self):
        x = torch.randn(1, 4, 8, 8, requires_grad=True)
        t = dict(boxes=torch.zeros(0, 4), masks=torch.zeros(0, 64, 64), sam_quality=torch.tensor(0.))
        value = group_loss(x, [t], 'sam')
        value.backward(); self.assertEqual(x.grad.abs().sum().item(), 0.)

    def test_separation_is_rewarded(self):
        ids = (torch.tensor([0, 1]), torch.tensor([2, 3, 4, 5]))
        separated = torch.tensor([[[1., 1., 0., 0., 0., 0.]],
                                  [[0., 0., 1., 1., 1., 1.]]], requires_grad=True)
        collapsed = torch.ones_like(separated, requires_grad=True)
        a, _ = contrast(separated, ids); b, _ = contrast(collapsed, ids)
        self.assertLess(a.item(), b.item())
        a.backward(); self.assertTrue(torch.isfinite(separated.grad).all())

    def test_insufficient_foreground_skipped(self):
        x = torch.randn(4, 4, 4, requires_grad=True)
        loss, stats = contrast(x, (torch.tensor([0]), torch.tensor([1, 2, 3, 4])))
        self.assertIsNone(stats); self.assertEqual(loss.item(), 0.)
        loss.backward(); self.assertEqual(x.grad.abs().sum().item(), 0.)

    def test_rejected_and_empty_skipped(self):
        for boxes in (torch.zeros(0, 4), torch.tensor([[.5, .5, .5, .5]])):
            t = dict(boxes=boxes, masks=torch.zeros(1, 32, 32), sam_quality=torch.tensor(0.))
            self.assertIsNone(selection(t, (4, 4), 'sam'))

    def test_selection_disjoint(self):
        mask = torch.zeros(1, 64, 64); mask[:, 24:40, 24:40] = 1
        t = dict(boxes=torch.tensor([[.5, .5, .25, .25]]), masks=mask, sam_quality=torch.tensor(1.))
        for source in ('sam', 'box'):
            p, n = selection(t, (8, 8), source)
            self.assertGreaterEqual(len(p), 2); self.assertGreaterEqual(len(n), 4)
            self.assertFalse(set(p.tolist()) & set(n.tolist()))


if __name__ == '__main__':
    unittest.main()
