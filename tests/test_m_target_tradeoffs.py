import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location("tradeoffs", Path(__file__).resolve().parents[1] / "tools/diagnose_m_target_tradeoffs.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TargetTradeoffTests(unittest.TestCase):
    def test_good_box_low_score_is_not_geometry_failure(self):
        p = {"boxes": torch.tensor([[20., 20., 30., 30.], [0., 0., 10., 10.]]),
             "scores": torch.tensor([.9, .2])}
        result = mod.target_metrics(p, torch.tensor([0., 0., 10., 10.]))
        self.assertEqual(result["top1_iou"], 0)
        self.assertEqual(result["best_iou_300"], 1)
        self.assertEqual(result["correct_rank_75"], 2)

    def test_duplicate_background_and_empty_image(self):
        p = {"boxes": torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.], [20., 20., 30., 30.]]),
             "scores": torch.tensor([.9, .8, .7])}
        counts = mod.false_predictions(p, torch.tensor([[0., 0., 10., 10.]]), .5)
        self.assertEqual(counts, {"predictions": 3, "background": 1, "localization": 0, "duplicate": 1})
        empty = mod.false_predictions(p, torch.empty(0, 4), .5)
        self.assertEqual(empty["background"], 3)


if __name__ == "__main__":
    unittest.main()
