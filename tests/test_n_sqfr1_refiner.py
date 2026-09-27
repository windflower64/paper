from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.zoo.dfine import dfine_decoder


SparseQueryHighResolutionRefiner = getattr(
    dfine_decoder, "SparseQueryHighResolutionRefiner", None
)


class SQFRAvailabilityTest(unittest.TestCase):
    def test_sparse_query_refiner_is_available(self):
        self.assertIsNotNone(SparseQueryHighResolutionRefiner)


@unittest.skipIf(
    SparseQueryHighResolutionRefiner is None,
    "SQFR implementation is intentionally absent during the RED step",
)
class SparseQueryHighResolutionRefinerTest(unittest.TestCase):
    def _make_refiner(self):
        torch.manual_seed(7)
        return SparseQueryHighResolutionRefiner(
            in_channels=4,
            query_dim=8,
            hidden_dim=8,
            num_heads=2,
            roi_size=5,
            topk=2,
            context_scale=1.5,
            max_logit_delta=0.25,
        )

    @staticmethod
    def _inputs():
        feature = torch.arange(4 * 8 * 8, dtype=torch.float32).reshape(1, 4, 8, 8)
        feature = feature / feature.max()
        boxes = torch.tensor(
            [
                [
                    [0.25, 0.25, 0.20, 0.20],
                    [0.70, 0.30, 0.25, 0.20],
                    [0.35, 0.70, 0.20, 0.25],
                    [0.65, 0.70, 0.30, 0.20],
                    [0.50, 0.50, 0.40, 0.40],
                ]
            ],
            dtype=torch.float32,
        )
        logits = torch.tensor([[[5.0], [4.0], [3.0], [2.0], [1.0]]])
        queries = torch.randn(1, 5, 8)
        return feature, boxes, logits, queries

    def test_zero_initialization_is_bit_exact_identity(self):
        refiner = self._make_refiner()
        refiner.eval()
        feature, boxes, logits, queries = self._inputs()

        refined = refiner(feature, boxes, logits, queries)

        self.assertTrue(torch.equal(refined, boxes))
        self.assertEqual(tuple(refined.shape), (1, 5, 4))
        self.assertEqual(refiner.last_selected_indices.tolist(), [[0, 1]])

    def test_amp_residual_is_merged_in_the_coarse_box_dtype(self):
        refiner = self._make_refiner()
        feature, boxes, logits, queries = self._inputs()

        with torch.autocast("cpu", dtype=torch.bfloat16):
            refined = refiner(feature, boxes, logits, queries)

        self.assertEqual(refined.dtype, boxes.dtype)
        self.assertTrue(torch.equal(refined, boxes))

    def test_opened_refiner_changes_only_selected_queries(self):
        refiner = self._make_refiner()
        refiner.eval()
        feature, boxes, logits, queries = self._inputs()
        torch.nn.init.normal_(refiner.delta_head[-1].weight, mean=0.0, std=0.1)

        refined = refiner(feature, boxes, logits, queries)

        self.assertFalse(torch.equal(refined[:, :2], boxes[:, :2]))
        self.assertTrue(torch.equal(refined[:, 2:], boxes[:, 2:]))

    def test_training_uses_all_queries_so_matched_boxes_can_supervise_refiner(self):
        refiner = self._make_refiner()
        refiner.train()
        feature, boxes, logits, queries = self._inputs()

        refined = refiner(feature, boxes, logits, queries)

        self.assertEqual(
            refiner.last_selected_indices.tolist(), [[0, 1, 2, 3, 4]]
        )
        self.assertTrue(torch.equal(refined, boxes))

    def test_zero_feature_intervention_removes_entire_correction(self):
        refiner = self._make_refiner()
        feature, boxes, logits, queries = self._inputs()
        torch.nn.init.normal_(refiner.delta_head[-1].weight, mean=0.0, std=0.1)
        refiner.feature_mode = "zero"

        refined = refiner(feature, boxes, logits, queries)

        self.assertTrue(torch.equal(refined, boxes))

    def test_shifted_s8_changes_the_localization_correction(self):
        refiner = self._make_refiner()
        feature, boxes, logits, queries = self._inputs()
        torch.nn.init.normal_(refiner.delta_head[-1].weight, mean=0.0, std=0.1)

        full = refiner(feature, boxes, logits, queries)
        refiner.feature_mode = "shifted"
        shifted = refiner(feature, boxes, logits, queries)

        self.assertFalse(torch.equal(full[:, :2], shifted[:, :2]))

    def test_refinement_loss_reaches_s8_but_not_query_or_score_inputs(self):
        refiner = self._make_refiner()
        feature, boxes, logits, queries = self._inputs()
        feature.requires_grad_(True)
        boxes.requires_grad_(True)
        logits.requires_grad_(True)
        queries.requires_grad_(True)
        torch.nn.init.normal_(refiner.delta_head[-1].weight, mean=0.0, std=0.1)

        refiner(feature, boxes, logits, queries).sum().backward()

        self.assertIsNotNone(feature.grad)
        self.assertGreater(float(feature.grad.abs().sum()), 0.0)
        self.assertIsNotNone(boxes.grad)
        self.assertGreater(float(boxes.grad.abs().sum()), 0.0)
        self.assertIsNone(logits.grad)
        self.assertIsNone(queries.grad)


if __name__ == "__main__":
    unittest.main()
