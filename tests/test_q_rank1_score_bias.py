from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.zoo.dfine import dfine_decoder


QueryRankAdaptiveScoreBias = getattr(
    dfine_decoder, "QueryRankAdaptiveScoreBias", None
)


class QRankAvailabilityTest(unittest.TestCase):
    def test_query_rank_adaptive_score_bias_is_available(self):
        self.assertIsNotNone(QueryRankAdaptiveScoreBias)


@unittest.skipIf(
    QueryRankAdaptiveScoreBias is None,
    "Q-Rank1 implementation is intentionally absent during the RED step",
)
class QueryRankAdaptiveScoreBiasTest(unittest.TestCase):
    def _make_module(self):
        return QueryRankAdaptiveScoreBias(
            num_layers=2,
            num_queries=3,
            num_classes=1,
        )

    def test_zero_initialization_is_bit_exact_identity_with_denoising_prefix(self):
        module = self._make_module()
        scores = torch.tensor([[[9.0], [8.0], [3.0], [2.0], [1.0]]])

        calibrated = module(scores, layer_index=0)

        self.assertTrue(torch.equal(calibrated, scores))
        self.assertEqual(module.last_normal_query_count, 3)
        self.assertEqual(module.last_denoising_query_count, 2)

    def test_learned_bias_changes_only_normal_queries_by_their_rank_slot(self):
        module = self._make_module()
        scores = torch.tensor([[[9.0], [8.0], [3.0], [2.0], [1.0]]])
        with torch.no_grad():
            module.rank_bias[1, :, 0].copy_(torch.tensor([0.5, -0.25, 1.0]))

        calibrated = module(scores, layer_index=1)

        expected = torch.tensor([[[9.0], [8.0], [3.5], [1.75], [2.0]]])
        self.assertTrue(torch.equal(calibrated, expected))

    def test_zero_intervention_removes_a_trained_bias_exactly(self):
        module = self._make_module()
        scores = torch.tensor([[[3.0], [2.0], [1.0]]])
        with torch.no_grad():
            module.rank_bias[0, :, 0].copy_(torch.tensor([0.5, -0.25, 1.0]))
        module.intervention_mode = "zero"

        calibrated = module(scores, layer_index=0)

        self.assertTrue(torch.equal(calibrated, scores))

    def test_small_learned_bias_is_not_lost_when_scores_are_float16(self):
        module = self._make_module()
        scores = torch.full((1, 3, 1), -4.0, dtype=torch.float16)
        with torch.no_grad():
            module.rank_bias[0, :, 0].copy_(
                torch.tensor([0.0004, -0.0004, 0.0008])
            )

        calibrated = module(scores, layer_index=0)

        expected_delta = torch.tensor([0.0004, -0.0004, 0.0008])
        actual_delta = calibrated[0, :, 0].float() - scores[0, :, 0].float()
        self.assertEqual(calibrated.dtype, torch.float32)
        self.assertTrue(torch.allclose(actual_delta, expected_delta, atol=1e-7))

    def test_rejects_fewer_queries_than_the_rank_table(self):
        module = self._make_module()

        with self.assertRaisesRegex(ValueError, "fewer queries"):
            module(torch.zeros(1, 2, 1), layer_index=0)


class ExistingTopKOrderContractTest(unittest.TestCase):
    def test_default_selector_keeps_queries_in_descending_encoder_score_order(self):
        selector = SimpleNamespace(
            query_select_method="default",
            num_classes=1,
            training=False,
        )
        memory = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
        logits = torch.tensor([[[0.2], [5.0], [2.0], [-1.0]]])
        anchors = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)

        selected_memory, _, _, selected_indices = (
            dfine_decoder.DFINETransformer._select_topk(
                selector,
                memory,
                logits,
                anchors,
                topk=3,
                return_indices=True,
            )
        )

        self.assertEqual(selected_indices.tolist(), [[1, 2, 0]])
        self.assertEqual(selected_memory.flatten().tolist(), [1.0, 2.0, 0.0])


if __name__ == "__main__":
    unittest.main()
