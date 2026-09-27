import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.zoo.dfine.dfine_decoder import (
    SharedK1ThermalQueryCoupler,
    ThermalQueryCoupler,
)


def _copy_and_fold_single_token_coupler(source, target):
    target.query_norm.load_state_dict(source.query_norm.state_dict())
    target.token_norm.load_state_dict(source.token_norm.state_dict())
    target.channel_gate.load_state_dict(source.channel_gate.state_dict())
    target.safety_gate.load_state_dict(source.safety_gate.state_dict())
    target.context_proj.load_state_dict(source.context_proj.state_dict())
    target.output_norm.load_state_dict(source.output_norm.state_dict())

    hidden_dim = source.cross_attn.embed_dim
    value_weight = source.cross_attn.in_proj_weight[2 * hidden_dim :]
    value_bias = source.cross_attn.in_proj_bias[2 * hidden_dim :]
    output_weight = source.cross_attn.out_proj.weight
    output_bias = source.cross_attn.out_proj.bias
    with torch.no_grad():
        target.context_linear.weight.copy_(output_weight @ value_weight)
        target.context_linear.bias.copy_(output_weight @ value_bias + output_bias)
        target.layer_residual_scales[0].copy_(source.residual_scale)


class SharedK1ThermalQueryCouplerTest(unittest.TestCase):
    def test_folded_k1_context_matches_cross_attention(self):
        torch.manual_seed(7)
        hidden_dim = 128
        source = ThermalQueryCoupler(hidden_dim, num_heads=4)
        target = SharedK1ThermalQueryCoupler(hidden_dim, num_layers=1)
        _copy_and_fold_single_token_coupler(source, target)

        with torch.no_grad():
            source.residual_scale.fill_(0.37)
            target.layer_residual_scales[0].fill_(0.37)
        source.set_fusion_progress(0.6)
        target.set_fusion_progress(0.6)

        rgb_queries = torch.randn(2, 37, hidden_dim)
        tokens = torch.randn(2, 1, hidden_dim)
        quality = torch.rand(2, 1)
        uncertainty = torch.rand(2, 1)
        presence = torch.randn(2)

        source_output, source_gate = source(
            rgb_queries, tokens, quality, uncertainty, presence
        )
        target_output, target_gate = target.forward_layer(
            0, rgb_queries, tokens, quality, uncertainty, presence
        )
        torch.testing.assert_close(
            target_output, source_output, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            target_gate, source_gate, rtol=1e-5, atol=1e-6
        )

    def test_shared_coupler_is_smaller_than_three_independent_couplers(self):
        hidden_dim = 128
        independent = torch.nn.ModuleList(
            [ThermalQueryCoupler(hidden_dim, num_heads=4) for _ in range(3)]
        )
        shared = SharedK1ThermalQueryCoupler(hidden_dim, num_layers=3)
        independent_parameters = sum(p.numel() for p in independent.parameters())
        shared_parameters = sum(p.numel() for p in shared.parameters())
        self.assertEqual(independent_parameters, 944262)
        self.assertEqual(shared_parameters, 265220)
        self.assertLess(shared_parameters, 0.3 * independent_parameters)

    def test_shared_coupler_rejects_multiple_tokens(self):
        coupler = SharedK1ThermalQueryCoupler(hidden_dim=128, num_layers=3)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            coupler.forward_layer(
                0,
                torch.randn(1, 10, 128),
                torch.randn(1, 2, 128),
                torch.rand(1, 2),
                torch.rand(1, 2),
                torch.randn(1),
            )


if __name__ == "__main__":
    unittest.main()
