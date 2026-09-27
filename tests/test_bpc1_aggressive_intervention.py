import sys
import unittest
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.nn.backbone.hgnetv2 import BoundaryPolyphaseCarrier  # noqa: E402


def make_branch():
    torch.manual_seed(7)
    branch = BoundaryPolyphaseCarrier(
        in_channels=4,
        out_channels=4,
        detail_channels=2,
        project_groups=1,
        max_scale=0.25,
        init_scale=0.05,
    ).eval()
    with torch.no_grad():
        branch.reduce.weight.normal_(0.0, 0.4)
        branch.locator[0].weight.normal_(0.0, 0.3)
        branch.locator[-1].weight.normal_(0.0, 0.3)
        branch.locator[-1].bias.fill_(-2.0)
        branch.project.weight.normal_(0.0, 0.25)
    return branch


class BPC1AggressiveInterventionTest(unittest.TestCase):
    def test_output_gain_multiplies_only_the_bpc_enhancement(self):
        branch = make_branch()
        x = torch.randn(1, 4, 8, 10)
        standard = torch.randn(1, 4, 4, 5)

        baseline = branch(x, standard)
        branch.intervention_output_gain = 2.0
        amplified = branch(x, standard)

        torch.testing.assert_close(amplified - standard, 2.0 * (baseline - standard))

    def test_gate_logit_bias_opens_the_learned_boundary_gate(self):
        branch = make_branch()
        x = torch.randn(1, 4, 8, 10)
        standard = torch.zeros(1, 4, 4, 5)

        branch(x, standard)
        baseline_gate = branch.last_gate.clone()
        baseline_logits = branch.last_boundary_logits.clone()
        branch.intervention_gate_logit_bias = 2.0
        branch(x, standard)

        expected = torch.sigmoid(baseline_logits + 2.0)
        torch.testing.assert_close(branch.last_gate, expected)
        self.assertGreater(branch.last_gate.mean(), baseline_gate.mean())

    def test_linear_mode_removes_the_final_tanh_compression(self):
        branch = make_branch()
        x = torch.randn(1, 4, 8, 10)
        standard = torch.randn(1, 4, 4, 5)
        branch.intervention_linear_residual = True

        output = branch(x, standard)

        with torch.no_grad():
            z = branch.reduce(x)
            logits = branch.locator(z)
            gate = torch.sigmoid(logits)
            phase_detail = branch.phase_residual(z)
            gate_phases = torch.nn.functional.pixel_unshuffle(gate, 2).unsqueeze(1)
            gated = (phase_detail * gate_phases).flatten(1, 2)
            residual = branch.project(gated)
            scale = branch.max_scale * torch.tanh(branch.scale_logit)
            expected = standard + scale * residual

        torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    unittest.main()
