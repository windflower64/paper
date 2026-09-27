import copy
import ast
from pathlib import Path
import unittest
import torch
from src.zoo.dfine.dfine import SpatiallyDecoupledThermalTokenConditioner


class FinalResidualScaleTest(unittest.TestCase):
    def test_default_matches_archived_original_module(self):
        import src.zoo.dfine.dfine as module
        path = Path(__file__).resolve().parents[2] / 'outputs/C_PLUS_M_SD22_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0/artifacts/src/zoo/dfine/dfine.py'
        source = path.read_text(encoding='utf-8')
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
                    and n.name == 'SpatiallyDecoupledThermalTokenConditioner')
        namespace = dict(vars(module))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        args = dict(hidden_dim=8, bottleneck_dim=4, num_heads=2, num_tokens=2, thermal_contrastive=True)
        torch.manual_seed(4)
        old = namespace[node.name](**args)
        for g in old.channel_generators:
            torch.nn.init.normal_(g[-1].weight, std=0.1)
        new = SpatiallyDecoupledThermalTokenConditioner(**args)
        new.load_state_dict(old.state_dict(), strict=True)
        x = [torch.randn(2, 8, 4, 5), torch.randn(2, 8, 2, 3)]
        t = [torch.randn_like(v) for v in x]
        for a, b in zip(old(x, t), new(x, t)):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_half_scales_final_output_and_fusion_gradient_in_train_and_eval(self):
        torch.manual_seed(4)
        full = SpatiallyDecoupledThermalTokenConditioner(
            hidden_dim=8, bottleneck_dim=4, num_heads=2, num_tokens=2,
            thermal_contrastive=True,
        )
        for generator in full.channel_generators:
            torch.nn.init.normal_(generator[-1].weight, std=0.1)
        half = copy.deepcopy(full)
        half.final_residual_scale = 0.5
        visible = [torch.randn(2, 8, 4, 5), torch.randn(2, 8, 2, 3)]
        thermal = [torch.randn_like(x) for x in visible]
        for training in (True, False):
            full.train(training); half.train(training)
            full.zero_grad(); half.zero_grad()
            a = full(visible, thermal)
            b = half(visible, thermal)
            self.assertGreater(float((a[0] - visible[0]).abs().max()), 1e-5)
            for x, y, z in zip(visible, a, b):
                torch.testing.assert_close(z, x + 0.5 * (y - x), atol=0, rtol=0)
            sum(x.sum() for x in a).backward()
            sum(x.sum() for x in b).backward()
            for p, q in zip(full.parameters(), half.parameters()):
                if p.grad is not None:
                    torch.testing.assert_close(q.grad, 0.5*p.grad, atol=1e-6, rtol=1e-4)
            torch.testing.assert_close(half.last_rms_ratio_by_level,
                                       full.last_rms_ratio_by_level*0.5, atol=1e-6, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
