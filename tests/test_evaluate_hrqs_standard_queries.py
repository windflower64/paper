from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.eval_hrqs_intervention import configure_standard_queries


class _RecordLevelCount(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_level_count = None

    def forward(self, features):
        self.last_level_count = len(features)
        return features


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hrqs_enabled = True


class _RgbtModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _Decoder()
        self.encoder = _RecordLevelCount()
        self.thermal_encoder = _RecordLevelCount()


class StandardQueryInterventionTest(unittest.TestCase):
    def test_removes_s8_before_visible_and_thermal_encoders(self):
        model = _RgbtModel()
        hooks = configure_standard_queries(model)
        three_levels = [torch.zeros(1), torch.ones(1), torch.full((1,), 2.0)]

        model.encoder(three_levels)
        model.thermal_encoder(three_levels)

        self.assertFalse(model.decoder.hrqs_enabled)
        self.assertEqual(model.encoder.last_level_count, 2)
        self.assertEqual(model.thermal_encoder.last_level_count, 2)
        self.assertEqual(len(hooks), 2)
        for hook in hooks:
            hook.remove()


if __name__ == "__main__":
    unittest.main()

