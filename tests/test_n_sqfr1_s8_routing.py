from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.zoo.dfine.dfine import DFINE


class _ThreeLevelBackbone(nn.Module):
    def forward(self, _images):
        return [
            torch.full((1, 4, 8, 8), 8.0),
            torch.full((1, 8, 4, 4), 16.0),
            torch.full((1, 16, 2, 2), 32.0),
        ]


class _TwoLevelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = [8, 16]
        self.last_features = None

    def forward(self, features):
        self.last_features = features
        return features


class _SQFRDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hrqs_enabled = False
        self.sqfr_enabled = True
        self.last_s8 = None

    def forward(self, features, targets=None, hrqs_source=None, **_kwargs):
        self.last_s8 = hrqs_source
        return {
            "pred_logits": features[0].new_zeros((1, 5, 1)),
            "pred_boxes": features[0].new_zeros((1, 5, 4)),
        }


class SQFRS8RoutingTest(unittest.TestCase):
    def test_sqfr_receives_s8_while_encoder_keeps_only_s16_and_s32(self):
        backbone = _ThreeLevelBackbone()
        encoder = _TwoLevelEncoder()
        decoder = _SQFRDecoder()
        model = DFINE(backbone=backbone, encoder=encoder, decoder=decoder)

        model(torch.zeros(1, 3, 64, 64))

        self.assertEqual(len(encoder.last_features), 2)
        self.assertEqual(float(encoder.last_features[0].mean()), 16.0)
        self.assertEqual(float(encoder.last_features[1].mean()), 32.0)
        self.assertIsNotNone(decoder.last_s8)
        self.assertEqual(float(decoder.last_s8.mean()), 8.0)


if __name__ == "__main__":
    unittest.main()
