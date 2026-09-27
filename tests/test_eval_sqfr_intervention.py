import unittest
from pathlib import Path
import sys

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.eval_sqfr_intervention import configure_sqfr_mode


class _SQFR(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_mode = "full"


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sqfr_refiner = _SQFR()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _Decoder()


class ConfigureSQFRModeTest(unittest.TestCase):
    def test_feature_interventions_keep_trained_branch_active(self):
        model = _Model()

        for mode in ("full", "zero", "shifted"):
            refiner = configure_sqfr_mode(model, mode)
            self.assertIs(refiner, model.decoder.sqfr_refiner)
            self.assertEqual(refiner.feature_mode, mode)

    def test_disabled_physically_removes_branch(self):
        model = _Model()

        refiner = configure_sqfr_mode(model, "disabled")

        self.assertIsNotNone(refiner)
        self.assertIsNone(model.decoder.sqfr_refiner)

    def test_model_without_trained_branch_is_rejected(self):
        model = _Model()
        model.decoder.sqfr_refiner = None

        with self.assertRaisesRegex(RuntimeError, "trained SQFR1"):
            configure_sqfr_mode(model, "full")


if __name__ == "__main__":
    unittest.main()
