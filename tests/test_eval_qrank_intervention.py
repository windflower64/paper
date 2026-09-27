import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "eval_qrank_intervention.py"
SPEC = importlib.util.spec_from_file_location("eval_qrank_intervention", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvalQRankInterventionTest(unittest.TestCase):
    def test_zero_mode_keeps_module_and_sets_zero_intervention(self):
        ranker = SimpleNamespace(intervention_mode="learned")
        model = SimpleNamespace(decoder=SimpleNamespace(qrank_score_bias=ranker))

        returned = MODULE.configure_qrank_mode(model, "zero")

        self.assertIs(returned, ranker)
        self.assertIs(model.decoder.qrank_score_bias, ranker)
        self.assertEqual(ranker.intervention_mode, "zero")

    def test_disabled_mode_physically_removes_ranker(self):
        ranker = SimpleNamespace(intervention_mode="learned")
        model = SimpleNamespace(decoder=SimpleNamespace(qrank_score_bias=ranker))

        returned = MODULE.configure_qrank_mode(model, "disabled")

        self.assertIs(returned, ranker)
        self.assertIsNone(model.decoder.qrank_score_bias)


if __name__ == "__main__":
    unittest.main()
