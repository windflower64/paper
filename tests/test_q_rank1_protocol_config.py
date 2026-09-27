from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import YAMLConfig


class QRank1ProtocolConfigTest(unittest.TestCase):
    def test_q_rank1_is_the_same_protocol_c_plus_rank_bias_experiment(self):
        repo = Path(__file__).resolve().parents[1]
        config_path = (
            repo
            / "experiments"
            / "phase_q"
            / "q_rank1_c_gq1_b8a4_20e_testdev_local.yml"
        )
        self.assertTrue(config_path.is_file(), f"missing config: {config_path}")
        cfg = YAMLConfig(str(config_path)).yaml_cfg

        self.assertEqual(cfg["epochs"], 20)
        self.assertEqual(cfg["gradient_accumulation_steps"], 4)
        self.assertEqual(cfg["train_dataloader"]["total_batch_size"], 8)
        self.assertEqual(cfg["val_dataloader"]["total_batch_size"], 8)

        self.assertIs(cfg["DFINE"]["rgbt_enabled"], False)
        self.assertIs(cfg["DFINE"]["rgbt_sd2_enabled"], False)
        self.assertIs(cfg["DFINETransformer"]["hrqs_enabled"], False)
        self.assertIs(cfg["DFINETransformer"]["sqfr_enabled"], False)
        self.assertIs(cfg["DFINETransformer"]["qrank_enabled"], True)
        self.assertEqual(cfg["HGNetv2"]["return_idx"], [2, 3])
        self.assertEqual(cfg["HGNetv2"]["pat_sf_variant"], "global_query")

        val_dataset = cfg["val_dataloader"]["dataset"]
        self.assertTrue(val_dataset["img_folder"].endswith("/images/test"))
        self.assertTrue(
            val_dataset["ann_file"].endswith(
                "/annotations/instances_visible_common_test.json"
            )
        )


if __name__ == "__main__":
    unittest.main()
