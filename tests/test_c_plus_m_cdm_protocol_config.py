from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import YAMLConfig


class CPlusMCdmProtocolConfigTest(unittest.TestCase):
    def test_c_plus_m_removes_d_and_keeps_the_frozen_cdm_protocol(self):
        repo = Path(__file__).resolve().parents[1]
        config_path = (
            repo
            / "experiments"
            / "phase_m"
            / "c_plus_m_sd22_b8a4_20e_testdev_local.yml"
        )
        cfg = YAMLConfig(str(config_path)).yaml_cfg

        self.assertEqual(cfg["epochs"], 20)
        self.assertEqual(cfg["gradient_accumulation_steps"], 4)
        self.assertEqual(cfg["train_dataloader"]["total_batch_size"], 8)
        self.assertEqual(cfg["val_dataloader"]["total_batch_size"], 8)

        self.assertIs(cfg["DFINE"]["rgbt_enabled"], True)
        self.assertIs(cfg["DFINE"]["rgbt_sd2_enabled"], True)
        self.assertIs(cfg["DFINE"]["rgbt_sd2_thermal_contrastive"], True)
        self.assertIs(cfg["DFINE"]["rgbt_freeze_thermal_stream"], True)
        self.assertIs(cfg["DFINETransformer"]["hrqs_enabled"], False)
        self.assertEqual(cfg["HGNetv2"]["return_idx"], [2, 3])
        self.assertEqual(cfg["HGNetv2"]["pat_sf_variant"], "global_query")

        train_dataset = cfg["train_dataloader"]["dataset"]
        val_dataset = cfg["val_dataloader"]["dataset"]
        self.assertEqual(train_dataset["type"], "RGBTCocoDetection")
        self.assertEqual(val_dataset["type"], "RGBTCocoDetection")
        self.assertTrue(train_dataset["img_folder"].endswith("/images/train"))
        self.assertTrue(val_dataset["img_folder"].endswith("/images/test"))
        self.assertTrue(
            val_dataset["ann_file"].endswith(
                "/annotations/instances_visible_common_test.json"
            )
        )


if __name__ == "__main__":
    unittest.main()
