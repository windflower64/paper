from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import YAMLConfig


class SSgc2CPlusMConfigTest(unittest.TestCase):
    def test_joint_config_keeps_sam_on_rgb_s8_and_m_sd22_on_rgbt(self):
        repo = Path(__file__).resolve().parents[1]
        config_path = (
            repo
            / "experiments"
            / "phase_s"
            / "s_sgc2_sam_c_plus_m_sd22_b8a4_20e.yml"
        )
        cfg = YAMLConfig(str(config_path)).yaml_cfg

        self.assertEqual(cfg["epochs"], 20)
        self.assertEqual(cfg["gradient_accumulation_steps"], 4)
        self.assertEqual(cfg["train_dataloader"]["total_batch_size"], 8)
        self.assertEqual(cfg["val_dataloader"]["total_batch_size"], 8)

        model_cfg = cfg["DFINE"]
        self.assertIs(model_cfg["rgbt_enabled"], True)
        self.assertIs(model_cfg["rgbt_sd2_enabled"], True)
        self.assertIs(model_cfg["rgbt_sd2_thermal_contrastive"], True)
        self.assertIs(model_cfg["rgbt_freeze_thermal_stream"], True)
        self.assertIs(model_cfg["sgc_enabled"], True)
        self.assertEqual(model_cfg["sgc_supervision"], "sam")
        self.assertEqual(model_cfg["sgc_aux_weight"], 10.0)
        self.assertEqual(model_cfg["sgc_decay_start"], 9)
        self.assertEqual(model_cfg["sgc_decay_end"], 14)

        self.assertIs(cfg["DFINETransformer"]["hrqs_enabled"], False)
        self.assertEqual(cfg["HGNetv2"]["return_idx"], [1, 2, 3])
        self.assertEqual(cfg["HGNetv2"]["pat_sf_variant"], "global_query")

        train_dataset = cfg["train_dataloader"]["dataset"]
        val_dataset = cfg["val_dataloader"]["dataset"]
        self.assertEqual(train_dataset["type"], "RGBTCocoDetection")
        self.assertEqual(val_dataset["type"], "RGBTCocoDetection")
        self.assertTrue(train_dataset["infrared_folder"].endswith("/train/infrared/images"))
        self.assertTrue(val_dataset["infrared_folder"].endswith("/test/infrared/images"))
        self.assertEqual(
            train_dataset["sam_mask_root"],
            "E:/two_paper/reports/104_sam3_role_control/masks_train",
        )
        self.assertIsNone(val_dataset["sam_mask_root"])
        self.assertTrue(
            val_dataset["ann_file"].endswith(
                "/annotations/instances_visible_common_test.json"
            )
        )
        self.assertEqual(
            cfg["output_dir"],
            "E:/two_paper/outputs/C_PLUS_M_SD22_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0",
        )


if __name__ == "__main__":
    unittest.main()
