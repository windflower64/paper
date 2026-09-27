import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "export_bpc1_sam_carrier_overview.py"
)
SPEC = importlib.util.spec_from_file_location("export_bpc1_sam_carrier_overview", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BPC1CarrierOverviewTest(unittest.TestCase):
    def test_presentation_uses_unambiguous_gate_and_delta_names(self):
        self.assertIn("学生门控", MODULE.PRESENTATION_HEADERS)
        self.assertIn("BPC补偿增量", MODULE.PRESENTATION_HEADERS)
        self.assertNotIn("学生边界门", MODULE.PRESENTATION_HEADERS)
        self.assertNotIn("BPC载体响应", MODULE.PRESENTATION_HEADERS)

    def test_manifest_selection_preserves_the_fifth_page_sample_order(self):
        records = [
            {"image_id": 30, "annotation_id": 300, "accepted": True},
            {"image_id": 10, "annotation_id": 100, "accepted": True},
            {"image_id": 20, "annotation_id": 200, "accepted": True},
        ]
        manifest = {
            "samples": [
                {"image_id": 20},
                {"image_id": 10},
            ]
        }

        selected = MODULE.records_from_manifest(records, manifest)

        self.assertEqual([row["image_id"] for row in selected], [20, 10])


if __name__ == "__main__":
    unittest.main()
