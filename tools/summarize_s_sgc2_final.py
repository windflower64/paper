"""Create the reproducible final SGC2 comparison used by the Chinese report."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/109_sgc2_teacher_retirement"
RUNS = {
    "C": ROOT / "outputs/C_ONLY_GQ1_B8A4_20E_TESTDEV/seed0/log.txt",
    "SGC1_SAM": ROOT / "outputs/S_SGC1_SAM_B8A4_20E_TESTDEV/seed0/log.txt",
    "SGC1_BOX": ROOT / "outputs/S_SGC1_BOX_B8A4_20E_TESTDEV/seed0/log.txt",
    "SGC2_SAM": ROOT / "outputs/S_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0/log.txt",
    "SGC2_BOX": ROOT / "outputs/S_SGC2_BOX_DECAY9_14_B8A4_20E_TESTDEV/seed0/log.txt",
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def delta_points(left, right):
    return [(a - b) * 100 for a, b in zip(left, right)]


def main():
    runs = {}
    for name, path in RUNS.items():
        rows = read_rows(path)
        assert [row["epoch"] for row in rows] == list(range(20))
        best = max(rows, key=lambda row: row["test_coco_eval_bbox"][0])
        runs[name] = {
            "best_epoch": best["epoch"],
            "coco_eval_bbox": best["test_coco_eval_bbox"],
            "last_ap": rows[-1]["test_coco_eval_bbox"][0],
            "tail5_ap_mean": mean(row["test_coco_eval_bbox"][0] for row in rows[-5:]),
        }

    metrics = {name: item["coco_eval_bbox"] for name, item in runs.items()}
    schedule_sam = delta_points(metrics["SGC2_SAM"], metrics["SGC1_SAM"])
    schedule_box = delta_points(metrics["SGC2_BOX"], metrics["SGC1_BOX"])
    result = {
        "metric_order": [
            "AP", "AP50", "AP75", "APS", "APM", "APL",
            "AR1", "AR10", "AR100", "ARS", "ARM", "ARL",
        ],
        "units": "COCO fractions in runs; comparison deltas are AP points",
        "runs": runs,
        "deltas_points": {
            "SGC2_SAM_minus_C": delta_points(metrics["SGC2_SAM"], metrics["C"]),
            "SGC2_SAM_minus_SGC2_BOX": delta_points(
                metrics["SGC2_SAM"], metrics["SGC2_BOX"]
            ),
            "SGC2_SAM_minus_SGC1_SAM": schedule_sam,
            "SGC2_BOX_minus_SGC1_BOX": schedule_box,
            "schedule_by_teacher_interaction": [
                sam - box for sam, box in zip(schedule_sam, schedule_box)
            ],
        },
        "independent_evaluation": {
            "sam": json.loads((REPORT / "evaluation/best_ema.json").read_text(encoding="utf-8")),
            "box": json.loads((REPORT / "evaluation/best_ema_box.json").read_text(encoding="utf-8")),
        },
        "per_image": json.loads(
            (REPORT / "per_image_comparison.json").read_text(encoding="utf-8")
        ),
        "decision": {
            "selected": "SGC2_SAM",
            "role": "training-only SAM shape grouping curriculum; no inference branch",
            "boundary": "small-target-specific candidate, not universal contour restoration",
            "next_gate": "joint C+M RGB-T training under the same test-development protocol",
        },
    }
    (REPORT / "final_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["deltas_points"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
