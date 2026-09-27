"""Read-only paired target audit; never treats query indices as identities."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "SAM": "reports/112_sgc2_rgbt_half/final/diagnostics/epoch8/targets.json",
    "NONE": "reports/113_sgc0_half_control/final/diagnostics/epoch13/targets.json",
}


def compare(rows, left, right):
    result = {"n": len(rows)}
    for field in ("top1_iou", "best_iou_10", "best_iou_300"):
        delta = [right(r)[field] - left(r)[field] for r in rows]
        result[field] = {
            "mean_delta": sum(delta) / max(1, len(delta)),
            "improved_gt_0_01": sum(d > .01 for d in delta),
            "worsened_gt_0_01": sum(d < -.01 for d in delta),
        }
    for threshold in (.5, .75):
        a = [left(r)["top1_iou"] >= threshold for r in rows]
        b = [right(r)["top1_iou"] >= threshold for r in rows]
        result[f"top1_hits_{threshold}"] = {
            "recovered": sum(not x and y for x, y in zip(a, b)),
            "lost": sum(x and not y for x, y in zip(a, b)),
        }
    return result


def main():
    data = {k: json.loads((ROOT / p).read_text(encoding="utf-8")) for k, p in SOURCES.items()}
    # This dataset has one GT per positive image. Fail rather than silently
    # pairing multiple targets without a stable annotation identifier.
    for rows in data.values():
        assert len({r["image_id"] for r in rows}) == len(rows)
    maps = {k: {r["image_id"]: r for r in rows} for k, rows in data.items()}
    assert maps["SAM"].keys() == maps["NONE"].keys()
    assert all(maps["SAM"][i]["scale"] == maps["NONE"][i]["scale"] for i in maps["SAM"])
    result = {"sources": SOURCES, "caveats": [
        "cached best checkpoints at different epochs; exploratory, not significance",
        "top1 hits ignore score thresholds; not AP or precision",
        "geometry and ranking do not directly measure feature separability",
        "M on/off is same-weight intervention, not independently trained RGB baseline",
    ], "within_checkpoint": {}, "between_checkpoints": {}}
    for name, rows in data.items():
        result["within_checkpoint"][name] = {
            scale: compare([r for r in rows if r["scale"] == scale],
                           lambda r: r["0.0"], lambda r: r["1.0"])
            for scale in ("small", "medium")}
    for scale in ("small", "medium"):
        rows = [r for r in data["NONE"] if r["scale"] == scale]
        result["between_checkpoints"][scale] = compare(
            rows, lambda r: r["1.0"], lambda r: maps["SAM"][r["image_id"]]["1.0"])
    def failure(r, threshold):
        if r["best_iou_300"] < threshold:
            return "no_good_candidate"
        if r["top1_iou"] < threshold:
            return "good_candidate_not_first"
        return "good_candidate_first"
    result["failure_decomposition"] = {}
    for name, rows in data.items():
        result["failure_decomposition"][name] = {}
        for scale in ("small", "medium"):
            selected = [r for r in rows if r["scale"] == scale]
            groups = {}
            for mode in ("0.0", "1.0"):
                groups[mode] = {}
                for threshold in (.5, .75, .9):
                    counts = {k: 0 for k in ("no_good_candidate", "good_candidate_not_first", "good_candidate_first")}
                    for row in selected:
                        counts[failure(row[mode], threshold)] += 1
                    assert sum(counts.values()) == len(selected)
                    groups[mode][str(threshold)] = counts
            result["failure_decomposition"][name][scale] = groups
    result["paired_failure_transitions"] = {}
    for threshold in (.75, .9):
        transitions = {}
        for row in data["NONE"]:
            if row["scale"] != "small":
                continue
            key = failure(row["1.0"], threshold) + " -> " + failure(maps["SAM"][row["image_id"]]["1.0"], threshold)
            transitions[key] = transitions.get(key, 0) + 1
        result["paired_failure_transitions"][str(threshold)] = transitions
    output = ROOT / "reports/147_core_claim_audit/cached_target_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
