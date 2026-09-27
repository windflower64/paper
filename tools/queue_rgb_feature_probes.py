"""Durable two-arm diagnostic queue. Stops on failure; no implicit retraining."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
REPORT = ROOT / "reports/97_rgb_preservation"


def now():
    return datetime.datetime.now().astimezone().isoformat()


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    status = {"started_at": now(), "status": "running", "arms": []}
    path = REPORT / "probe_queue_status.json"

    def save():
        path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, source in (
        ("C", "C_ONLY_GQ1_B8A4_20E_TESTDEV"),
        ("CM", "C_PLUS_M_SD22_B8A4_20E_TESTDEV"),
    ):
        output = ROOT / f"outputs/P0_RGB_PROBE_{name}_6E/seed0"
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", str(REPO / "tools/run_rgb_feature_probe.py"),
                   "--source", str(ROOT / "outputs" / source / "seed0/best_stg1.pth"),
                   "--output", str(output), "--epochs", "6"]
        arm = {"arm": name, "command": command, "started_at": now(), "status": "running"}
        status["arms"].append(arm)
        save()
        with (output / "console.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            arm["pid"] = process.pid
            save()
            code = process.wait()
        arm.update(returncode=code, finished_at=now(), status="completed" if code == 0 else "failed")
        save()
        if code:
            status["status"] = "failed"
            save()
            raise SystemExit(code)
    results = [json.loads((ROOT / f"outputs/P0_RGB_PROBE_{a}_6E/seed0/probe_result.json").read_text()) for a in ("C", "CM")]
    equal_heads = results[0]["initial_hashes"]["decoder"] == results[1]["initial_hashes"]["decoder"]
    status.update(status="completed" if equal_heads else "invalid_head_mismatch", finished_at=now(),
                  identical_decoder_initialization=equal_heads,
                  CM_minus_C_best_AP=results[1]["best_AP"] - results[0]["best_AP"],
                  CM_minus_C_last3_AP=results[1]["last3_AP"] - results[0]["last3_AP"],
                  next_action="Review diagnostics before launching P1 paired full training")
    save()


if __name__ == "__main__":
    main()
