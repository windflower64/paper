"""Run the matched M-OTE2/no-SAM control warmup and main training."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "D-FINE"
REPORT = ROOT / "reports/M_R2_SAM_FACTORIAL_20E_TESTDEV"
OUTPUT = ROOT / "outputs/M_R2_SAM_FACTORIAL_20E_TESTDEV"
STATUS = REPORT / "queue_status.json"
PHASES = (("warmup", "m_r2_no_s_warmup.yml", OUTPUT / "no_s_warmup"),
          ("main", "m_r2_no_s_main.yml", OUTPUT / "no_s_seed0"))


def stamp():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def save(record):
    STATUS.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if STATUS.exists():
        raise FileExistsError(STATUS)
    preflight = json.loads((REPORT / "no_s_preflight.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise RuntimeError("No-S preflight not passed")
    for _, _, directory in PHASES:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite nonempty output: {directory}")
    record = {"schema": "m_r2_no_s_queue_v1", "status": "running",
              "started_at": stamp(), "python": str(Path(sys.executable).resolve()), "phases": []}
    save(record)
    for name, config_name, directory in PHASES:
        directory.mkdir(parents=True, exist_ok=True)
        config = REPO / "experiments/phase_m" / config_name
        log = directory / "train_console.log"
        phase = {"name": name, "config": str(config), "log": str(log),
                 "output": str(directory), "status": "starting", "started_at": stamp()}
        record["phases"].append(phase)
        record["current_phase"] = name
        save(record)
        command = [sys.executable, "-u", str(REPO / "train.py"), "-c", str(config),
                   "--seed", "0", "--use-amp"]
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=REPO, stdout=stream,
                                       stderr=subprocess.STDOUT, env=environment)
            phase["status"] = "running"
            phase["pid"] = process.pid
            save(record)
            code = process.wait()
        phase["exit_code"] = code
        phase["ended_at"] = stamp()
        phase["status"] = "completed" if code == 0 else "failed"
        save(record)
        if code != 0 or not (directory / "last.pth").is_file():
            record["status"] = "failed"
            record["ended_at"] = stamp()
            save(record)
            raise RuntimeError(f"{name} failed; see {log}")
    record["current_phase"] = None
    record["status"] = "completed"
    record["ended_at"] = stamp()
    save(record)


if __name__ == "__main__":
    main()
