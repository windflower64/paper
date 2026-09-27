"""Persistent warmup/main training queue for the ordinary RGB-T fusion control."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "D-FINE"
REPORT = ROOT / "reports/M_GLOBAL_CONTROL_20E_TESTDEV"
RUN = ROOT / "outputs/M_GLOBAL_CONTROL_20E_TESTDEV"
STATUS = REPORT / "queue_status.json"


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def save(value):
    STATUS.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    preflight = json.loads((REPORT / "preflight.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise RuntimeError("Global control preflight not passed")
    if STATUS.exists():
        raise FileExistsError(STATUS)
    record = {"schema": "m_global_control_queue_v1", "status": "running",
              "started_at": now(), "phases": []}
    save(record)
    for name, config_name, output in (
        ("warmup", "m_global_control_warmup.yml", RUN / "warmup"),
        ("main", "m_global_control_main.yml", RUN / "seed0"),
    ):
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
        log = output / "train_console.log"
        config = REPO / "experiments/phase_m" / config_name
        phase = {"name": name, "status": "starting", "config": str(config),
                 "output": str(output), "log": str(log), "started_at": now()}
        record["phases"].append(phase)
        record["current_phase"] = name
        save(record)
        command = [sys.executable, "-u", str(REPO / "train.py"), "-c", str(config),
                   "--seed", "0", "--use-amp"]
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        with log.open("w", encoding="utf-8") as stream:
            proc = subprocess.Popen(command, cwd=REPO, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
            phase["status"] = "running"
            phase["pid"] = proc.pid
            save(record)
            code = proc.wait()
        phase["status"] = "completed" if code == 0 else "failed"
        phase["exit_code"] = code
        phase["ended_at"] = now()
        save(record)
        if code or not (output / "last.pth").is_file():
            record["status"] = "failed"
            record["ended_at"] = now()
            save(record)
            raise RuntimeError(f"{name} failed; see {log}")
    record["status"] = "completed"
    record["current_phase"] = None
    record["ended_at"] = now()
    save(record)


if __name__ == "__main__":
    main()
