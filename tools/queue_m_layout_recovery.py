"""Run the approved R1/R2 recovery experiment as a persistent local queue."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "D-FINE"
RUN = ROOT / "outputs/M_LAYOUT_RECOVERY_20E_TESTDEV"
REPORT = ROOT / "reports/M_LAYOUT_RECOVERY_20E_TESTDEV"
STATUS = REPORT / "queue_status.json"
PYTHON = Path(sys.executable).resolve()
PHASES = (
    ("R1_warmup", "r1_warmup", RUN / "R1_SD22/warmup"),
    ("R2_warmup", "r2_warmup", RUN / "R2_OTE2_LAYOUT/warmup"),
    ("R1_main", "r1_main", RUN / "R1_SD22/seed0"),
    ("R2_main", "r2_main", RUN / "R2_OTE2_LAYOUT/seed0"),
)


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def save(record):
    STATUS.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    for name in ("preflight.json", "training_path_preflight.json"):
        data = json.loads((REPORT / name).read_text(encoding="utf-8"))
        if data.get("status") != "PASS":
            raise RuntimeError(f"{name}: preflight did not pass")
    if not (RUN / "common_init.pth").is_file():
        raise FileNotFoundError(RUN / "common_init.pth")
    if STATUS.exists():
        raise FileExistsError(f"Refusing duplicate queue: {STATUS}")
    record = {"schema": "m_layout_recovery_queue_v1", "started_at": now(),
              "status": "running", "python": str(PYTHON), "phases": []}
    save(record)
    for name, config_name, output in PHASES:
        config = REPO / f"experiments/phase_m/m_layout_recovery_{config_name}.yml"
        if output.exists():
            entries = list(output.iterdir())
            if entries:
                raise FileExistsError(f"Refusing nonempty phase output: {output}")
        output.mkdir(parents=True, exist_ok=True)
        log = output / "train_console.log"
        command = [str(PYTHON), "-u", str(REPO / "train.py"), "-c", str(config),
                   "--seed", "0", "--use-amp"]
        phase = {"name": name, "config": str(config), "output": str(output),
                 "log": str(log), "status": "starting", "started_at": now()}
        record["phases"].append(phase)
        record["current_phase"] = name
        save(record)
        with log.open("w", encoding="utf-8") as stream:
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            proc = subprocess.Popen(command, cwd=REPO, stdout=stream,
                                    stderr=subprocess.STDOUT, env=env)
            phase["status"] = "running"
            phase["pid"] = proc.pid
            save(record)
            code = proc.wait()
        phase["exit_code"] = code
        phase["ended_at"] = now()
        phase["status"] = "completed" if code == 0 else "failed"
        save(record)
        if code:
            record["status"] = "failed"
            record["ended_at"] = now()
            save(record)
            raise RuntimeError(f"{name} exited with code {code}; see {log}")
        if not (output / "last.pth").is_file():
            record["status"] = "failed"
            save(record)
            raise RuntimeError(f"{name} finished without last.pth")
        time.sleep(2)
    record["current_phase"] = None
    record["status"] = "completed"
    record["ended_at"] = now()
    save(record)


if __name__ == "__main__":
    main()
