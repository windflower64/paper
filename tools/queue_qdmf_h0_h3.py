"""Run the frozen QDMF H0--H3 matrix serially on one GPU."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PYTHON = Path(sys.executable)
INIT = WORKSPACE / "weights/stql_qcer_common_init_v1.pth"
QUEUE_DIR = WORKSPACE / "reports/20260923_qdmf_v1/formal_queue"
STATE = QUEUE_DIR / "queue_state.json"

ARMS = [
    ("H0", ROOT / "experiments/phase_qdmf/h0_c_only.yml"),
    ("H1", ROOT / "experiments/phase_qdmf/h1_c_stql.yml"),
    ("H2", ROOT / "experiments/phase_qdmf/h2_c_qdmf.yml"),
    ("H3", ROOT / "experiments/phase_qdmf/h3_s_qdmf.yml"),
]


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def save(state):
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    if not INIT.exists():
        raise FileNotFoundError(INIT)
    state = {
        "schema": "qdmf_h0_h3_formal_queue_v1",
        "status": "running",
        "started": now(),
        "python": str(PYTHON),
        "initialization": str(INIT),
        "seed": 0,
        "order": [name for name, _ in ARMS],
        "active": None,
        "completed": [],
    }
    save(state)
    for name, config in ARMS:
        log_path = QUEUE_DIR / f"{name}.log"
        command = [
            str(PYTHON), "-u", str(ROOT / "train.py"),
            "-c", str(config), "-t", str(INIT), "--seed", "0",
        ]
        state["active"] = {
            "arm": name, "config": str(config), "log": str(log_path),
            "started": now(), "command": command,
        }
        save(state)
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            state["active"]["pid"] = process.pid
            save(state)
            return_code = process.wait()
        record = dict(state["active"])
        record.update(finished=now(), return_code=return_code)
        state["completed"].append(record)
        state["active"] = None
        if return_code != 0:
            state["status"] = "failed"
            state["failed_arm"] = name
            state["finished"] = now()
            save(state)
            raise SystemExit(return_code)
        save(state)
    state["status"] = "completed"
    state["finished"] = now()
    save(state)


if __name__ == "__main__":
    main()
