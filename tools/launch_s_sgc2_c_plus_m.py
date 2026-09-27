"""在联合预检通过后，启动并守护 C+M-SD2.2+SGC2-SAM 训练。"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
REPORT = ROOT / "reports/110_sgc2_rgbt_joint"
RUN = ROOT / "outputs/C_PLUS_M_SD22_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0"
CONFIG = REPO / "experiments/phase_s/s_sgc2_sam_c_plus_m_sd22_b8a4_20e.yml"
CHECKPOINT = ROOT / "weights/m_sd2_joint_coco_thermal_identity_init.pth"


def write_status(**data):
    data["updated_at"] = datetime.datetime.now().isoformat()
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "queue_status.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    preflight = json.loads((REPORT / "preflight.json").read_text(encoding="utf-8"))
    assert preflight["status"] == "PASS"
    assert max(preflight["inference_exact_without_sgc"].values()) == 0.0
    assert preflight["sgc_direct_m_gradient_max"] == 0.0
    if RUN.exists() and any(RUN.iterdir()):
        raise RuntimeError(f"拒绝覆盖已有输出目录：{RUN}")

    sources = list((REPO / "src").rglob("*.py"))
    sources += list((REPO / "configs").rglob("*.yml"))
    sources += list((REPO / "experiments").rglob("*.yml"))
    sources += [
        REPO / "train.py",
        Path(__file__),
        REPO / "tools/preflight_s_sgc2_c_plus_m.py",
        REPO / "tests/test_s_sgc2_c_plus_m_config.py",
    ]
    protected = [
        CHECKPOINT,
        ROOT / "reports/104_sam3_role_control/masks_train/records.json",
        ROOT / "data/antiuav6k_common/annotations/instances_visible_common_train.json",
        ROOT / "data/antiuav6k_common/annotations/instances_visible_common_test.json",
    ]
    sources += globals().get("EXTRA_SOURCES", [])
    protected += list(
        (ROOT / "reports/104_sam3_role_control/masks_train/masks").glob("*.png")
    )
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*sources, *protected]
    }

    artifacts = RUN / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    for source in sources:
        destination = artifacts / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (artifacts / "manifest.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, digest in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"启动前冻结输入发生变化：{name}")

    command = [
        sys.executable,
        "-u",
        "train.py",
        "-c",
        str(CONFIG),
        "-t",
        str(CHECKPOINT),
        "--seed",
        "0",
        "--use-amp",
    ]
    environment = os.environ.copy()
    environment.update(
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        CUDA_MODULE_LOADING="LAZY",
        TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT="100",
    )
    with (RUN / "train_console.log").open("x", encoding="utf-8") as stdout, (
        RUN / "train_error.log"
    ).open("x", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
        write_status(
            status="training",
            pid=process.pid,
            command=command,
            config=str(CONFIG),
            output_dir=str(RUN),
            console_log=str(RUN / "train_console.log"),
            metric_log=str(RUN / "log.txt"),
        )
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"训练进程异常退出，代码：{return_code}")
    rows = [
        json.loads(line)
        for line in (RUN / "log.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["epoch"] for row in rows] == list(range(20))
    write_status(status="completed", epochs=20, metric_log=str(RUN / "log.txt"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_status(status="failed", error=str(error))
        raise
