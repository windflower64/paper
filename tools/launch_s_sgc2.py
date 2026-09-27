"""Launch one immutable SGC2-SAM run after its preflight passes."""
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
REPORT = ROOT / "reports/109_sgc2_teacher_retirement"


def write_status(**data):
    data["updated_at"] = datetime.datetime.now().isoformat()
    arm = data.get("arm", "sam")
    destination = REPORT / ("queue_status.json" if arm == "sam" else "queue_status_box.json")
    destination.write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "sam"
    assert arm in ("sam", "box")
    preflight_path = REPORT / ("preflight.json" if arm == "sam" else "preflight_box.json")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["status"] == "PASS"
    assert preflight["inference_exact_C_equivalence"]
    run = ROOT / f"outputs/S_SGC2_{arm.upper()}_DECAY9_14_B8A4_20E_TESTDEV/seed0"
    if run.exists() and any(run.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty output: {run}")

    sources = list((REPO / "src").rglob("*.py"))
    sources += list((REPO / "configs").rglob("*.yml"))
    sources += list((REPO / "experiments").rglob("*.yml"))
    sources += [
        REPO / "train.py",
        Path(__file__),
        REPO / "tools/preflight_s_sgc2.py",
        REPO / "tests/test_s_sgc1.py",
    ]
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    checkpoint = ROOT / "weights/m_sd2_joint_coco_thermal_identity_init.pth"
    labels = ROOT / "reports/104_sam3_role_control/masks_train"
    protected = [checkpoint, labels / "records.json", *list((labels / "masks").glob("*.png"))]
    for path in protected:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    artifacts = run / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=False)
    for source in sources:
        destination = artifacts / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (artifacts / "manifest.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )
    for name, digest in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen input changed before launch: {name}")

    command = [
        sys.executable,
        "-u",
        "train.py",
        "-c",
        str(REPO / f"experiments/phase_s/s_sgc2_{arm}_decay9_14_b8a4_20e.yml"),
        "-t",
        str(checkpoint),
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
    with (run / "train_console.log").open("x", encoding="utf-8") as stdout, (
        run / "train_error.log"
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
            arm=arm,
            pid=process.pid,
            command=command,
            log=str(run / "train_console.log"),
        )
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Training failed with code {return_code}")
    rows = [
        json.loads(line)
        for line in (run / "log.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["epoch"] for row in rows] == list(range(20))
    write_status(status="completed", arm=arm, epochs=20, log=str(run / "log.txt"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        REPORT.mkdir(parents=True, exist_ok=True)
        failed_arm = sys.argv[1] if len(sys.argv) > 1 else "sam"
        write_status(status="failed", arm=failed_arm, error=str(error))
        raise
