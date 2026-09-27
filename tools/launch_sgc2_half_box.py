"""Run missing BOX arm using immutable reference sources, isolated outputs."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import launch_s_sgc2_c_plus_m as base

ROOT, REPO = base.ROOT, base.REPO
REPORT = ROOT / 'reports/148_sgc2_half_box'
RUN = ROOT / 'outputs/C_PLUS_M_SD22_HALF_SGC2_BOX_DECAY9_14_B8A4_20E_TESTDEV/seed0'
HISTORICAL = ROOT / 'outputs/C_PLUS_M_SD22_HALF_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0/artifacts'
CONFIG_NAME = 's_sgc2_box_c_plus_m_sd22_half_b8a4_20e.yml'


def main():
    preflight = json.loads((REPORT / 'preflight.json').read_text())
    assert preflight['status'] == 'PASS' and preflight['same_initial_weights']
    if RUN.exists() and any(RUN.iterdir()):
        raise RuntimeError('Refusing to overwrite BOX output')
    snapshot = REPORT / 'frozen_repo'
    if snapshot.exists():
        raise RuntimeError('Refusing to overwrite source snapshot')
    manifest = json.loads((HISTORICAL / 'manifest.json').read_text())
    for name, digest in manifest.items():
        path = Path(name)
        try:
            relative = path.relative_to(REPO)
            checked = HISTORICAL / relative
        except ValueError:
            checked = path
        assert hashlib.sha256(checked.read_bytes()).hexdigest() == digest, str(checked)
    shutil.copytree(HISTORICAL, snapshot)
    shutil.copy2(REPO / 'experiments/phase_s' / CONFIG_NAME, snapshot / 'experiments/phase_s' / CONFIG_NAME)
    RUN.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       CUDA_MODULE_LOADING='LAZY', TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='100')
    command = [sys.executable, '-u', 'train.py', '-c', str(snapshot / 'experiments/phase_s' / CONFIG_NAME),
               '-t', str(base.CHECKPOINT), '--seed', '0', '--use-amp']
    base.REPORT = REPORT
    with (RUN / 'train_console.log').open('x', encoding='utf-8') as out, (RUN / 'train_error.log').open('x', encoding='utf-8') as err:
        process = subprocess.Popen(command, cwd=snapshot, env=environment, stdout=out, stderr=err)
        base.write_status(status='training', pid=process.pid, command=command, output_dir=str(RUN),
                          source_snapshot=str(snapshot), reference_manifest_verified=True)
        code = process.wait()
    if code:
        raise RuntimeError(f'Training exited {code}')
    rows = [json.loads(x) for x in (RUN / 'log.txt').read_text().splitlines() if x.strip()]
    assert [r['epoch'] for r in rows] == list(range(20))
    base.write_status(status='completed', epochs=20, output_dir=str(RUN),
                      independent_evaluation_pending=True)


if __name__ == '__main__':
    base.REPORT = REPORT
    try:
        main()
    except Exception as error:
        base.write_status(status='failed', error=str(error))
        raise
