"""Wait for teacher completion, export, preflight, then frozen four-arm training."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil

REPO=Path(__file__).resolve().parents[1];ROOT=REPO.parent
REPORT=ROOT/'reports/104_sam3_role_control'
ARMS=['none','box','sam','edge']


def status(**data):
    data['updated_at']=datetime.datetime.now().isoformat()
    (REPORT/'queue_status.json').write_text(json.dumps(data,indent=2),encoding='utf-8')


def execute(command,out,err):
    with out.open('x',encoding='utf-8') as stdout,err.open('x',encoding='utf-8') as stderr:
        process=subprocess.Popen(command,cwd=REPO,env=ENV,stdout=stdout,stderr=stderr)
        status(status='running',command=command,pid=process.pid,log=str(out))
        code=process.wait()
    if code: raise RuntimeError(f'Command failed ({code}), see {err}')


ENV=os.environ.copy()
ENV.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_MODULE_LOADING='LAZY',TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='100')


def main():
    teacher_pid=int(sys.argv[1])
    sources=list((REPO/'src').rglob('*.py'))+list((REPO/'configs').rglob('*.yml'))+list((REPO/'experiments').rglob('*.yml'))
    sources += [REPO/'train.py',Path(__file__),REPO/'tools/preflight_s_rc1.py',REPO/'tools/export_sam3_role_labels.py',REPO/'tests/test_s_rc1.py',REPO/'tools/audit_sam3_positive_edges.py',REPO/'tools/sam3_teacher_pilot.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for arm in ARMS:
        run=ROOT/f'outputs/S_RC1_{arm.upper()}_B8A4_20E_TESTDEV/seed0'
        if run.exists() and any(run.iterdir()): raise RuntimeError(f'Nonempty output: {run}')
    while not (REPORT/'teacher_raw/summary.json').exists():
        if not psutil.pid_exists(teacher_pid):raise RuntimeError('Teacher stopped without complete summary')
        process=psutil.Process(teacher_pid)
        if 'audit_sam3_positive_edges.py' not in ' '.join(process.cmdline()):raise RuntimeError('Teacher PID reused')
        status(status='waiting_teacher',pid=teacher_pid)
        time.sleep(10)
    assert json.loads((REPORT/'teacher_raw/summary.json').read_text())['complete']
    execute(['D:/conda/envs/sam3_teacher/python.exe','-u',str(REPO/'tools/export_sam3_role_labels.py')],REPORT/'export.stdout.log',REPORT/'export.stderr.log')
    hashes[str(REPORT/'masks_train/records.json')]=hashlib.sha256((REPORT/'masks_train/records.json').read_bytes()).hexdigest()
    # Freeze labels as well as code: no mid-queue pseudo-label replacement.
    for p in (REPORT/'masks_train/masks').glob('*.png'):hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    for arm in ARMS:
        execute([sys.executable,'-u',str(REPO/'tools/preflight_s_rc1.py'),arm],REPORT/f'preflight_{arm}.stdout.log',REPORT/f'preflight_{arm}.stderr.log')
    checks=[json.loads((REPORT/f'preflight_{arm}.json').read_text()) for arm in ARMS]
    assert all(c['status']=='PASS' for c in checks)
    assert len({c['initial_model_sha256'] for c in checks})==1
    checkpoint=ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth'
    hashes[str(checkpoint)]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for arm in ARMS:
        for name,digest in hashes.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Frozen source or label changed: {name}')
        run=ROOT/f'outputs/S_RC1_{arm.upper()}_B8A4_20E_TESTDEV/seed0'
        artifacts=run/'artifacts';artifacts.mkdir(parents=True,exist_ok=False)
        for source in sources:
            relative=source.relative_to(REPO);dest=artifacts/relative
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        (artifacts/'manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
        config=REPO/f'experiments/phase_s/s_rc1_{arm}_b8a4_20e.yml'
        execute([sys.executable,'-u','train.py','-c',str(config),'-t',str(checkpoint),'--seed','0','--use-amp'],run/'train_console.log',run/'train_error.log')
        logs=[json.loads(line) for line in (run/'log.txt').read_text().splitlines() if line.strip()]
        assert [row['epoch'] for row in logs]==list(range(20)),f'Incomplete {arm}'
    status(status='completed',arms=ARMS)


if __name__=='__main__':
    try: main()
    except Exception as error:
        status(status='failed',error=str(error));raise
