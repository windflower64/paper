"""Immutable two-arm SGC1 training queue, fail closed on changes/errors."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]; ROOT=REPO.parent
REPORT=ROOT/'reports/105_sam_group_contrast_pilot'
ARMS=['sam','box']


def status(**data):
    data['updated_at']=datetime.datetime.now().isoformat()
    (REPORT/'queue_status.json').write_text(json.dumps(data,indent=2),encoding='utf-8')


def main():
    if (REPORT/'queue_status.json').exists():
        raise RuntimeError('Existing queue status; inspect before any restart')
    checks=[json.loads((REPORT/f'preflight_{a}.json').read_text()) for a in ARMS]
    assert all(c['status']=='PASS' and c['inference_exact_C_equivalence'] for c in checks)
    assert len({c['initial_model_sha256'] for c in checks})==1
    sources=list((REPO/'src').rglob('*.py'))+list((REPO/'configs').rglob('*.yml'))+list((REPO/'experiments').rglob('*.yml'))
    sources += [REPO/'train.py',Path(__file__),REPO/'tools/preflight_s_sgc1.py',REPO/'tools/pilot_s_sgc1.py',REPO/'tools/summarize_s_sgc1.py',REPO/'tests/test_s_sgc1.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    checkpoint=ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth'
    labels=ROOT/'reports/104_sam3_role_control/masks_train'
    for p in [checkpoint,labels/'records.json',*list((labels/'masks').glob('*.png'))]:
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    for arm in ARMS:
        run=ROOT/f'outputs/S_SGC1_{arm.upper()}_B8A4_20E_TESTDEV/seed0'
        if run.exists() and any(run.iterdir()): raise RuntimeError(f'Nonempty output: {run}')
    env=os.environ.copy()
    env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_MODULE_LOADING='LAZY',TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='100')
    for arm in ARMS:
        for name,digest in hashes.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'Frozen source/label changed: {name}')
        run=ROOT/f'outputs/S_SGC1_{arm.upper()}_B8A4_20E_TESTDEV/seed0'
        artifacts=run/'artifacts'; artifacts.mkdir(parents=True,exist_ok=False)
        for source in sources:
            dest=artifacts/source.relative_to(REPO)
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,dest)
        (artifacts/'manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
        command=[sys.executable,'-u','train.py','-c',str(REPO/f'experiments/phase_s/s_sgc1_{arm}_b8a4_20e.yml'),
                 '-t',str(checkpoint),'--seed','0','--use-amp']
        with (run/'train_console.log').open('x',encoding='utf-8') as out,(run/'train_error.log').open('x',encoding='utf-8') as err:
            process=subprocess.Popen(command,cwd=REPO,env=env,stdout=out,stderr=err)
            status(status='training',arm=arm,pid=process.pid,command=command,log=str(run/'train_console.log'))
            code=process.wait()
            if code: raise RuntimeError(f'Training failed: {arm}, code={code}')
        logs=[json.loads(x) for x in (run/'log.txt').read_text().splitlines() if x.strip()]
        assert [r['epoch'] for r in logs]==list(range(20)),f'Incomplete {arm}'
    status(status='completed',arms=ARMS)


if __name__=='__main__':
    try: main()
    except Exception as error:
        # Preserve existing status when a duplicate launch was rejected.
        if not (REPORT/'queue_status.json').exists() or 'Existing queue status' not in str(error):
            status(status='failed',error=str(error))
        raise
