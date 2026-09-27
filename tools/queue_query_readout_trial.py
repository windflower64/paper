"""Fixed 3x10 epoch queue, frozen source, stop on failure, no automatic resume."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
REPO=Path(__file__).resolve().parents[1];ROOT=REPO.parent
REPORT=ROOT/'reports/137_query_readout_trial';SNAP=REPORT/'frozen_repo'
STATE={'completed':[]}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(**kw):
    STATE.update(kw,updated_at=datetime.datetime.now().isoformat())
    (REPORT/'queue_status.json').write_text(json.dumps(STATE,indent=2),encoding='utf-8')

def process(arm,mode):
    run=ROOT/f'outputs/QUERY_READ_{arm.upper()}_B16A2_10E_TESTDEV/seed0'
    folder=run if mode=='train' else REPORT/arm
    folder.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(SNAP/'tools/run_query_readout_trial.py'),'--root',str(ROOT),'--arm',arm,'--mode',mode]
    env={k.upper():v for k,v in os.environ.items()}
    env.update(PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
               CUDA_MODULE_LOADING='LAZY',TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='100')
    with (folder/f'{mode}_console.log').open('x',encoding='utf-8') as out,(folder/f'{mode}_error.log').open('x',encoding='utf-8') as err:
        p=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        save(status=mode,arm=arm,pid=p.pid,command=cmd,console_log=str(folder/f'{mode}_console.log'))
        rc=p.wait()
    if rc:raise RuntimeError(f'{arm}/{mode} exited {rc}')

def main():
    if (REPORT/'queue_status.json').exists():raise RuntimeError('Existing queue status')
    for arm in ('none','box','sam'):
        if (ROOT/f'outputs/QUERY_READ_{arm.upper()}_B16A2_10E_TESTDEV/seed0').exists():raise RuntimeError('Existing run')
    assert shutil.disk_usage(ROOT).free>20*2**30
    SNAP.mkdir(exist_ok=False);hashes={}
    for directory,pattern in [('src','*.py'),('configs','*.yml'),('experiments','*.yml'),('tools','*.py')]:
        for p in (REPO/directory).rglob(pattern):
            dest=SNAP/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
            hashes[str(dest)]=sha(dest);assert sha(p)==hashes[str(dest)]
    masks=ROOT/'reports/104_sam3_role_control/masks_train'
    protected=[ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth',masks/'records.json',
               *list((masks/'masks').glob('*.png')),*list((ROOT/'data/antiuav6k_common/annotations').glob('instances_visible_common_*.json'))]
    hashes.update({str(p):sha(p) for p in protected})
    (REPORT/'manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    # SAM preflight was run interactively; the other arms are checked before any training.
    for arm in ('none','box'):
        process(arm,'preflight')
    checks=[json.loads((REPORT/f'preflight_{a}.json').read_text()) for a in ('none','box','sam')]
    assert all(c['status']=='PASS' for c in checks)
    assert len({c['initial_sha256'] for c in checks})==1
    for arm in ('none','box','sam'):
        assert all(sha(Path(p))==h for p,h in hashes.items()),'Frozen input changed'
        process(arm,'train')
        run=ROOT/f'outputs/QUERY_READ_{arm.upper()}_B16A2_10E_TESTDEV/seed0'
        protocol=json.loads((run/'protocol.json').read_text())
        assert protocol['initial_sha256']==checks[0]['initial_sha256']
        process(arm,'eval');STATE['completed'].append(arm)
    comparison={}
    for arm in ('none','box','sam'):
        run=ROOT/f'outputs/QUERY_READ_{arm.upper()}_B16A2_10E_TESTDEV/seed0'
        rows=[json.loads(s) for s in (run/'log.txt').read_text().splitlines() if s.strip()]
        telemetry=[json.loads(s) for s in (run/'guidance_batches.jsonl').read_text().splitlines() if s.strip()]
        assert len(telemetry)==2000
        best=max(rows,key=lambda r:r['test_coco_eval_bbox'][0])
        comparison[arm]={'best_epoch':best['epoch'],'best_metrics':best['test_coco_eval_bbox'],
            'last3_mean':[sum(r['test_coco_eval_bbox'][i] for r in rows[-3:])/3 for i in range(12)],
            'guidance_by_epoch':{str(e):{'eligible':sum(t['eligible'] for t in telemetry if t['epoch']==e),
                'images':sum(t['batch'] for t in telemetry if t['epoch']==e)} for e in range(10)}}
    (REPORT/'comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    save(status='complete',pid=None)

if __name__=='__main__':
    REPORT.mkdir(parents=True,exist_ok=True)
    if sys.argv[1:]==['--detach']:
        if (REPORT/'queue.lock').exists():raise RuntimeError('Existing lock')
        with (REPORT/'queue_console.log').open('x') as out,(REPORT/'queue_error.log').open('x') as err:
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve())],cwd=REPO,
                env={k.upper():v for k,v in os.environ.items()},stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),close_fds=True)
        print(p.pid);sys.exit(0)
    with (REPORT/'queue.lock').open('x') as f:f.write(str(os.getpid()))
    try:main()
    except Exception as e:save(status='failed',pid=None,error=str(e));raise
