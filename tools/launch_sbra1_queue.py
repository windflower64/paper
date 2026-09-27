"""Launch three fingerprint-checked independent SBRA1 arms, no overwrite."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]; ROOT=REPO.parent
REPORT=ROOT/'reports/119_sbra1'

def main():
    arms=('sam','none','box')
    pre=[json.loads((REPORT/f'preflight_{a}.json').read_text(encoding='utf-8')) for a in arms]
    assert all(p['status']=='PASS' and p['exact_initial_forward'] and p['validation_no_masks'] for p in pre)
    assert len({p['initial_sha256'] for p in pre})==1
    runs={a:ROOT/f'outputs/S_BRA1_{a.upper()}_B8A4_20E_TESTDEV/seed0' for a in arms}
    for p in runs.values():
        if p.exists() and any(p.iterdir()):raise RuntimeError(f'Refusing nonempty run: {p}')
    files=[REPO/'train.py',Path(__file__),REPO/'tools/preflight_sbra1.py']
    for folder,pattern in [('src','*.py'),('configs','*.yml'),('experiments','*.yml')]:files.extend((REPO/folder).rglob(pattern))
    checkpoint=ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth'
    protected=[checkpoint,ROOT/'reports/104_sam3_role_control/masks_train/records.json']
    protected.extend((ROOT/'reports/104_sam3_role_control/masks_train/masks').glob('*.png'))
    protected.extend((ROOT/'data/antiuav6k_common/annotations').glob('*.json'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files+protected}
    def status(**kwargs):
        kwargs['updated_at']=datetime.datetime.now().isoformat()
        (REPORT/'queue_status.json').write_text(json.dumps(kwargs,indent=2),encoding='utf-8')
    env=os.environ.copy(); env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_MODULE_LOADING='LAZY',TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='100')
    done=[]
    for arm in arms:
        for path,digest in hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                status(status='stopped_input_changed',path=path,completed=done); return
        run=runs[arm]; artifacts=run/'artifacts'; artifacts.mkdir(parents=True,exist_ok=False)
        for p in files:
            dest=artifacts/p.relative_to(REPO); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
        (artifacts/'manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
        command=[sys.executable,'-u','train.py','-c',str(REPO/f'experiments/phase_s/s_sbra1_{arm}_b8a4_20e.yml'),'-t',str(checkpoint),'--seed','0','--use-amp']
        with (run/'train_console.log').open('x',encoding='utf-8') as out,(run/'train_error.log').open('x',encoding='utf-8') as err:
            p=subprocess.Popen(command,cwd=REPO,env=env,stdout=out,stderr=err)
            status(status='training',arm=arm,pid=p.pid,completed=done,command=command)
            code=p.wait()
        if code:
            status(status='failed',arm=arm,exit_code=code,completed=done); return
        done.append(arm)
    status(status='complete',completed=done)

if __name__=='__main__':main()
