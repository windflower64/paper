"""Run the stronger joint fusion and matched RGB control from an immutable snapshot."""
import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('E:/two_paper'); REPO=ROOT/'D-FINE'
REPORT=ROOT/'reports/159_local_alignment_joint'; SNAP=REPORT/'frozen_repo'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    global REPORT, SNAP
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',type=int,choices=[8,16,32])
    parser.add_argument('--attempt',choices=['initial','stable16'],default='initial')
    args=parser.parse_args()
    preflight_report=REPORT
    if args.attempt=='stable16':
        REPORT=ROOT/'reports/159_local_alignment_joint_stable16'
        SNAP=REPORT/'frozen_repo'
    REPORT.mkdir(parents=True,exist_ok=True)
    assert not SNAP.exists() and not (REPORT/'queue_status.json').exists()
    candidates=[b for b in (32,16,8) if (args.batch is None or args.batch==b) and all((preflight_report/f'preflight_{a}_b{b}.json').exists() for a in ('fusion','rgb'))]
    assert candidates
    batch=candidates[0]
    assert all(json.loads((preflight_report/f'preflight_{a}_b{batch}.json').read_text())['status']=='PASS' for a in ('fusion','rgb'))
    assert shutil.disk_usage(ROOT).free>25*2**30
    manifest={}; SNAP.mkdir()
    for folder,pattern in [('src','*.py'),('configs','*.yml'),('experiments','*.yml'),('tools','*.py')]:
        for source in (REPO/folder).rglob(pattern):
            dest=SNAP/source.relative_to(REPO); dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest); manifest[str(dest)]=sha(dest)
    assets=[ROOT/'outputs/S_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0/best_stg1.pth',
            ROOT/'outputs/M_IR_RAW_VERIFIED_GQ1_30E_TESTDEV/seed0/best_stg1.pth',
            *list((ROOT/'data/antiuav6k_ir_raw_verified').rglob('*.txt')),
            *list((ROOT/'data/antiuav6k_ir_raw_verified').rglob('*.json')),
            ROOT/'data/antiuav6k_common/annotations/instances_visible_common_train.json',
            ROOT/'data/antiuav6k_common/annotations/instances_visible_common_test.json']
    for asset in assets: manifest[str(asset)]=sha(asset)
    write(REPORT/'manifest.json',manifest)
    state=dict(status='ready',batch=batch,effective_batch=32,epochs=30,completed=[])
    env={k.upper():v for k,v in os.environ.items()}
    env.update(PYTHONIOENCODING='utf-8',PYTHONUTF8='1',PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_MODULE_LOADING='LAZY',TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='100')
    def save(**kwargs):
        state.update(kwargs,updated_at=datetime.datetime.now().isoformat())
        write(REPORT/'queue_status.json',state)
    try:
        for arm in ('fusion','rgb'):
            for name,digest in manifest.items(): assert sha(Path(name))==digest,name
            run=ROOT/f'outputs/M_LOCAL_ALIGN_JOINT_{arm.upper()}_B{batch}A{32//batch}_30E_TESTDEV/seed0'
            run.mkdir(parents=True,exist_ok=False)
            for mode in ('train','eval'):
                cmd=[sys.executable,'-u',str(SNAP/'tools/run_local_alignment_joint.py'),'--mode',mode,
                     '--arm',arm,'--batch',str(batch),'--run',str(run)]
                with (run/f'{mode}_console.log').open('x',encoding='utf-8') as out,(run/f'{mode}_error.log').open('x',encoding='utf-8') as err:
                    proc=subprocess.Popen(cmd,cwd=SNAP,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    save(status=mode,arm=arm,pid=proc.pid,run=str(run))
                    code=proc.wait()
                assert code==0,f'{arm}/{mode} exit {code}; inspect {run}'
            state['completed'].append(arm)
        for name,digest in manifest.items(): assert sha(Path(name))==digest,name
        save(status='complete',pid=None)
    except Exception as exc:
        save(status='failed',pid=None,error=repr(exc))
        raise


if __name__=='__main__':
    main()
