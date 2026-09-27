"""RGB stability gate before matched local versus coarse-aligned fusion training."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('E:/two_paper');REPO=ROOT/'D-FINE'
REPORT=ROOT/'reports/160_frozen_backbone_alignment';SNAP=REPORT/'frozen_repo'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not SNAP.exists() and not (REPORT/'queue_status.json').exists()
    assert json.loads((REPORT/'bypass_summary.json').read_text())['status']=='PASS'
    for arm in ('rgb','local','aligned'):
        assert json.loads((REPORT/f'preflight_{arm}.json').read_text())['status']=='PASS'
    SNAP.mkdir();manifest={}
    for folder,pattern in [('src','*.py'),('configs','*.yml'),('experiments','*.yml'),('tools','*.py')]:
        for source in (REPO/folder).rglob(pattern):
            dest=SNAP/source.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest);manifest[str(dest)]=sha(dest)
    assets=[ROOT/'outputs/S_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0/best_stg1.pth',
        ROOT/'outputs/M_IR_RAW_VERIFIED_GQ1_30E_TESTDEV/seed0/best_stg1.pth',
        *list((ROOT/'data/antiuav6k_ir_raw_verified').rglob('*.txt')),
        *list((ROOT/'data/antiuav6k_ir_raw_verified').rglob('*.json')),
        ROOT/'data/antiuav6k_common/annotations/instances_visible_common_train.json',
        ROOT/'data/antiuav6k_common/annotations/instances_visible_common_test.json']
    for asset in assets:manifest[str(asset)]=sha(asset)
    write(REPORT/'manifest.json',manifest)
    state=dict(status='ready',batch=16,effective_batch=32,epochs=30,completed=[])
    def save(**kwargs):
        state.update(kwargs,updated_at=datetime.datetime.now().isoformat());write(REPORT/'queue_status.json',state)
    def check():
        for name,digest in manifest.items():assert sha(Path(name))==digest,name
    env={k.upper():v for k,v in os.environ.items()}
    env.update(PYTHONIOENCODING='utf-8',PYTHONUTF8='1',PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_MODULE_LOADING='LAZY',TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='100')
    def execute(mode,arm,run):
        cmd=[sys.executable,'-u',str(SNAP/'tools/run_frozen_backbone_alignment.py'),
             '--mode',mode,'--arm',arm,'--run',str(run)]
        run.mkdir(parents=True,exist_ok=True)
        with (run/f'{mode}_console.log').open('x',encoding='utf-8') as out,(run/f'{mode}_error.log').open('x',encoding='utf-8') as err:
            proc=subprocess.Popen(cmd,cwd=SNAP,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            save(status=mode,arm=arm,pid=proc.pid,run=str(run));code=proc.wait()
        assert code==0,f'{arm}/{mode} failed {code}'
    # Define engineering tolerances BEFORE any new training results are available.
    criteria=dict(best_ap_max_drop_points=.1,last5_mean_ap_max_drop_points=.2,last5_mean_aps_max_drop_points=.3)
    write(REPORT/'preregistered_stability_gate.json',dict(criteria=criteria,
        note='工程稳定性门槛，不是统计显著性检验；只用于阻止在明显退化的RGB训练设置上继续融合试验。'))
    try:
        execute('initial','rgb',REPORT/'initial')
        baseline=json.loads((REPORT/'initial/initial_verified.json').read_text())['metrics']
        for arm in ('rgb','local','aligned'):
            check()
            run=ROOT/f'outputs/M_FIXED_BACKBONE_{arm.upper()}_B16A2_30E_TESTDEV/seed0'
            assert not run.exists();run.mkdir(parents=True)
            execute('train',arm,run);execute('eval',arm,run)
            assert json.loads((run/'fixed_state_verified.json').read_text())['status']=='PASS'
            state['completed'].append(arm)
            if arm=='rgb':
                rows=[json.loads(s) for s in (run/'log.txt').read_text().splitlines() if s.strip()]
                best=max(r['test_coco_eval_bbox'][0] for r in rows)
                ap=sum(r['test_coco_eval_bbox'][0] for r in rows[-5:])/5
                aps=sum(r['test_coco_eval_bbox'][3] for r in rows[-5:])/5
                passed=best>=baseline[0]-.001 and ap>=baseline[0]-.002 and aps>=baseline[3]-.003
                write(REPORT/'rgb_stability_gate.json',dict(passed=passed,criteria=criteria,
                    best_ap=best,last5_ap=ap,last5_aps=aps,baseline=baseline))
                if not passed:
                    check();save(status='stopped_rgb_stability_failed',pid=None,
                        reason='RGB对照未满足预设稳定性门槛，未启动local/aligned训练。')
                    return
        check();save(status='complete',pid=None)
    except Exception as exc:
        save(status='failed',pid=None,error=repr(exc));raise


if __name__=='__main__':main()
