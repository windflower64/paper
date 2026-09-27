"""Frozen four-arm queue, each training followed by independent best-EMA eval."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('E:/two_paper')
REPO=ROOT/'D-FINE'
BASE_REPORT=ROOT/'reports/153_shape_feature_transfer'
REPORT=BASE_REPORT
SNAPSHOT=REPORT/'frozen_repo'
ARMS=('none','uniform','filled','sam')
STATE={'completed':[]}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(**values):
    STATE.update(values,updated_at=datetime.datetime.now().isoformat())
    (REPORT/'queue_status.json').write_text(json.dumps(STATE,ensure_ascii=False,indent=2),encoding='utf-8')


def run_process(command,out,err,**values):
    env={k.upper():v for k,v in os.environ.items()}
    env.update(PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4',CUDA_MODULE_LOADING='LAZY',TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='100')
    with out.open('x',encoding='utf-8') as stdout,err.open('x',encoding='utf-8') as stderr:
        process=subprocess.Popen(command,cwd=REPO,env=env,stdout=stdout,stderr=stderr,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        save(pid=process.pid,command=[str(c) for c in command],console_log=str(out),**values)
        code=process.wait()
    if code:raise RuntimeError(f'Exit {code}; inspect {err}')


def main(preflight_path):
    preflight=json.loads(preflight_path.read_text(encoding='utf-8'))
    assert preflight['status']=='PASS'
    batch=preflight['selected_batch']
    assert batch in (16,32) and len(preflight['runs'])==4
    teacher=BASE_REPORT/'teacher'
    assert json.loads((teacher/'status.json').read_text())['status']=='complete'
    for arm in ARMS:
        if (ROOT/f'outputs/S_FEATURE_{arm.upper()}_B{batch}A{32//batch}_20E_TESTDEV/seed0').exists():
            raise RuntimeError('Existing output; will not overwrite or silently resume')
    assert shutil.disk_usage(ROOT).free>25*2**30
    SNAPSHOT.mkdir(exist_ok=False)
    files=[]
    for folder,pattern in (('src','*.py'),('configs','*.yml'),('experiments','*.yml'),('tools','*.py'),('tests','*.py')):
        files+=list((REPO/folder).rglob(pattern))
    frozen={}
    for source in files:
        dest=SNAPSHOT/source.relative_to(REPO)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        frozen[str(dest)]=sha(dest)
        assert sha(source)==frozen[str(dest)]
    protected=[ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth',
               teacher/'geometry.json',teacher/'normalization.json',teacher/'interface.json',
               ROOT/'reports/104_sam3_role_control/masks_train/records.json',
               ROOT/'data/antiuav6k_common/annotations/instances_visible_common_train.json',
               ROOT/'data/antiuav6k_common/annotations/instances_visible_common_test.json']
    protected+=list((ROOT/'reports/104_sam3_role_control/masks_train/masks').glob('*.png'))
    manifest={str(p):sha(p) for p in protected}
    feature_hashes=json.loads((teacher/'feature_hashes.json').read_text(encoding='utf-8'))
    for name,value in feature_hashes.items():
        path=Path(name)
        if not path.is_absolute():path=ROOT/path
        manifest[str(path.resolve())]=value
    manifest.update(frozen)
    (REPORT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    save(status='snapshot_complete',pid=None,batch=batch,accumulation=32//batch,
         effective_batch=32,preflight=str(preflight_path),snapshot=str(SNAPSHOT))
    results={}
    for arm in ARMS:
        for name,value in manifest.items():
            assert sha(Path(name))==value,f'Frozen asset changed: {name}'
        run=ROOT/f'outputs/S_FEATURE_{arm.upper()}_B{batch}A{32//batch}_20E_TESTDEV/seed0'
        run.mkdir(parents=True,exist_ok=False)
        config=SNAPSHOT/'experiments/phase_s/shape_feature_transfer_b16a2_20e.yml'
        command=[sys.executable,'-u',str(SNAPSHOT/'tools/run_shape_feature_transfer.py'),
                 '--config',str(config),'--arm',arm,'--run',str(run),'--batch',str(batch)]
        run_process(command,run/'train_console.log',run/'train_error.log',status='training',
                    arm=arm,output_dir=str(run),metric_log=str(run/'log.txt'),
                    progress=str(run/'training_progress.json'))
        rows=[json.loads(s) for s in (run/'log.txt').read_text().splitlines() if s.strip()]
        assert [r['epoch'] for r in rows]==list(range(20))
        telemetry=[json.loads(s) for s in (run/'feature_kd.jsonl').read_text().splitlines() if s.strip()]
        assert len(telemetry)==20*3200//batch
        assert all(t['weighted_loss']==0 for t in telemetry if t['epoch']>=10)
        assert all(t['weighted_loss']==0 for t in telemetry) if arm=='none' else any(t['weighted_loss']>0 for t in telemetry)
        run_process(command+['--eval'],run/'evaluation_console.log',run/'evaluation_error.log',
                    status='evaluating',arm=arm)
        best=max(rows,key=lambda r:r['test_coco_eval_bbox'][0])
        results[arm]=dict(best_epoch=best['epoch'],best_metrics=best['test_coco_eval_bbox'],
            last6_mean=[sum(r['test_coco_eval_bbox'][i] for r in rows[-6:])/6 for i in range(12)],
            all_epochs=[dict(epoch=r['epoch'],metrics=r['test_coco_eval_bbox']) for r in rows])
        (REPORT/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        STATE['completed'].append(arm);save(status='arm_complete',pid=None)
    lines=['# SAM形状选区特征迁移四臂自动汇总','',
        f'完整C+M半强度；20轮，seed0，batch{batch}，有效batch32。原test1820为开发集。',
        'FILLED含SAM外接范围，不是完全无SAM控制。以下为数值汇总，不自动宣称边缘恢复。','',
        '| 组别 | 最佳轮 | AP | AP75 | APS | 末6轮AP | 末6轮APS |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,r in results.items():
        b,t=r['best_metrics'],r['last6_mean']
        lines.append(f"| {arm} | {r['best_epoch']} | {100*b[0]:.4f} | {100*b[2]:.4f} | {100*b[3]:.4f} | {100*t[0]:.4f} | {100*t[3]:.4f} |")
    (REPORT/'四臂自动汇总.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(status='complete',pid=None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--attempt',type=int,default=1)
    p.add_argument('--detach',action='store_true');args=p.parse_args()
    if args.attempt<1:raise ValueError('Attempt must be positive')
    if args.attempt>1:
        REPORT=BASE_REPORT/f'attempt{args.attempt:02d}'
        SNAPSHOT=REPORT/'frozen_repo'
    REPORT.mkdir(parents=True,exist_ok=True)
    if args.detach:
        if (REPORT/'queue.lock').exists() or (REPORT/'queue_status.json').exists():
            raise RuntimeError('Existing queue lock/state')
        env={k.upper():v for k,v in os.environ.items()};env['PYTHONIOENCODING']='utf-8'
        with (REPORT/'queue_stdout.log').open('x',encoding='utf-8') as out,(REPORT/'queue_stderr.log').open('x',encoding='utf-8') as err:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                '--preflight',str(args.preflight.resolve()),'--attempt',str(args.attempt)],cwd=REPO,env=env,stdout=out,stderr=err,
                stdin=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        print(json.dumps({'queue_pid':process.pid}),flush=True)
    else:
        with (REPORT/'queue.lock').open('x',encoding='utf-8') as lock:lock.write(str(os.getpid()))
        try:main(args.preflight)
        except Exception as error:save(status='failed',pid=None,error=str(error));raise
