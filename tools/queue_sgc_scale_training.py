"""Frozen-source three-arm training queue. Stop on any failure, never resume silently."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO.parent
REPORT=ROOT/'reports/133_sgc_scale_training'
SNAPSHOT=REPORT/'frozen_repo'
INIT=ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth'
ARMS=('none','box','sam')
STATE={'completed':[]}


def save(**values):
    STATE.update(values,updated_at=datetime.datetime.now().isoformat())
    (REPORT/'queue_status.json').write_text(json.dumps(STATE,ensure_ascii=False,indent=2),encoding='utf-8')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run_process(command,stdout,stderr,**state):
    env=os.environ.copy()
    env.update(PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
               CUDA_MODULE_LOADING='LAZY',TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='100')
    with stdout.open('x',encoding='utf-8') as out,stderr.open('x',encoding='utf-8') as err:
        p=subprocess.Popen(command,cwd=REPO,env=env,stdout=out,stderr=err,
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        save(pid=p.pid,command=[str(x) for x in command],console_log=str(stdout),**state)
        code=p.wait()
    if code:raise RuntimeError(f'Process exited {code}: {stderr}')


def summarize():
    results={}
    for arm in ARMS:
        run=ROOT/f'outputs/SGC_SCALE_{arm.upper()}_B16A2_20E_TESTDEV/seed0'
        rows=[json.loads(s) for s in (run/'log.txt').read_text(encoding='utf-8').splitlines() if s.strip()]
        assert [r['epoch'] for r in rows]==list(range(20))
        assert all(len(r['test_coco_eval_bbox'])==12 and all(math.isfinite(v) for v in r['test_coco_eval_bbox']) for r in rows)
        best=max(rows,key=lambda r:r['test_coco_eval_bbox'][0])
        results[arm]={'best_epoch':best['epoch'],'best_metrics':best['test_coco_eval_bbox'],
          'last6_mean':[sum(r['test_coco_eval_bbox'][i] for r in rows[-6:])/6 for i in range(12)],
          'all_epochs':[{'epoch':r['epoch'],'metrics':r['test_coco_eval_bbox']} for r in rows]}
    (REPORT/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    lines=['# 尺度适配三臂训练自动汇总','',
      '三组均20轮、seed0、batch16累积2；原S8均为SAM。原test作为开发验证集，不是未见最终测试。',
      '各自总AP最佳轮与末6轮均值同时展示，不能将不同最佳轮的小目标差值当作全程稳定收益。','',
      '| S4补充 | 最佳轮 | AP | AP75 | 小目标AP | 末6轮AP | 末6轮小目标AP |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm,r in results.items():
        b=r['best_metrics'];t=r['last6_mean']
        lines.append(f"| {arm} | {r['best_epoch']} | {100*b[0]:.4f} | {100*b[2]:.4f} | {100*b[3]:.4f} | {100*t[0]:.4f} | {100*t[3]:.4f} |")
    lines+=['','独立最佳EMA复评保存在各臂evaluation/best_ema.json；完整12项COCO指标与逐轮曲线见comparison.json。',
      '这是自动数值汇总，不自动宣布SAM形状有效、边缘恢复成功或论文创新成立；不自动增加轮数或强度。']
    (REPORT/'三臂训练自动汇总.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    preflight=json.loads((REPORT/'preflight.json').read_text(encoding='utf-8'))
    assert preflight['status']=='PASS' and len(preflight['runs'])==3
    if (REPORT/'queue_status.json').exists():raise RuntimeError('Existing queue state; manual review required')
    for arm in ARMS:
        run=ROOT/f'outputs/SGC_SCALE_{arm.upper()}_B16A2_20E_TESTDEV/seed0'
        if run.exists():raise RuntimeError(f'Output already exists: {run}')
    assert shutil.disk_usage(ROOT).free>30*2**30,'Less than 30 GiB free'
    SNAPSHOT.mkdir(exist_ok=False)
    sources=[]
    for folder,pattern in (('src','*.py'),('configs','*.yml'),('experiments','*.yml'),('tools','*.py'),('tests','*.py')):
        sources+=list((REPO/folder).rglob(pattern))
    sources.append(REPO/'train.py')
    frozen={}
    for path in sources:
        dest=SNAPSHOT/path.relative_to(REPO)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        frozen[str(dest)]=sha(dest)
        assert sha(path)==frozen[str(dest)]
    masks=ROOT/'reports/104_sam3_role_control/masks_train'
    protected=[INIT,masks/'records.json',*list((masks/'masks').glob('*.png')),
        ROOT/'data/antiuav6k_common/annotations/instances_visible_common_train.json',
        ROOT/'data/antiuav6k_common/annotations/instances_visible_common_test.json']
    # 3043 positive-image files; only 2987 pass the existing quality gate.
    assert len(list((masks/'masks').glob('*.png')))==3043
    hashes={str(p):sha(p) for p in protected};hashes.update(frozen)
    (REPORT/'manifest.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    for arm in ARMS:
        for name,digest in hashes.items():
            assert sha(Path(name))==digest,f'Frozen input changed: {name}'
        run=ROOT/f'outputs/SGC_SCALE_{arm.upper()}_B16A2_20E_TESTDEV/seed0'
        run.mkdir(parents=True,exist_ok=False)
        config=SNAPSHOT/f'experiments/phase_s/sgc_scale_{arm}_b16a2_20e.yml'
        command=[sys.executable,'-u',str(SNAPSHOT/'train.py'),'-c',str(config),'-t',str(INIT),'--seed','0','--use-amp']
        run_process(command,run/'train_console.log',run/'train_error.log',status='training',arm=arm,
                    metric_log=str(run/'log.txt'),output_dir=str(run))
        rows=[json.loads(s) for s in (run/'log.txt').read_text(encoding='utf-8').splitlines() if s.strip()]
        assert [r['epoch'] for r in rows]==list(range(20))
        evaluation=REPORT/arm/'evaluation';evaluation.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,'-u',str(SNAPSHOT/'tools/eval_sgc_scale_training.py'),
                 '--snapshot',str(SNAPSHOT),'--config',str(config),'--run',str(run),
                 '--output',str(evaluation/'best_ema.json')]
        run_process(command,evaluation/'stdout.log',evaluation/'stderr.log',status='evaluating',arm=arm)
        STATE['completed'].append(arm);save(status='arm_complete',pid=None)
    summarize();save(status='complete',pid=None)


if __name__=='__main__':
    REPORT.mkdir(parents=True,exist_ok=True)
    if sys.argv[1:]==['--detach']:
        if (REPORT/'queue.lock').exists():raise RuntimeError('Existing queue lock')
        # Normalize duplicate Windows PATH/Path keys without changing system settings.
        env={k.upper():v for k,v in os.environ.items()}
        env['PYTHONIOENCODING']='utf-8'
        with (REPORT/'queue_python_stdout.log').open('x',encoding='utf-8') as out, (REPORT/'queue_python_stderr.log').open('x',encoding='utf-8') as err:
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve())],
                cwd=REPO,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),close_fds=True)
        print(json.dumps({'queue_pid':p.pid}),flush=True)
        sys.exit(0)
    # Exclusive lock remains as an audit trail, even after failure/completion.
    with (REPORT/'queue.lock').open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    try:main()
    except Exception as error:
        save(status='failed',pid=None,error=str(error));raise
