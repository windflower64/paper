"""Immutable native-Windows queue: verified IR source, then two fusion arms."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('E:/two_paper'); REPO=ROOT/'D-FINE'
REPORT=ROOT/'reports/158_mbudet_rgb_alignment'; SNAPSHOT=REPORT/'frozen_repo'
IR_RUN=ROOT/'outputs/M_IR_RAW_VERIFIED_GQ1_30E_TESTDEV/seed0'
STATE={'completed':[]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(**values):
    STATE.update(values,updated_at=datetime.datetime.now().isoformat())
    (REPORT/'queue_status.json').write_text(json.dumps(STATE,ensure_ascii=False,indent=2),encoding='utf-8')


def check(manifest):
    for name,digest in manifest.items():
        assert sha(Path(name))==digest, f'Protected asset changed: {name}'


def execute(command,folder,name,**state):
    folder.mkdir(parents=True,exist_ok=True)
    env={k.upper():v for k,v in os.environ.items()}
    env.update(PYTHONIOENCODING='utf-8',PYTHONUTF8='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',CUDA_MODULE_LOADING='LAZY',TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='100')
    out=folder/(name+'_console.log'); err=folder/(name+'_error.log')
    with out.open('x',encoding='utf-8') as stdout,err.open('x',encoding='utf-8') as stderr:
        process=subprocess.Popen(command,cwd=SNAPSHOT,env=env,stdout=stdout,stderr=stderr,
            stdin=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        save(pid=process.pid,command=[str(c) for c in command],console_log=str(out),error_log=str(err),**state)
        code=process.wait()
    assert code==0, f'Exit {code}; inspect {err}'


def main():
    audit=json.loads((REPORT/'raw_ir_label_audit.json').read_text(encoding='utf-8'))
    preflight=json.loads((REPORT/'ir_preflight.json').read_text(encoding='utf-8'))
    pair=json.loads((REPORT/'corrected_training_pair_audit.json').read_text(encoding='utf-8'))
    assert audit['status']==preflight['status']==pair['status']=='PASS' and audit['bug_confirmed']
    assert not (REPORT/'queue_status.json').exists() and not IR_RUN.exists()
    assert not SNAPSHOT.exists() and shutil.disk_usage(ROOT).free>25*2**30
    SNAPSHOT.mkdir(); manifest={}
    for folder,pattern in [('src','*.py'),('configs','*.yml'),('experiments','*.yml'),('tools','*.py')]:
        for source in (REPO/folder).rglob(pattern):
            dest=SNAPSHOT/source.relative_to(REPO); dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest); manifest[str(dest)]=sha(dest)
            assert sha(source)==manifest[str(dest)]
    assets=[ROOT/'weights/dfine_n_coco.pth',
        ROOT/'outputs/S_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0/best_stg1.pth',
        REPORT/'raw_ir_label_audit.json', REPORT/'ir_preflight.json', REPORT/'corrected_training_pair_audit.json',
        *list((ROOT/'data/antiuav6k_ir_raw_verified').rglob('*.json')),
        *list((ROOT/'data/antiuav6k_ir_raw_verified').rglob('*.txt')),
        ROOT/'data/antiuav6k_common/annotations/instances_visible_common_train.json',
        ROOT/'data/antiuav6k_common/annotations/instances_visible_common_test.json']
    for asset in assets: manifest[str(asset)]=sha(asset)
    (REPORT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save(status='snapshot_complete',snapshot=str(SNAPSHOT),ir_batch=preflight['batch'],effective_batch=32)
    check(manifest)
    ir_command=[sys.executable,'-u',str(SNAPSHOT/'tools/run_verified_ir_pretraining.py'),
                '--batch',str(preflight['batch'])]
    IR_RUN.mkdir(parents=True,exist_ok=False)
    execute(ir_command+['--mode','train'],IR_RUN,'train',status='training',phase='verified_IR_pretraining',
        progress=str(IR_RUN/'training_progress.json'),metric_log=str(IR_RUN/'log.txt'))
    execute(ir_command+['--mode','eval'],IR_RUN,'evaluation',status='evaluating',phase='verified_IR_pretraining')
    assert json.loads((IR_RUN/'independent_best_ema.json').read_text())['status']=='PASS'
    STATE['completed'].append('verified_IR_pretraining')
    manifest[str(IR_RUN/'best_stg1.pth')]=sha(IR_RUN/'best_stg1.pth')
    (REPORT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    check(manifest)
    alignment=[sys.executable,'-u',str(SNAPSHOT/'tools/run_mbudet_alignment.py')]
    execute(alignment+['--mode','preflight'],REPORT,'corrected_alignment_preflight',
        status='preflight',phase='RGB_reference_feature_alignment')
    alignment_pf=json.loads((REPORT/'corrected_preflight.json').read_text(encoding='utf-8'))
    assert alignment_pf['status']=='PASS'; batch=alignment_pf['selected_batch']
    save(alignment_batch=batch)
    execute(alignment+['--mode','initial','--arm','aligned','--batch',str(batch),'--run',str(REPORT/'initial')],
        REPORT/'initial','validation',status='evaluating',phase='fixed_RGB_initial_AP')
    comparison={}
    for arm in ('unaligned','aligned'):
        check(manifest)
        run=ROOT/f'outputs/M_MBUDET_RGB_{arm.upper()}_B{batch}A{32//batch}_20E_TESTDEV/seed0'
        assert not run.exists(); run.mkdir(parents=True)
        command=alignment+['--arm',arm,'--batch',str(batch),'--run',str(run)]
        execute(command+['--mode','train'],run,'train',status='training',phase='fusion_comparison',arm=arm,
            progress=str(run/'training_progress.json'),metric_log=str(run/'log.txt'))
        execute(command+['--mode','eval'],run,'evaluation',status='evaluating',phase='fusion_comparison',arm=arm)
        rows=[json.loads(s) for s in (run/'log.txt').read_text().splitlines() if s.strip()]
        best=max(rows,key=lambda r:r['test_coco_eval_bbox'][0])
        comparison[arm]=dict(best_epoch=best['epoch'],best_metrics=best['test_coco_eval_bbox'],
            last3_mean=[sum(r['test_coco_eval_bbox'][i] for r in rows[-3:])/3 for i in range(12)],
            output_dir=str(run))
        (REPORT/'comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
        STATE['completed'].append(arm)
    check(manifest)
    initial=json.loads((REPORT/'initial/initial_verified.json').read_text(encoding='utf-8'))['metrics']
    lines=['# 显式目标对齐首轮训练结果','',
        f'正确框IR预训练30轮；融合两组各20轮，物理batch {batch}，有效batch 32。',
        f'固定C＋既有SAM的RGB底座AP：{initial[0]*100:.4f}。',
        '本轮没有新增SAM形状监督，不能据此归因新增SAM机制。','',
        '| 组别 | 最佳轮 | AP | AP50 | AP75 | APS |','|---|---:|---:|---:|---:|---:|']
    for arm,value in comparison.items():
        m=value['best_metrics']; lines.append(f"| {arm} | {value['best_epoch']} | {m[0]*100:.4f} | {m[1]*100:.4f} | {m[2]*100:.4f} | {m[3]*100:.4f} |")
    lines+=['','最佳点、末3轮与同权重旁路结果均保留。是否进入SAM形状约束阶段，需结合完整曲线和位置学习检查分析。']
    doc=ROOT/'knowledge/M系列知识库/09_显式目标对齐与SAM形状约束/03_首轮训练结果自动汇总.md'
    doc.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(status='complete',pid=None,phase='await_result_analysis',arm=None)


if __name__=='__main__':
    try: main()
    except Exception as error:
        save(status='failed',pid=None,error=str(error))
        raise
