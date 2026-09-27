"""Re-evaluate both SGC1 best EMA checkpoints without training labels."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]; ROOT=REPO.parent
REPORT=ROOT/'reports/105_sam_group_contrast_pilot'


def run(arm):
    import torch
    sys.path.insert(0,str(REPO))
    from src.core import YAMLConfig
    from src.solver import TASKS
    from src.solver.det_engine import evaluate
    torch.set_num_threads(4)
    run_dir=ROOT/f'outputs/S_SGC1_{arm.upper()}_B8A4_20E_TESTDEV/seed0'
    for p in (run_dir/'artifacts/src').rglob('*.py'):
        assert p.read_bytes()==(REPO/p.relative_to(run_dir/'artifacts')).read_bytes(),str(p)
    ckpt=run_dir/'best_stg1.pth'
    digest=hashlib.sha256(ckpt.read_bytes()).hexdigest()
    out=REPORT/'evaluation'/arm; out.mkdir(parents=True,exist_ok=True)
    destination=out/'best_ema.json'
    if destination.exists():
        assert json.loads(destination.read_text())['checkpoint_sha256']==digest
        return
    cfg=YAMLConfig(str(REPO/f'experiments/phase_s/s_sgc1_{arm}_b8a4_20e.yml'),
                   resume=str(ckpt),output_dir=str(out/'runtime'))
    solver=TASKS[cfg.yaml_cfg['task']](cfg); solver.eval()
    assert solver.val_dataloader.dataset.sam_mask_root is None
    assert len(solver.val_dataloader.dataset)==1820
    assert solver.ema is not None and solver.ema.module.sgc_enabled
    metrics,_=evaluate(solver.ema.module,solver.criterion,solver.postprocessor,
        solver.val_dataloader,solver.evaluator,solver.device,epoch=-1,use_wandb=False)
    values=metrics['coco_eval_bbox']
    logs=[json.loads(x) for x in (run_dir/'log.txt').read_text().splitlines() if x.strip()]
    best=max(logs,key=lambda r:r['test_coco_eval_bbox'][0])
    assert max(abs(a-b) for a,b in zip(values,best['test_coco_eval_bbox']))<1e-6
    result=dict(arm=arm,status='PASS',checkpoint_sha256=digest,weight_source='ema',
                best_epoch=best['epoch'],images=1820,coco_eval_bbox=values,
                all_metrics_match_log=True,validation_sam_cache=None)
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('RESULT',result,flush=True)


if __name__=='__main__':
    if sys.argv[1]=='all':
        for arm in ('sam','box'):
            with (REPORT/f'eval_{arm}.stdout.log').open('x',encoding='utf-8') as stdout, (REPORT/f'eval_{arm}.stderr.log').open('x',encoding='utf-8') as stderr:
                subprocess.run([sys.executable,'-u',str(Path(__file__)),arm],cwd=REPO,
                               stdout=stdout,stderr=stderr,check=True)
        (REPORT/'evaluation_complete.json').write_text(json.dumps(dict(complete=True,evaluations=2)),encoding='utf-8')
    else:
        assert sys.argv[1] in ('sam','box')
        run(sys.argv[1])
