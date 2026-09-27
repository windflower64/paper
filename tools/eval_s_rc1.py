"""Best EMA replication and selected spatial interventions after RC1 completion."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1];ROOT=REPO.parent
REPORT=ROOT/'reports/104_sam3_role_control'


def main(arm):
    import torch
    sys.path.insert(0,str(REPO))
    from src.core import YAMLConfig
    from src.solver import TASKS
    from src.solver.det_engine import evaluate
    assert arm in ['none','box','sam','edge']
    torch.set_num_threads(4)
    run=ROOT/f'outputs/S_RC1_{arm.upper()}_B8A4_20E_TESTDEV/seed0'
    output=REPORT/'interventions'/arm;output.mkdir(parents=True,exist_ok=True)
    # Check all inference source files, not just module class.
    for source in (run/'artifacts/src').rglob('*.py'):
        current=REPO/source.relative_to(run/'artifacts')
        assert current.read_bytes()==source.read_bytes(),str(current)
    checkpoint=run/'best_stg1.pth'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cfg=YAMLConfig(str(REPO/f'experiments/phase_s/s_rc1_{arm}_b8a4_20e.yml'),resume=str(checkpoint),output_dir=str(output/'runtime'))
    solver=TASKS[cfg.yaml_cfg['task']](cfg);solver.eval()
    assert solver.ema is not None
    model=solver.ema.module
    assert model.mfam is not None and solver.val_dataloader.dataset.sam_mask_root is None
    modes=['learned','shifted','disabled'] if arm in ['box','edge'] else ['learned']
    for mode in modes:
        dest=output/f'{mode}.json'
        if dest.exists():
            assert json.loads(dest.read_text())['checkpoint_sha256']==digest
            continue
        model.mfam.intervention='disabled' if mode=='disabled' else 'learned'
        hook=None
        if mode=='shifted':
            hook=model.mfam.mask_head.register_forward_hook(lambda m,i,o:o.roll((o.shape[-2]//2,o.shape[-1]//2),(-2,-1)))
        try:
            stats,_=evaluate(model,solver.criterion,solver.postprocessor,solver.val_dataloader,solver.evaluator,solver.device,epoch=-1,use_wandb=False)
        finally:
            if hook:hook.remove()
        metrics=stats['coco_eval_bbox']
        if mode=='learned':
            rows=[json.loads(x) for x in (run/'log.txt').read_text().splitlines() if x.strip()]
            expected=max(r['test_coco_eval_bbox'][0] for r in rows)
            assert abs(metrics[0]-expected)<1e-6,(expected,metrics[0])
        result=dict(arm=arm,mode=mode,checkpoint_sha256=digest,weight_source='ema',coco_eval_bbox=metrics)
        dest.write_text(json.dumps(result,indent=2),encoding='utf-8');print('RESULT',result,flush=True)


if __name__=='__main__':
    if sys.argv[1]=='all':
        for arm in ['none','box','sam','edge']:
            with (REPORT/f'eval_{arm}.stdout.log').open('x',encoding='utf-8') as out,(REPORT/f'eval_{arm}.stderr.log').open('x',encoding='utf-8') as err:
                subprocess.run([sys.executable,'-u',str(Path(__file__)),arm],cwd=REPO,stdout=out,stderr=err,check=True)
        (REPORT/'evaluation_complete.json').write_text(json.dumps(dict(complete=True,evaluations=8)),encoding='utf-8')
    else: main(sys.argv[1])
