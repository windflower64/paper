"""Independent EMA evaluation and temporary relation/carrier interventions."""
import hashlib
import json
from pathlib import Path
import sys
import torch
REPO=Path(__file__).resolve().parents[1];ROOT=REPO.parent
sys.path.insert(0,str(REPO))
from src.core import YAMLConfig
from src.solver import TASKS
from src.solver.det_engine import evaluate

def main():
    arm=sys.argv[1];assert arm in ('sam','none','box')
    torch.set_num_threads(4)
    run=ROOT/f'outputs/S_BRA1_{arm.upper()}_B8A4_20E_TESTDEV/seed0'
    out=ROOT/'reports/119_sbra1/verification'/arm;out.mkdir(parents=True,exist_ok=True)
    for p in (run/'artifacts/src').rglob('*.py'):
        assert p.read_bytes()==(REPO/p.relative_to(run/'artifacts')).read_bytes(),str(p)
    cfg=YAMLConfig(str(REPO/f'experiments/phase_s/s_sbra1_{arm}_b8a4_20e.yml'),resume=str(run/'best_stg1.pth'),output_dir=str(out/'runtime'))
    solver=TASKS[cfg.yaml_cfg['task']](cfg);solver.eval()
    assert solver.ema is not None and solver.val_dataloader.dataset.sam_mask_root is None
    model=solver.ema.module.eval().requires_grad_(False)
    module=model.backbone.stages[2].sbra
    best=max([json.loads(l) for l in (run/'log.txt').read_text(encoding='utf-8').splitlines() if l.strip()],key=lambda r:r['test_coco_eval_bbox'][0])
    results={}
    for mode in (('normal',) if arm=='box' else ('normal','uniform','identity','carrier_off')):
        handle=None
        if mode=='uniform':handle=module.relation.register_forward_hook(lambda m,a,o:torch.zeros_like(o))
        elif mode=='identity':
            def identity(m,a,o):
                b,_,h,w=o.shape
                logits=torch.full((b,4,4,h,w),-80.,device=o.device,dtype=o.dtype)
                for i in range(4):logits[:,i,i]=0
                return logits.reshape_as(o)
            handle=module.relation.register_forward_hook(identity)
        elif mode=='carrier_off':handle=module.register_forward_hook(lambda m,a,o:a[1])
        try:
            metrics,_=evaluate(model,solver.criterion,solver.postprocessor,solver.val_dataloader,solver.evaluator,solver.device,epoch=-1,use_wandb=False)
        finally:
            if handle is not None:handle.remove()
        values=metrics['coco_eval_bbox'];results[mode]=values
        if mode=='normal':assert max(abs(a-b) for a,b in zip(values,best['test_coco_eval_bbox']))<1e-7
        result=dict(arm=arm,best_epoch=best['epoch'],images=len(solver.val_dataloader.dataset),checkpoint_sha256=hashlib.sha256((run/'best_stg1.pth').read_bytes()).hexdigest(),normal_exact=True,results=results,
                    caveat='Interventions are same-checkpoint causal probes, not retrained ablations; original test is development set')
        (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('VERIFIED',arm,mode,values[0],flush=True)

if __name__=='__main__':main()
