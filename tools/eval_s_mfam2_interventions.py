"""Separate the inference roles of support and conditional shape at fixed EMA."""
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.core import YAMLConfig
from src.solver import TASKS
from src.solver.det_engine import evaluate


def transform_logits(logits,mode):
    transformed=logits.clone()
    channel=0 if mode.startswith('support') else 1
    value=logits[:,channel:channel+1]
    if mode.endswith('shift'):
        value=value.roll((value.shape[-2]//2,value.shape[-1]//2),(-2,-1))
    elif mode.endswith('mean'):
        value=torch.logit(value.sigmoid().mean((-2,-1),keepdim=True).clamp(1e-6,1-1e-6)).expand_as(value)
    else:raise ValueError(mode)
    transformed[:,channel:channel+1]=value
    return transformed


def main():
    arm=sys.argv[1].upper();assert arm in ('SAM','BOX')
    torch.set_num_threads(8)
    run=ROOT.parent/f'outputs/S_MFAM2_C_{arm}_B8A4_20E_TESTDEV/seed0'
    output=ROOT.parent/f'reports/99_s_mfam2/interventions/{arm}'
    output.mkdir(parents=True,exist_ok=True)
    config=ROOT/f'experiments/phase_s/s_mfam2_c_{arm.lower()}_b8a4_20e_testdev_local.yml'
    checkpoint=run/'best_stg1.pth'
    for name in ('dfine.py','dfine_criterion.py','sam_support_shape.py','sam_mask_aggregation.py'):
        assert (ROOT/'src/zoo/dfine'/name).read_bytes()==(run/'artifacts'/name).read_bytes(),name
    cfg=YAMLConfig(str(config),resume=str(checkpoint),output_dir=str(output/'runtime'))
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    solver=TASKS[cfg.yaml_cfg['task']](cfg);solver.eval()
    model=solver.ema.module if solver.ema else solver.model
    assert model.mfam_variant=='support_shape'
    assert solver.val_dataloader.dataset.sam_mask_root is None
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for mode in ('learned','support_shift','shape_shift','shape_mean','disabled'):
        destination=output/f'{mode}.json'
        if destination.exists():
            assert json.loads(destination.read_text())['checkpoint_sha256']==digest
            continue
        model.mfam.intervention='disabled' if mode=='disabled' else 'learned'
        handle=None
        if mode not in ('learned','disabled'):
            handle=model.mfam.mask_head.register_forward_hook(lambda m,i,o:transform_logits(o,mode))
        try:
            stats,_=evaluate(model,solver.criterion,solver.postprocessor,solver.val_dataloader,
                             solver.evaluator,solver.device,epoch=-1,use_wandb=False)
        finally:
            if handle is not None:handle.remove()
        result=dict(arm=arm,mode=mode,checkpoint=str(checkpoint),checkpoint_sha256=digest,
                    weight_source='ema' if solver.ema else 'model',coco_eval_bbox=stats['coco_eval_bbox'])
        destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('MFAM2_RESULT',json.dumps(result),flush=True)


if __name__=='__main__':main()
