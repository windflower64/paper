"""Privileged GT diagnostics, NEVER deployable AP or training selection claims."""
import json
import sys
from pathlib import Path

import torch

ROOT=Path('E:/two_paper'); REPORT=ROOT/'reports/158_mbudet_rgb_alignment'
SNAPSHOT=REPORT/'frozen_repo'
sys.path.insert(0,str(SNAPSHOT));sys.path.insert(0,str(SNAPSHOT/'tools'))
from run_mbudet_alignment import prepare,write,fixed_hashes
from src.solver import TASKS
from src.solver.det_engine import evaluate


class DiagnosticLoader:
    def __init__(self,loader,context): self.loader=loader;self.context=context
    def __len__(self): return len(self.loader)
    def __iter__(self):
        for samples,targets in self.loader:
            self.context['targets']=targets
            yield samples,targets


def region_and_truth(flow,targets):
    b,_,h,w=flow.shape
    region=torch.zeros((b,1,h,w),device=flow.device,dtype=flow.dtype)
    truth=torch.zeros_like(flow);eligible=[]
    for i,target in enumerate(targets):
        r=target['boxes'];t=target['infrared_boxes']
        valid=len(r)==len(t)==1;eligible.append(valid)
        if not valid: continue
        r=r[0].to(flow);t=t[0].to(flow)
        center=(r[:2]+r[2:])/2;wh=r[2:]-r[:2]
        scale=flow.new_tensor([w/640,h/512])
        extent=torch.maximum(wh*scale*1.5,flow.new_tensor([2.,2.]))
        lo=torch.floor(center*scale-extent/2).long().tolist()
        hi=torch.ceil(center*scale+extent/2).long().tolist()
        region[i,:,max(0,lo[1]):min(h,hi[1]),max(0,lo[0]):min(w,hi[0])]=1
        truth[i]=(((t[:2]+t[2:])/2-center)*scale)[:,None,None]
    return region,truth,eligible


def main():
    output=REPORT/'privileged_same_weight_diagnostics.json';assert not output.exists()
    trained=ROOT/'outputs/M_MBUDET_RGB_ALIGNED_B32A1_20E_TESTDEV/seed0'
    values={};context={}
    for mode in ('oracle_target_translation','foreground_region_only'):
        runtime=REPORT/'same_weight_diagnostics'/mode;runtime.mkdir(parents=True,exist_ok=False)
        cfg=prepare('aligned',runtime,32);cfg.resume=str(trained/'best_stg1.pth')
        solver=TASKS[cfg.yaml_cfg['task']](cfg);solver.eval()
        model=solver.ema.module;before=fixed_hashes(model);handles=[]
        for level in model.alignment:
            if mode=='oracle_target_translation':
                def offset_hook(module,inputs,flow):
                    mask,truth,_=region_and_truth(flow,context['targets'])
                    return torch.where(mask.bool(),truth,flow)
                handles.append(level.offset.register_forward_hook(offset_hook))
            else:
                def fusion_hook(module,inputs,outputs):
                    value,flow,increment=outputs
                    mask,_,eligible=region_and_truth(flow,context['targets'])
                    # Do not use GT absence to suppress negative/one-sided samples.
                    for i,valid in enumerate(eligible):
                        if not valid:mask[i]=1
                    return inputs[0]+increment*mask,flow,increment*mask
                handles.append(level.register_forward_hook(fusion_hook))
        result,_=evaluate(model,solver.criterion,solver.postprocessor,
            DiagnosticLoader(solver.val_dataloader,context),solver.evaluator,
            solver.device,epoch=-1,use_wandb=False)
        for handle in handles:handle.remove()
        assert before==fixed_hashes(model)
        baseline=json.loads((trained/'eval_verified.json').read_text(encoding='utf-8'))['metrics']
        values[mode]=dict(metrics=result['coco_eval_bbox'],
            minus_learned_percentage_points=[(a-b)*100 for a,b in zip(result['coco_eval_bbox'],baseline)])
        print('PRIVILEGED_DIAG',mode,json.dumps(values[mode]),flush=True)
    write(output,dict(status='PASS',checkpoint=str(trained/'best_stg1.pth'),epoch=3,images=1820,
        deployable=False,uses_test_GT_in_forward=True,trained_weights_unchanged=True,
        controls_negative_and_one_sided_samples_unchanged=True,results=values,
        limitation_zh='两项均用真实框决定局部处理，是特权机制诊断，不是合法部署AP、可实现增益承诺或SAM独有形状证据；两项单独作用，不构成完整因子实验。'))


if __name__=='__main__':main()
