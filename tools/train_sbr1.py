"""Isolated SBR1 pilot. Frozen RGB-T baseline, paired initialization and sampler.

Uses hooks only to expose existing SQER inputs/memory; never patches baseline
forward or its criterion. All original weights remain immutable and eval-mode.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torchvision.ops import box_convert, box_iou
from preflight_sqer1 import build_config, move_targets, seed_all, sha256_file

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'D-FINE'
CHECKPOINT=ROOT/'outputs/M_LAYOUT_RECOVERY_20E_TESTDEV/R2_OTE2_LAYOUT/seed0/best_stg1.pth'
BASE_SHA='d7fc60568887e12ba66fbe5b474d09be5cfcc5dd4fe14bc102556b5e5b5aa952'
RUN=ROOT/'outputs/SBR1_20261003'


def write_json(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
    os.replace(temp,path)


def save_checkpoint(path,data):
    temp=path.with_suffix('.tmp')
    torch.save(data,temp); os.replace(temp,path)


def digest(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FrozenSource:
    def __init__(self,model):
        self.model=model.eval().requires_grad_(False)
        self.cache={}
        self.handles=[model.sqer.register_forward_pre_hook(self.capture_inputs),
                      model.sqer.fusion.register_forward_hook(self.capture_memory)]
    def capture_inputs(self,module,args):
        self.cache['queries']=args[2].detach()
        self.cache['base_logits']=args[3].detach()
    def capture_memory(self,module,args,out):
        self.cache['memory']=out.detach()
    def __call__(self,samples):
        self.cache.clear()
        with torch.no_grad(): out=self.model(samples)
        cache=dict(self.cache); self.cache.clear()
        assert set(cache)=={'queries','base_logits','memory'}
        return out,cache
    def close(self):
        for h in self.handles: h.remove()
        self.cache.clear()


def setup(batch):
    cfg=build_config(REPO,REPO/'experiments/phase_m/m_layout_recovery_r2_main.yml')
    for name in ['train_dataloader','val_dataloader']:
        dl=cfg.yaml_cfg[name]
        dl.update(total_batch_size=batch,num_workers=0,drop_last=False,shuffle=False,persistent_workers=False)
        dl['collate_fn']={'type':'BatchImageCollateFunction'}
    trans=cfg.yaml_cfg['train_dataloader']['dataset']['transforms']
    trans['ops']=[x for x in trans['ops'] if not x['type'].startswith('Random')]
    trans.pop('policy',None)
    if sha256_file(CHECKPOINT)!=BASE_SHA: raise RuntimeError('Baseline checkpoint changed')
    model=cfg.model.cuda().eval()
    saved=torch.load(CHECKPOINT,map_location='cpu')
    model.load_state_dict(saved['ema']['module'],strict=True)
    model.set_training_epoch(int(saved['last_epoch']))
    del saved
    from src.zoo.dfine.sam_boundary_refiner import SAMBoundaryRefiner,refinement_loss
    return cfg,FrozenSource(model),SAMBoundaryRefiner().cuda(),refinement_loss


def readout(refiner,cache,out,hw):
    return refiner(cache['memory'],cache['queries'],cache['base_logits'],out['pred_boxes'],hw)


def matching(matcher,out,targets):
    return matcher({'pred_logits':out['base_pred_logits'],'pred_boxes':out['pred_boxes']},targets)['indices']


@torch.no_grad()
def evaluate(cfg,source,refiner,baseline=False,limit=0,readout_fn=readout):
    refiner.eval()
    evaluator=cfg.evaluator; evaluator.cleanup()
    post=cfg.postprocessor.cuda().eval()
    coverage=[]; raw_iou=[]; new_iou=[]; image_count=0
    matcher=cfg.criterion.matcher.cuda()
    for bi,(samples,targets) in enumerate(cfg.val_dataloader):
        samples=samples.cuda(); targets=move_targets(targets,torch.device('cuda'))
        # FP32 baseline/evaluation exactly matches the historical evaluation protocol.
        out,cache=source(samples)
        final=dict(out)
        if not baseline:
            refined=readout_fn(refiner,cache,out,samples.shape[-2:])
            final['pred_boxes']=refined['boxes']
            if not torch.equal(final['pred_logits'],out['pred_logits']): raise AssertionError('Scores changed')
        size=torch.stack([t['orig_size'] for t in targets])
        predictions=post(final,size)
        evaluator.update({int(t['image_id'].item()):{k:v.cpu() for k,v in r.items()} for t,r in zip(targets,predictions)})
        normalized=[]
        for target in targets:
            t=dict(target)
            xy=target['boxes'].as_subclass(torch.Tensor).float()
            t['boxes']=box_convert(xy,'xyxy','cxcywh')/xy.new_tensor([samples.shape[-1],samples.shape[-2]]*2)
            normalized.append(t)
        matches=matching(matcher,out,normalized)
        for b,(src,dst) in enumerate(matches):
            if not len(src): continue
            g=box_convert(normalized[b]['boxes'][dst],'cxcywh','xyxy')
            old=torch.diag(box_iou(box_convert(out['pred_boxes'][b,src],'cxcywh','xyxy'),g))
            new=torch.diag(box_iou(box_convert(final['pred_boxes'][b,src],'cxcywh','xyxy'),g))
            raw_iou.extend(old.tolist()); new_iou.extend(new.tolist())
        image_count+=len(targets)
        if bi%25==0: print(f'eval images={image_count}',flush=True)
        if limit and image_count>=limit: break
    evaluator.synchronize_between_processes(); evaluator.accumulate(); evaluator.summarize()
    metrics=evaluator.coco_eval['bbox'].stats.tolist()
    paired={'targets':len(raw_iou),'mean_base_iou':float(np.mean(raw_iou)), 'mean_refined_iou':float(np.mean(new_iou)),
            'iou75_base':int(sum(x>=.75 for x in raw_iou)),'iou75_refined':int(sum(x>=.75 for x in new_iou)),
            'improved':int(sum(y>x+1e-6 for x,y in zip(raw_iou,new_iou))),
            'worsened':int(sum(y<x-1e-6 for x,y in zip(raw_iou,new_iou)))}
    return {'coco_eval_bbox':metrics,'paired_base_assignment':paired,'images':image_count}


def preflight(args):
    cfg,source,refiner,lossfn=setup(args.batch_size)
    initial_digest=digest(source.model)
    RUN.mkdir(parents=True,exist_ok=True)
    init=RUN/'common_refiner_init.pth'
    if init.exists(): refiner.load_state_dict(torch.load(init,map_location='cuda'),strict=True)
    else: save_checkpoint(init,refiner.state_dict())
    samples,targets=next(iter(cfg.train_dataloader))
    samples=samples.cuda(); targets=move_targets(targets,torch.device('cuda'))
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(): out,cache=source(samples)
    matcher=cfg.criterion.matcher.cuda(); matches=matching(matcher,out,targets)
    result=readout(refiner,cache,out,samples.shape[-2:])
    if not torch.equal(result['boxes'],out['pred_boxes']): raise AssertionError('Initialization not identical')
    opt=torch.optim.AdamW(refiner.parameters(),lr=1e-4)
    # Detection-only backward must reach all structural submodules by second update.
    steps=[]
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            result=readout(refiner,cache,out,samples.shape[-2:])
        with torch.autocast('cuda',enabled=False):
            loss,stats=lossfn(result,out['pred_boxes'],targets,matches,shape_weight=0)
        loss.backward()
        gradient={name:float(p.grad.float().norm()) if p.grad is not None else 0 for name,p in refiner.named_parameters()}
        if not all(math.isfinite(v) for v in gradient.values()): raise AssertionError('Nonfinite gradient')
        if step==2:
            for prefix in ['position','query_projection','roles','blocks.0','blocks.1','head']:
                if sum(v for k,v in gradient.items() if k.startswith(prefix))<=0: raise AssertionError(f'Dead gradient: {prefix}')
        torch.nn.utils.clip_grad_norm_(refiner.parameters(),1); opt.step()
        steps.append({'loss':float(loss),'stats':stats,'gradients':gradient})
    mask=torch.ones_like(result['indices'],dtype=torch.bool)
    selected=torch.zeros_like(out['pred_boxes'][...,0],dtype=torch.bool).scatter(1,result['indices'],mask)
    if not torch.equal(result['boxes'][~selected],out['pred_boxes'][~selected]): raise AssertionError('Unselected changed')
    originally_valid=(out['pred_boxes'][...,2:]>0).all(-1)
    if (result['boxes'][originally_valid][...,2:]<=0).any(): raise AssertionError('Refiner invalidated a valid box')
    if not torch.equal(result['boxes'][~originally_valid],out['pred_boxes'][~originally_valid]):
        raise AssertionError('Originally invalid boxes must remain unchanged')
    invalid_base_count=int((~originally_valid).sum())
    if any(p.grad is not None for p in source.model.parameters()): raise AssertionError('Frozen source has gradients')
    if digest(source.model)!=initial_digest: raise AssertionError('Source mutated')
    refiner.load_state_dict(torch.load(init,map_location='cuda'))
    del result,cache,out,loss,opt,samples,targets
    torch.cuda.empty_cache()
    # Full baseline reproduction: do not compare a subset against the full AP.
    baseline=evaluate(cfg,source,refiner,baseline=True)
    expected=0.5368043108
    if abs(baseline['coco_eval_bbox'][0]-expected)>0.0002: raise AssertionError(f'Baseline AP mismatch: {baseline}')
    report={'status':'PASS','base_checkpoint_sha256':BASE_SHA,'init_sha256':sha256_file(init),
            'batch_size':args.batch_size,'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3,
            'detection_gradient_steps':steps,'baseline':baseline,'source_immutable':True,'invalid_base_boxes_unchanged':invalid_base_count,
            'protocol':'frozen full RGB-T, original SQER scores, no shape teacher in forward'}
    write_json(RUN/'preflight.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    source.close()


def train(args,setup_fn=setup,readout_fn=readout,evaluate_fn=evaluate,run_dir=RUN,extra_protocol=None,sam_shape_weight=0.05):
    pre=json.loads((run_dir/'preflight.json').read_text(encoding='utf-8'))
    if pre['status']!='PASS' or pre['batch_size']!=args.batch_size: raise RuntimeError('Preflight missing/mismatched')
    arm=args.arm; shape_weight=sam_shape_weight if arm=='R2_SAM' else 0.0
    directory=run_dir/arm
    directory.mkdir(parents=True,exist_ok=True)
    cfg,source,refiner,lossfn=setup_fn(args.batch_size)
    initial_digest=digest(source.model)
    init=run_dir/'common_refiner_init.pth'
    refiner.load_state_dict(torch.load(init,map_location='cuda'),strict=True)
    optimizer=torch.optim.AdamW(refiner.parameters(),lr=1e-4,weight_decay=1e-4)
    ema=copy.deepcopy(refiner).eval().requires_grad_(False)
    start=0; best=-1.; checkpoint=directory/'last.pth'
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cuda')
        if state['base_sha']!=BASE_SHA or state['shape_weight']!=shape_weight: raise RuntimeError('Resume identity mismatch')
        refiner.load_state_dict(state['model']); ema.load_state_dict(state['ema']); optimizer.load_state_dict(state['optimizer'])
        start=state['epoch']+1; best=state['best_ap']
        existing=[json.loads(x) for x in (directory/'log.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()] if (directory/'log.jsonl').exists() else []
        if [r['epoch'] for r in existing]!=list(range(len(existing))): raise RuntimeError('Noncontiguous epoch log')
        if len(existing)==start-1 and 'last_row' in state:
            with (directory/'log.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(state['last_row'],ensure_ascii=False)+'\n')
        elif len(existing)!=start: raise RuntimeError('Checkpoint/log mismatch')
    elif (directory/'log.jsonl').exists(): raise RuntimeError('Log without resumable checkpoint')
    protocol={'arm':arm,'epochs':args.epochs,'batch_size':args.batch_size,'shape_weight':shape_weight,
              'base_sha':BASE_SHA,'init_sha':sha256_file(init),'seed':0,'lr':1e-4,'radius':.45,'topk':64,
              'roi_size':32,'frozen_baseline':True,'baseline_contains_old_SAM':True,'new_shape_teacher_only_difference':True}
    protocol.update(extra_protocol or {})
    if checkpoint.exists() and state['protocol']!=protocol:
        raise RuntimeError('Resume protocol differs from saved experiment')
    write_json(directory/'protocol.json',protocol)
    dataset=cfg.train_dataloader.dataset; collate=cfg.train_dataloader.collate_fn
    matcher=cfg.criterion.matcher.cuda()
    for epoch in range(start,args.epochs):
        if extra_protocol and 'epoch_random_seed_base' in extra_protocol:
            seed_all(int(extra_protocol['epoch_random_seed_base'])+epoch)
        refiner.train(); started=time.monotonic()
        generator=torch.Generator().manual_seed(10000+epoch)
        loader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=0,
                                           collate_fn=collate,generator=generator,drop_last=False)
        sums={}; count=0
        for step,(samples,targets) in enumerate(loader):
            progress=(epoch+step/len(loader))
            lr=1e-4*max((step+1)/len(loader),.01) if epoch==0 else 1e-6+(1e-4-1e-6)*.5*(1+math.cos(math.pi*(progress-1)/(args.epochs-1)))
            for group in optimizer.param_groups: group['lr']=lr
            samples=samples.cuda(); targets=move_targets(targets,torch.device('cuda'))
            # The frozen baseline is always FP32. Keep the small refiner in FP32
            # too: its four directional softmax maps are long (32x32), and
            # half-precision backward overflowed at epoch 0 step 140.
            out,cache=source(samples)
            matches=matching(matcher,out,targets)
            optimizer.zero_grad(set_to_none=True)
            result=readout_fn(refiner,cache,out,samples.shape[-2:])
            loss,stats=lossfn(result,out['pred_boxes'],targets,matches,shape_weight)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(refiner.parameters(),1)
            if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            with torch.no_grad():
                for ep,pp in zip(ema.parameters(),refiner.parameters()): ep.lerp_(pp,.01)
            count+=1
            for name,value in {'loss':float(loss.detach()),**stats}.items(): sums[name]=sums.get(name,0)+value
            if step%10==0:
                status={'state':'training','arm':arm,'epoch':epoch,'step':step,'steps':len(loader),'loss':float(loss.detach()),'lr':lr,'best_ap':best}
                status['updated_at']=time.time()
                write_json(run_dir/'status.json',status); print(json.dumps(status),flush=True)
            del out,cache,result,loss
        write_json(run_dir/'status.json',{'state':'validating','arm':arm,'epoch':epoch,'updated_at':time.time()})
        validation=evaluate_fn(cfg,source,ema)
        ap=validation['coco_eval_bbox'][0]; improved=ap>best; best=max(best,ap)
        row={'epoch':epoch,'train':{k:v/count for k,v in sums.items()},'validation':validation,'best_ap':best,'seconds':time.monotonic()-started}
        state={'epoch':epoch,'model':refiner.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
               'best_ap':best,'base_sha':BASE_SHA,'shape_weight':shape_weight,'protocol':protocol,'last_row':row}
        save_checkpoint(checkpoint,state)
        if improved: save_checkpoint(directory/'best.pth',state)
        with (directory/'log.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
    if digest(source.model)!=initial_digest: raise AssertionError('Baseline changed during training')
    write_json(directory/'complete.json',{'epochs':args.epochs,'best_ap':best,'source_immutable':True})
    write_json(run_dir/'status.json',{'state':'arm_complete','arm':arm,'best_ap':best,'updated_at':time.time()})
    source.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--preflight',action='store_true')
    p.add_argument('--arm',choices=['R1_GT_ONLY','R2_SAM']); p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--epochs',type=int,default=30)
    args=p.parse_args(); seed_all(0); torch.set_num_threads(4)
    if args.preflight: preflight(args)
    elif args.arm: train(args)
    else: p.error('Choose --preflight or --arm')
