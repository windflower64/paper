"""SBR2 paired experiment: trainable local encoder, otherwise SBR1 protocol."""
import argparse
import json
from pathlib import Path
import time
import traceback
import torch
from torchvision.ops import box_convert,box_iou
import train_sbr1 as engine
from preflight_sqer1 import seed_all,move_targets,sha256_file

RUN=engine.ROOT/'outputs/SBR2_20261004'


class SourceV2:
    def __init__(self,model):
        self.model=model.eval().requires_grad_(False)
        self.cache={}
        self.handles=[model.sqer.register_forward_pre_hook(self.capture_inputs),
                      model.sqer.fusion.register_forward_hook(self.capture_memory)]
    def capture_inputs(self,module,args):
        self.cache.update(s4=args[0].detach(),s8=args[1].detach(),queries=args[2].detach(),base_logits=args[3].detach())
    def capture_memory(self,module,args,out): self.cache['memory']=out.detach()
    def __call__(self,samples):
        self.cache.clear()
        with torch.no_grad(): out=self.model(samples)
        cache=dict(self.cache); self.cache.clear()
        if set(cache)!={'s4','s8','queries','base_logits','memory'}: raise RuntimeError('Missing captured feature')
        return out,cache
    def close(self):
        for h in self.handles: h.remove()
        self.cache.clear()


def diagnostic_loss(result,boxes,targets,matches,shape_weight=0.05):
    from src.zoo.dfine.sam_boundary_refiner import refinement_loss
    loss,stats=refinement_loss(result,boxes,targets,matches,shape_weight)
    direction=[]; magnitude=[]; before=[]; after=[]
    hw=result['image_hw']
    with torch.no_grad():
        for b,(src,dst) in enumerate(matches):
            for q,t in zip(src.tolist(),dst.tolist()):
                if not (result['indices'][b]==q).any(): continue
                old=box_convert(boxes[b,q:q+1].float(),'cxcywh','xyxy')[0]
                new=box_convert(result['boxes'][b,q:q+1].float(),'cxcywh','xyxy')[0]
                gt=box_convert(targets[b]['boxes'][t:t+1].float(),'cxcywh','xyxy')[0]
                scale=old.new_tensor([hw[1],hw[0],hw[1],hw[0]])
                needed=(gt-old)*scale; actual=(new-old)*scale
                mask=(needed.abs()>.25)
                if mask.any(): direction.extend(((actual[mask]*needed[mask])>0).float().tolist())
                magnitude.extend(actual.abs().tolist())
                before.extend(needed.abs().tolist()); after.extend(((gt-new)*scale).abs().tolist())
    avg=lambda x:sum(x)/len(x) if x else 0.
    stats.update(positive_side_abs_shift_px=avg(magnitude),positive_direction_correct_fraction=avg(direction),
                 positive_side_error_before_px=avg(before),positive_side_error_after_px=avg(after))
    return loss,stats


def setup(batch):
    cfg,old_source,unused,lossfn=engine.setup(batch)
    model=old_source.model; old_source.close(); del unused
    from src.zoo.dfine.sam_boundary_refiner_v2 import SAMBoundaryRefinerV2
    state=torch.load(engine.RUN/'common_refiner_init.pth',map_location='cpu')
    refiner=SAMBoundaryRefinerV2(model.sqer,state).cuda()
    return cfg,SourceV2(model),refiner,diagnostic_loss


def readout(refiner,cache,out,hw):
    return refiner(cache['s4'],cache['s8'],cache['queries'],cache['base_logits'],out['pred_boxes'],hw)


def evaluate(cfg,source,refiner):
    return engine.evaluate(cfg,source,refiner,readout_fn=readout)


def preflight(args):
    cfg,source,refiner,lossfn=setup(args.batch_size)
    RUN.mkdir(parents=True,exist_ok=True)
    init=RUN/'common_refiner_init.pth'
    if init.exists(): refiner.load_state_dict(torch.load(init,map_location='cuda'),strict=True)
    else: engine.save_checkpoint(init,refiner.state_dict())
    original=engine.digest(source.model)
    samples,targets=next(iter(cfg.train_dataloader))
    samples=samples.cuda(); targets=move_targets(targets,torch.device('cuda'))
    out,cache=source(samples)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        encoded=refiner.encode(cache['s4'],cache['s8'])
        if not torch.equal(encoded,cache['memory']): raise AssertionError('Initial local features changed')
        result=readout(refiner,cache,out,samples.shape[-2:])
        if not torch.equal(result['boxes'],out['pred_boxes']): raise AssertionError('Initial boxes changed')
        # Parameters must be equal-valued copies, never aliases of the classifier.
        for name in ['s4_projection','s8_projection','fusion']:
            for a,b in zip(getattr(refiner,name).parameters(),getattr(source.model.sqer,name).parameters()):
                if a.data_ptr()==b.data_ptr(): raise AssertionError('Unexpected shared trainable storage')
    matches=engine.matching(cfg.criterion.matcher.cuda(),out,targets)
    checks={}
    for label,weight in [('GT_ONLY',0.0),('SAM',0.05)]:
        refiner.load_state_dict(torch.load(init,map_location='cuda'))
        opt=torch.optim.AdamW(refiner.parameters(),lr=1e-4,weight_decay=1e-4)
        steps=[]
        for step in range(4):
            opt.zero_grad(set_to_none=True)
            result=readout(refiner,cache,out,samples.shape[-2:])
            loss,stats=lossfn(result,out['pred_boxes'],targets,matches,shape_weight=weight)
            loss.backward()
            grads={name:float(p.grad.norm()) if p.grad is not None else 0. for name,p in refiner.named_parameters()}
            if not all(torch.isfinite(torch.tensor(v)) for v in grads.values()): raise AssertionError('Nonfinite gradient')
            norm=torch.nn.utils.clip_grad_norm_(refiner.parameters(),1)
            opt.step()
            steps.append({'step':step,'loss':float(loss),'stats':stats,'gradient_norm':float(norm),
                          'encoder_gradients':{k:v for k,v in grads.items() if k.startswith(('s4_projection','s8_projection','fusion'))}})
        for prefix in ['s4_projection','s8_projection','fusion','blocks.0','blocks.1']:
            if sum(v for k,v in grads.items() if k.startswith(prefix))<=0: raise AssertionError(f'No gradient in {prefix}')
        checks[label]=steps
    if engine.digest(source.model)!=original: raise AssertionError('Baseline changed')
    if any(p.grad is not None for p in source.model.parameters()): raise AssertionError('Baseline has gradients')
    peak=torch.cuda.max_memory_allocated()/1024**3
    del result,loss,opt,encoded,samples,targets,out,cache
    torch.cuda.empty_cache()
    refiner.load_state_dict(torch.load(init,map_location='cuda'))
    # Entire zero-initialized path, not just a bypass, must reproduce the baseline.
    baseline=evaluate(cfg,source,refiner)
    old=json.loads((engine.RUN/'preflight.json').read_text(encoding='utf-8'))['baseline']['coco_eval_bbox']
    error=max(abs(a-b) for a,b in zip(baseline['coco_eval_bbox'],old))
    if error>0.0002: raise AssertionError(f'Baseline reproduction failed {error}')
    report={'status':'PASS','batch_size':args.batch_size,'precision':'float32','peak_memory_gib':peak,
            'parameters':sum(p.numel() for p in refiner.parameters()),'base_sha':engine.BASE_SHA,
            'init_sha':sha256_file(init),'baseline':baseline,'baseline_max_abs_error':error,
            'initial_feature_identity':True,'independent_encoder_storage':True,'source_immutable':True,'gradient_checks':checks,
            'source_hashes':{str(p):sha256_file(p) for p in [Path(__file__),Path(engine.__file__),engine.REPO/'src/zoo/dfine/sam_boundary_refiner_v2.py',engine.REPO/'src/zoo/dfine/sam_boundary_refiner.py']}}
    engine.write_json(RUN/'preflight.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='gradient_checks'},ensure_ascii=False),flush=True)
    source.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--arm',choices=['R2_SAM','R1_GT_ONLY'])
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--epochs',type=int,default=30)
    args=parser.parse_args()
    seed_all(0); torch.set_num_threads(4)
    if args.preflight: preflight(args)
    elif args.arm:
        metadata={'precision':'float32','module':'SBR2','local_encoder_trainable':True,
                  'local_encoder_init':'exact copy of frozen SQER2 S4/S8 projection and fusion',
                  'feature_backbone_frozen':True,'source_hashes':{str(p):sha256_file(p) for p in [Path(__file__),Path(engine.__file__),engine.REPO/'src/zoo/dfine/sam_boundary_refiner_v2.py',engine.REPO/'src/zoo/dfine/sam_boundary_refiner.py']}}
        engine.train(args,setup_fn=setup,readout_fn=readout,evaluate_fn=evaluate,run_dir=RUN,extra_protocol=metadata)
    else: parser.error('Specify --preflight or --arm')


if __name__=='__main__':
    try: main()
    except Exception as exc:
        if RUN.exists(): engine.write_json(RUN/'status.json',{'state':'failed','error':repr(exc),'updated_at':time.time()})
        raise
