"""Matched joint training: S4/S8 detail vs detail plus S16 strip context."""
import argparse
import json
from pathlib import Path
import torch
import train_sgc_multiscale_joint as joint
from preflight_sqer1 import seed_all, move_targets, sha256_file

ORIGINAL_SETUP=joint.setup
RUN=joint.task.engine.ROOT/'outputs/SBR3_CONTEXT_20261005'


class ContextTap(joint.task.FeatureTap):
    def forward(self,*args,**kwargs):
        result=super().forward(*args,**kwargs)
        self.cache['s16']=self.s16
        self.s16=None
        return result


class Source:
    def __init__(self,model,handle): self.model,self.handle=model,handle
    def __call__(self,samples):
        with torch.no_grad(): out=self.model(samples)
        cache=self.model.sqer.cache; self.model.sqer.cache={}
        return out,cache
    def close(self):
        self.handle.remove(); self.model.sqer.cache={}


def setup(batch,arm):
    cfg,source,refiner,criterion=ORIGINAL_SETUP(batch,'S8')
    base=source.model
    if arm=='CONTEXT':
        from src.zoo.dfine.sam_side_search_multiscale import SAMSideSearchContext
        new=SAMSideSearchContext(refiner,base.encoder.in_channels[0]).cuda()
        missing,unexpected=new.load_state_dict(refiner.state_dict(),strict=False)
        assert not unexpected and all(n.startswith(('context_projection.','context_score.')) for n in missing)
        refiner=new
    base.sqer=ContextTap()
    def capture(module,args): base.sqer.s16=args[0][0].detach()
    handle=base.encoder.register_forward_pre_hook(capture)
    return cfg,Source(base,handle),refiner,criterion


def readout(refiner,cache,out,hw):
    if hasattr(refiner,'context_projection'):
        return refiner(cache['s4'],cache['s8'],cache['s16'],cache['base_logits'],out['pred_boxes'],hw)
    return refiner(cache['s4'],cache['s8'],cache['base_logits'],out['pred_boxes'],hw)


def evaluate(cfg,source,refiner):
    return joint.task.engine.evaluate(cfg,source,refiner,readout_fn=readout)


def preflight(args):
    RUN.mkdir(parents=True,exist_ok=True)
    report={'status':'PASS','batch_size':args.batch_size,'arms':{},'sgc':'S8 fixed'}
    reference=None
    for arm in ('DETAIL','CONTEXT'):
        seed_all(0)
        cfg,source,refiner,criterion=setup(args.batch_size,arm)
        base=source.model
        samples,targets=next(iter(cfg.train_dataloader))
        samples=samples.cuda(); targets=move_targets(targets,torch.device('cuda'))
        base.eval(); refiner.eval()
        with torch.no_grad():
            out,cache=source(samples)
            prediction=readout(refiner,cache,out,samples.shape[-2:])['boxes'].cpu()
        if reference is None: reference=prediction
        else: assert torch.equal(reference,prediction),'Context initialization changes boxes'
        del out,cache
        joint.train_mode(base,refiner)
        opt=joint.optimizer(base,refiner)
        torch.cuda.reset_peak_memory_stats()
        for i in range(3):
            seed_all(100+i)
            opt.zero_grad(set_to_none=True)
            loss,logs=joint.step(base,refiner,criterion,samples,targets,0)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']],1.)
            assert torch.isfinite(loss) and torch.isfinite(norm)
            opt.step()
        grads={n:float(p.grad.norm()) for n,p in refiner.named_parameters() if p.grad is not None}
        if arm=='CONTEXT':
            for prefix in ('context_projection','context_score'):
                assert sum(v for n,v in grads.items() if n.startswith(prefix))>0
            refiner.eval(); base.eval()
            with torch.no_grad():
                out,cache=source(samples)
                normal=readout(refiner,cache,out,samples.shape[-2:])['boxes']
                cache['s16']=torch.zeros_like(cache['s16'])
                altered=readout(refiner,cache,out,samples.shape[-2:])['boxes']
                effect=float((normal-altered).abs().max())
                assert effect>0,'S16 has no effect'
        else: effect=None
        report['arms'][arm]={'loss':float(loss),'peak_gib':torch.cuda.max_memory_allocated()/1024**3,
                             's16_intervention_max_box_delta':effect,'refiner_gradients':grads}
        source.close()
        del cfg,source,refiner,criterion,base,opt,samples,targets,loss
        torch.cuda.empty_cache()
    report['init_refiner_sha']=sha256_file(joint.INIT)
    joint.task.engine.write_json(RUN/'preflight.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--preflight',action='store_true')
    p.add_argument('--arm',choices=['DETAIL','CONTEXT']); p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=16); args=p.parse_args()
    joint.RUN=RUN; joint.setup=setup
    joint.task.readout=readout; joint.task.evaluate=evaluate
    seed_all(0); torch.set_num_threads(4)
    if args.preflight: preflight(args)
    elif args.arm:
        joint.train(args,extra_protocol={'sgc_scales':[8], 'module':'SBR3_CONTEXT',
            'context_enabled':args.arm=='CONTEXT', 'context_width':32,'initial_context_score_zero':True,
            'experiment_source_hashes':{str(f):sha256_file(f) for f in [Path(__file__),
                joint.task.engine.REPO/'src/zoo/dfine/sam_side_search_multiscale.py']}})
    else: p.error('Specify --preflight or --arm')
