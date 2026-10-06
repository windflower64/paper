"""Paired M scale experiment; fixed S8 SGC and original S4/S8 SBR3."""
import argparse
import json
from pathlib import Path
import torch
import train_sgc_multiscale_joint as joint
from preflight_sqer1 import seed_all,move_targets,sha256_file

ORIGINAL_SETUP=joint.setup
ORIGINAL_MODE=joint.train_mode
RUN=joint.task.engine.ROOT/'outputs/M_P32_20261005'


def setup(batch,arm):
    cfg,source,refiner,criterion=ORIGINAL_SETUP(batch,'S8')
    base=source.model
    if arm=='P32':
        from src.zoo.dfine.thermal_p32_extension import ThermalP32Fusion
        base.mote_fusion=ThermalP32Fusion(base.mote_fusion,base.thermal_encoder.hidden_dim).cuda()
        def capture(module,args,out): base.mote_fusion.p32=out[-1].detach()
        handle=base.thermal_encoder.register_forward_hook(capture)
        oldclose=source.close
        def close():
            handle.remove(); base.mote_fusion.p32=None; oldclose()
        source.close=close
    base.mote_fusion.requires_grad_(True)
    return cfg,source,refiner,criterion


def train_mode(base,refiner):
    ORIGINAL_MODE(base,refiner)
    base.mote_fusion.train()


def preflight(args):
    RUN.mkdir(parents=True,exist_ok=True)
    report={'status':'PASS','batch_size':args.batch_size,'arms':{}}
    reference=None
    for arm in ('ORIGINAL','P32'):
        seed_all(0)
        cfg,source,refiner,criterion=setup(args.batch_size,arm)
        base=source.model
        samples,targets=next(iter(cfg.train_dataloader))
        samples=samples.cuda(); targets=move_targets(targets,torch.device('cuda'))
        base.eval(); refiner.eval()
        with torch.no_grad():
            out,cache=source(samples)
            boxes=joint.task.readout(refiner,cache,out,samples.shape[-2:])['boxes'].cpu()
            prediction=(boxes,out['pred_logits'].cpu())
        if reference is None: reference=prediction
        else: assert all(torch.equal(a,b) for a,b in zip(reference,prediction))
        del out,cache
        train_mode(base,refiner)
        opt=joint.optimizer(base,refiner)
        frozen={n:p.detach().clone() for n,p in base.named_parameters() if not p.requires_grad}
        torch.cuda.reset_peak_memory_stats()
        for i in range(3):
            seed_all(100+i)
            opt.zero_grad(set_to_none=True)
            loss,logs=joint.step(base,refiner,criterion,samples,targets,0)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']],1.)
            assert torch.isfinite(loss) and torch.isfinite(norm)
            opt.step()
        mg=sum(float(p.grad.norm()) for p in base.mote_fusion.parameters() if p.grad is not None)
        assert mg>0
        assert all(torch.equal(p,frozen[n]) for n,p in base.named_parameters() if n in frozen)
        effect=None
        if arm=='P32':
            assert base.mote_fusion.object_token_p32.weight.grad.norm()>0
            base.eval()
            with torch.no_grad():
                normal,_=source(samples)
                hook=base.mote_fusion.object_token_p32.register_forward_hook(lambda m,a,o:torch.zeros_like(o))
                altered,_=source(samples)
                hook.remove()
                effect=max(float((normal[k]-altered[k]).abs().max()) for k in ('pred_logits','pred_boxes'))
                assert effect>0,'P32 has no functional influence'
        report['arms'][arm]={'loss':float(loss),'losses':logs,'m_gradient':mg,'p32_intervention_max_delta':effect,
                              'peak_gib':torch.cuda.max_memory_allocated()/1024**3,'frozen_unchanged':True}
        source.close()
        del cfg,source,refiner,criterion,base,opt,frozen,samples,targets,loss
        torch.cuda.empty_cache()
    report['init_refiner_sha']=sha256_file(joint.INIT)
    joint.task.engine.write_json(RUN/'preflight.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true')
    p.add_argument('--arm',choices=['ORIGINAL','P32']);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=16);args=p.parse_args()
    joint.RUN=RUN;joint.setup=setup;joint.train_mode=train_mode
    seed_all(0);torch.set_num_threads(4)
    if args.preflight:preflight(args)
    elif args.arm:
        joint.train(args,extra_protocol={'module':'M_P32','sgc_scales':[8],'sbr_scales':[4,8],
            'ir_p32_enabled':args.arm=='P32','rgb_write_levels':['P16'],
            'updated':['RGB backbone','RGB encoder','decoder','SBR3','M'],
            'frozen':['IR backbone','IR encoder','BN running statistics'],
            'experiment_hashes':{str(f):sha256_file(f) for f in [Path(__file__),
                joint.task.engine.REPO/'src/zoo/dfine/thermal_p32_extension.py',
                joint.task.engine.REPO/'src/zoo/dfine/target_evidence_thermal_fusion.py']}})
    else:p.error('Specify preflight or arm')
