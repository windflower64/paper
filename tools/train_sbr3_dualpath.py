"""S16-assisted training with an independently supervised detail-only readout."""
import argparse
from pathlib import Path
import torch
import train_sbr3_context as context
from preflight_sqer1 import seed_all, move_targets, sha256_file

joint=context.joint
RUN=joint.task.engine.ROOT/'outputs/SBR3_DUALPATH_20261006'


def setup(batch,arm):
    return context.setup(batch,'CONTEXT')


def evaluate(cfg,source,refiner):
    hook=refiner.context_score.register_forward_hook(lambda m,a,o:torch.zeros_like(o))
    try: return context.evaluate(cfg,source,refiner)
    finally: hook.remove()


def step(base,refiner,criterion,samples,targets,epoch):
    out=base(samples,targets)
    cache=base.sqer.cache; base.sqer.cache={}
    losses=criterion(out,targets,epoch=epoch)
    matches=joint.task.engine.matching(criterion.matcher,out,targets)
    on=context.readout(refiner,cache,out,samples.shape[-2:])
    rng=torch.cuda.get_rng_state()
    # .5*(geometry_on + 2*shape) + .5*geometry_off: shape counted once.
    on_loss,on_stats=refiner.loss(on,out['pred_boxes'],targets,matches,shape_weight=2.)
    after=torch.cuda.get_rng_state()
    hook=refiner.context_score.register_forward_hook(lambda m,a,o:o*0)
    try:
        off=context.readout(refiner,cache,out,samples.shape[-2:])
        torch.cuda.set_rng_state(rng)
        off_loss,off_stats=refiner.loss(off,out['pred_boxes'],targets,matches,shape_weight=0.)
    finally:
        torch.cuda.set_rng_state(after)
        hook.remove()
    assert abs(on_stats['shape_loss']-off_stats['shape_loss'])<1e-6
    geometry=.5*(on_loss+off_loss)
    loss=sum(losses.values())+geometry
    logs={k:float(v.detach()) for k,v in losses.items()}
    logs.update(refiner_loss=float(geometry.detach()),shape_loss=on_stats['shape_loss'],
                geometry_on=float(on_loss.detach())-2*on_stats['shape_loss'],
                geometry_off=float(off_loss.detach()),
                paired_shape_difference=abs(on_stats['shape_loss']-off_stats['shape_loss']))
    return loss,logs


def preflight(args):
    RUN.mkdir(parents=True,exist_ok=True)
    cfg,source,refiner,criterion=setup(args.batch_size,'DUAL')
    base=source.model
    samples,targets=next(iter(cfg.train_dataloader))
    samples=samples.cuda(); targets=move_targets(targets,torch.device('cuda'))
    base.eval(); refiner.eval()
    with torch.no_grad():
        out,cache=source(samples)
        on=context.readout(refiner,cache,out,samples.shape[-2:])['boxes']
        hook=refiner.context_score.register_forward_hook(lambda m,a,o:o*0)
        try: off=context.readout(refiner,cache,out,samples.shape[-2:])['boxes']
        finally: hook.remove()
        assert torch.equal(on,off)
    del out,cache,on,off
    frozen={n:p.detach().cpu().clone() for n,p in base.named_parameters() if not p.requires_grad}
    joint.train_mode(base,refiner)
    # At zero context initialization the new loss equals the old objective.
    original_readout=joint.task.readout; joint.task.readout=context.readout
    seed_all(401)
    old,old_logs=ORIGINAL_STEP(base,refiner,criterion,samples,targets,0)
    old_value=float(old.detach()); del old
    seed_all(401)
    new,new_logs=step(base,refiner,criterion,samples,targets,0)
    delta=abs(float(new.detach())-old_value); del new
    assert delta<1e-5,delta
    joint.task.readout=original_readout
    opt=joint.optimizer(base,refiner)
    torch.cuda.reset_peak_memory_stats()
    steps=[]
    for i in range(3):
        seed_all(402+i); opt.zero_grad(set_to_none=True)
        loss,logs=step(base,refiner,criterion,samples,targets,0)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']],1.)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        grads={prefix:sum(float(p.grad.norm()) for n,p in refiner.named_parameters() if n.startswith(prefix) and p.grad is not None)
               for prefix in ('context_score','context_projection','score','shape_head','fusion')}
        opt.step(); steps.append(dict(loss=float(loss.detach()),norm=float(norm),gradients=grads,logs=logs))
        del loss
    assert all(v>0 for v in grads.values()),grads
    assert all(torch.equal(p.detach().cpu(),frozen[n]) for n,p in base.named_parameters() if n in frozen)
    report=dict(status='PASS',batch_size=args.batch_size,init_refiner_sha=sha256_file(joint.INIT),
                zero_init_objective_difference=delta,initial_on_off_boxes_equal=True,
                frozen_parameters_unchanged=True,peak_gib=torch.cuda.max_memory_allocated()/1024**3,steps=steps,
                script_sha=sha256_file(Path(__file__)))
    joint.task.engine.write_json(RUN/'preflight.json',report)
    print(report,flush=True); source.close()


ORIGINAL_STEP=joint.step

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--preflight',action='store_true')
    p.add_argument('--arm',choices=['DUAL']); p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=16); args=p.parse_args()
    seed_all(0); torch.set_num_threads(4)
    joint.RUN=RUN; joint.setup=setup; joint.step=step
    joint.task.evaluate=evaluate
    if args.preflight: preflight(args)
    elif args.arm:
        joint.train(args,extra_protocol=dict(module='SBR3_DUALPATH',sgc_scales=[8],sbr_scales=[4,8,16],
            inference='context_score output zero; detail-only readout',
            selection_metric='S16-disabled AP, fixed before training',
            objective='detector + .5*geometry_on + .5*geometry_off + SAM_shape',
            paired_synthetic_rng=True,baseline_box_gradient_preserved=True,
            initialization='same historical base and SBR3 SAM initialization as CONTEXT; not its best checkpoint',
            experiment_hashes={str(f):sha256_file(f) for f in [Path(__file__),Path(context.__file__),
                joint.task.engine.REPO/'src/zoo/dfine/sam_side_search.py',
                joint.task.engine.REPO/'src/zoo/dfine/sam_side_search_multiscale.py']}))
    else:p.error('Specify --preflight or --arm DUAL')
