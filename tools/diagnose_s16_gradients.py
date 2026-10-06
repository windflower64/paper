"""Read-only matched-batch gradient audit. No optimizer steps or checkpoint writes."""
import json
import math
from pathlib import Path
import torch
import train_sbr3_context as task
from preflight_sqer1 import seed_all, move_targets, sha256_file


def summarize(left, right):
    aa = sum(float(x.double().square().sum()) for x in left)
    bb = sum(float(x.double().square().sum()) for x in right)
    ab = sum(float((x.double()*y.double()).sum()) for x,y in zip(left,right))
    dd = sum(float((x.double()-y.double()).square().sum()) for x,y in zip(left,right))
    return dict(on_norm=math.sqrt(aa),off_norm=math.sqrt(bb),
                cosine=ab/math.sqrt(aa*bb) if aa*bb else None,
                difference_norm=math.sqrt(dd),relative_difference_to_off=math.sqrt(dd/bb) if bb else None)


def main():
    seed_all(0); torch.set_num_threads(4)
    cfg,source,refiner,criterion=task.setup(16,'CONTEXT')
    base=source.model
    checkpoint=task.RUN/'CONTEXT/best.pth'
    saved=torch.load(checkpoint,map_location='cpu')
    base.load_state_dict(saved['base'],strict=True)
    refiner.load_state_dict(saved['refiner'],strict=True)
    del saved
    task.joint.train_mode(base,refiner)
    initial=task.joint.task.engine.digest(base)+task.joint.task.engine.digest(refiner)
    named=[('base.'+n,p) for n,p in base.named_parameters() if p.requires_grad]
    named += [('refiner.'+n,p) for n,p in refiner.named_parameters() if p.requires_grad]
    groups={
        'base':[i for i,(n,p) in enumerate(named) if n.startswith('base.')],
        'backbone':[i for i,(n,p) in enumerate(named) if n.startswith('base.backbone.')],
        'decoder':[i for i,(n,p) in enumerate(named) if n.startswith('base.decoder.')],
        'shared_refiner':[i for i,(n,p) in enumerate(named) if n.startswith('refiner.') and not n.startswith(('refiner.context_score.','refiner.context_projection.'))],
        'context':[i for i,(n,p) in enumerate(named) if n.startswith(('refiner.context_score.','refiner.context_projection.'))],
    }
    loader=torch.utils.data.DataLoader(cfg.train_dataloader.dataset,batch_size=16,shuffle=True,
        num_workers=0,collate_fn=cfg.train_dataloader.collate_fn,generator=torch.Generator().manual_seed(10004))
    report={'checkpoint':str(checkpoint),'checkpoint_sha':sha256_file(checkpoint),
            'batch_size':16,'optimizer_steps':0,'epoch_for_criterion':4,'batches':[]}
    directory=task.joint.task.engine.ROOT/'reports/S16_GRADIENT_AUDIT_20261006'
    directory.mkdir(parents=True,exist_ok=True)
    for batch,(samples,targets) in enumerate(loader):
        if batch==3: break
        samples=samples.cuda(); targets=move_targets(targets,torch.device('cuda'))
        modes={}; gradients={}
        for mode in ('on','off'):
            seed_all(24000+batch)
            base.zero_grad(set_to_none=True); refiner.zero_grad(set_to_none=True)
            hook=refiner.context_score.register_forward_hook(lambda m,a,o:o*0) if mode=='off' else None
            try:
                out=base(samples,targets); cache=base.sqer.cache; base.sqer.cache={}
                losses=criterion(out,targets,epoch=4)
                matches=task.joint.task.engine.matching(criterion.matcher,out,targets)
                result=task.readout(refiner,cache,out,samples.shape[-2:])
                geometry,stats=refiner.loss(result,out['pred_boxes'],targets,matches,shape_weight=1.)
                # Explicitly test geometric-loss gradients to trainable detector parameters.
                bg=torch.autograd.grad(geometry,[p for n,p in named if n.startswith('base.')],retain_graph=True,allow_unused=True)
                bn=math.sqrt(sum(float(g.detach().double().square().sum()) for g in bg if g is not None))
                del bg
                detection=sum(losses.values()); loss=detection+geometry
                loss.backward()
                gradients[mode]=[p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p,device='cpu') for n,p in named]
                norm=math.sqrt(sum(float(g.double().square().sum()) for g in gradients[mode]))
                modes[mode]=dict(total_loss=float(loss.detach()),detector_loss=float(detection.detach()),
                    refiner_loss=float(geometry.detach()),refiner_loss_to_base_gradient_norm=bn,
                    total_gradient_norm=norm,global_clip_coefficient=min(1.,1./(norm+1e-6)),stats=stats)
                del out,cache,result,geometry,detection,loss,losses
            finally:
                if hook: hook.remove()
        comparison={group:summarize([gradients['on'][i] for i in indices],[gradients['off'][i] for i in indices]) for group,indices in groups.items()}
        report['batches'].append(dict(batch=batch,modes=modes,gradient_comparison=comparison))
        task.joint.task.engine.write_json(directory/'audit.json',report)
        print(json.dumps(report['batches'][-1]),flush=True)
        del gradients,samples,targets
    assert initial==task.joint.task.engine.digest(base)+task.joint.task.engine.digest(refiner),'Model state changed'
    report['state_unchanged']=True
    report['script_sha']=sha256_file(Path(__file__))
    task.joint.task.engine.write_json(directory/'audit.json',report)
    source.close()


if __name__=='__main__': main()
