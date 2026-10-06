"""Matched S8 vs S4/S8/S16 fine-tuning, with SQER removed and SBR3 adapting."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
import train_sbr3_replace as task
from preflight_sqer1 import seed_all, move_targets, sha256_file

RUN = task.engine.ROOT/'outputs/SGC_MULTISCALE_JOINT_20261005'
INIT = task.RUN/'R2_SAM/best.pth'


def setup(batch, arm):
    cfg, source, refiner, _ = task.setup(batch)
    refiner.load_state_dict(torch.load(INIT, map_location='cpu')['ema'], strict=True)
    base = source.model
    base.sgc_enabled = True
    base.sgc_multiscale = arm == 'MULTISCALE'
    base.sgc_s4_rescue = 'none'
    base.sgc_decay_start = base.sgc_decay_end = -1
    base.sgc_aux_weight = 10.
    for module in (base.backbone, base.encoder, base.decoder):
        for name, parameter in module.named_parameters():
            if parameter.is_floating_point() and name not in ('up', 'reg_scale'):
                parameter.requires_grad_(True)
    criterion = cfg.criterion.cuda()
    criterion.sqer_shape_weight = 0.
    return cfg, source, refiner, criterion


def train_mode(base, refiner):
    base.train()
    # Frozen branches must not change normalization statistics.
    for name in ('thermal_backbone', 'thermal_encoder', 'mote_fusion'):
        module = getattr(base, name, None)
        if module is not None:
            module.eval()
    for module in base.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    refiner.train()


def step(base, refiner, criterion, samples, targets, epoch):
    out = base(samples, targets)
    cache = base.sqer.cache
    base.sqer.cache = {}
    assert torch.equal(out['pred_logits'], out['base_pred_logits'])
    losses = criterion(out, targets, epoch=epoch)
    matches = task.engine.matching(criterion.matcher, out, targets)
    result = task.readout(refiner, cache, out, samples.shape[-2:])
    geometry, stats = refiner.loss(result, out['pred_boxes'], targets, matches, shape_weight=1.)
    loss = sum(losses.values()) + geometry
    logs = {k:float(v.detach()) for k,v in losses.items()}
    logs.update(refiner_loss=float(geometry.detach()), shape_loss=stats['shape_loss'])
    if base.sgc_multiscale:
        logs.update({f'sgc_valid_s{s}': n for s,n in base.sgc_multiscale_stats['scale_eligible_images'].items()})
    return loss, logs


def optimizer(base, refiner):
    return torch.optim.AdamW([
        {'params':[p for p in base.parameters() if p.requires_grad], 'lr':1e-5, 'initial_lr':1e-5},
        {'params':list(refiner.parameters()), 'lr':5e-5, 'initial_lr':5e-5}], weight_decay=1e-4)


def preflight(args):
    RUN.mkdir(parents=True, exist_ok=True)
    report = {'status':'PASS', 'batch_size':args.batch_size, 'arms':{}}
    reference = None
    for arm in ('S8','MULTISCALE'):
        seed_all(0)
        cfg, source, refiner, criterion = setup(args.batch_size, arm)
        base = source.model
        digest = task.engine.digest(base) + task.engine.digest(refiner)
        if reference is None:
            reference = digest
            baseline = task.evaluate(cfg, source, refiner)
            report['initial_validation'] = baseline
        else:
            assert reference == digest, 'Initial weights differ'
        train_mode(base, refiner)
        frozen = {n:p.detach().clone() for n,p in base.named_parameters() if not p.requires_grad}
        opt = optimizer(base, refiner)
        torch.cuda.reset_peak_memory_stats()
        for i,(samples,targets) in enumerate(cfg.train_dataloader):
            samples = samples.cuda(); targets = move_targets(targets, torch.device('cuda'))
            opt.zero_grad(set_to_none=True)
            loss, logs = step(base, refiner, criterion, samples, targets, 0)
            assert torch.isfinite(loss)
            loss.backward()
            grads = {n:sum(float(p.grad.norm()) for p in m.parameters() if p.grad is not None)
                     for n,m in [('backbone',base.backbone),('encoder',base.encoder),('decoder',base.decoder),('refiner',refiner)]}
            assert all(v>0 and math.isfinite(v) for v in grads.values()), grads
            assert logs['loss_sgc_group']>0
            params = [p for g in opt.param_groups for p in g['params']]
            norm = torch.nn.utils.clip_grad_norm_(params,1.)
            assert torch.isfinite(norm)
            opt.step()
            if i == 1:
                break
        assert all(torch.equal(p, frozen[n]) for n,p in base.named_parameters() if n in frozen)
        report['arms'][arm] = {'gradient_norm_sums':grads, 'losses':logs,
                              'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                              'frozen_parameters_unchanged':True}
        source.close()
        del cfg, source, refiner, criterion, base, opt, params, frozen, loss, samples, targets
        torch.cuda.empty_cache()
    report['init_refiner_sha'] = sha256_file(INIT)
    task.engine.write_json(RUN/'preflight.json',report)
    print(json.dumps(report),flush=True)


def train(args, extra_protocol=None):
    pre = json.loads((RUN/'preflight.json').read_text(encoding='utf-8'))
    assert pre['status']=='PASS' and pre['batch_size']==args.batch_size
    assert pre['init_refiner_sha']==sha256_file(INIT)
    directory = RUN/args.arm
    directory.mkdir(parents=True,exist_ok=False)
    cfg, source, refiner, criterion = setup(args.batch_size,args.arm)
    base = source.model
    opt = optimizer(base,refiner)
    protocol = {'arm':args.arm,'epochs':args.epochs,'batch_size':args.batch_size,'precision':'FP32',
                'base_sha':task.engine.BASE_SHA,'refiner_init_sha':pre['init_refiner_sha'],
                'updated':['RGB backbone','RGB encoder','decoder','SBR3'],
                'frozen':['IR backbone','IR encoder','M parameters','BN running statistics'],
                'sgc_weight':10,'refiner_sam_weight':1,'ema':False,'seed':0,
                'source_hash':sha256_file(Path(__file__)),
                'dependency_hashes':{str(f):sha256_file(f) for f in [
                    task.engine.REPO/'src/zoo/dfine/dfine.py',
                    task.engine.REPO/'src/zoo/dfine/sam_multiscale_group.py',
                    task.engine.REPO/'src/zoo/dfine/sam_side_search.py']},
                'note':'matched fine-tuning, not scratch; no old SQER readout'}
    protocol.update(extra_protocol or {})
    task.engine.write_json(directory/'protocol.json',protocol)
    best=-1.
    for epoch in range(args.epochs):
        seed_all(20000+epoch)
        train_mode(base,refiner)
        loader=torch.utils.data.DataLoader(cfg.train_dataloader.dataset,batch_size=args.batch_size,
            shuffle=True,num_workers=0,collate_fn=cfg.train_dataloader.collate_fn,
            generator=torch.Generator().manual_seed(10000+epoch))
        sums={}; start=time.monotonic()
        for i,(samples,targets) in enumerate(loader):
            for g in opt.param_groups:
                g['lr']=g['initial_lr']*(.1+.9*.5*(1+math.cos(math.pi*epoch/args.epochs)))
            opt.zero_grad(set_to_none=True)
            loss, logs=step(base,refiner,criterion,samples.cuda(),move_targets(targets,torch.device('cuda')),epoch)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']],1.)
            if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradient')
            opt.step()
            for k,v in dict(loss=float(loss.detach()),**logs).items(): sums[k]=sums.get(k,0)+v
            if i%10==0:
                status={'state':'training','arm':args.arm,'epoch':epoch,'step':i,'steps':len(loader),'loss':float(loss.detach())}
                task.engine.write_json(RUN/'status.json',status); print(json.dumps(status),flush=True)
            del loss
        base.eval()
        validation=task.evaluate(cfg,source,refiner)
        row={'epoch':epoch,'train':{k:v/len(loader) for k,v in sums.items()},'validation':validation,'seconds':time.monotonic()-start}
        ap=validation['coco_eval_bbox'][0]
        state={'base':base.state_dict(),'refiner':refiner.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'protocol':protocol,'last_row':row}
        task.engine.save_checkpoint(directory/'last.pth',state)
        if ap>best:
            best=ap; task.engine.save_checkpoint(directory/'best.pth',state)
        with (directory/'log.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    task.engine.write_json(directory/'complete.json',{'epochs':args.epochs,'best_ap':best})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--preflight',action='store_true')
    p.add_argument('--arm',choices=['S8','MULTISCALE']); p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--epochs',type=int,default=20); args=p.parse_args()
    seed_all(0); torch.set_num_threads(4)
    if args.preflight: preflight(args)
    elif args.arm: train(args)
    else: p.error('Specify --preflight or --arm')
