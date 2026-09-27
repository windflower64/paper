"""Joint optimization with a matched RGB control; native Windows."""
import argparse
import datetime
import json
from pathlib import Path
import torch
from run_mbudet_alignment import prepare as old_prepare, write, ROOT, REPO, RGB, THERMAL
from local_alignment_joint import JointDetector
from src.optim.ema import ModelEMA
from src.solver import TASKS
from src.misc import dist_utils

REPORT = ROOT/'reports/159_local_alignment_joint'
CONFIG = REPO/'experiments/phase_m/local_alignment_joint_30e.yml'


class JointEMA(ModelEMA):
    def update(self, model):
        super().update(model)
        src = dist_utils.de_parallel(model)
        fixed = {n for n, p in src.named_parameters() if not p.requires_grad}
        fixed.update(n for n, _ in src.named_buffers())
        state = src.state_dict()
        with torch.no_grad():
            for n, p in self.module.state_dict().items():
                if n in fixed:
                    p.copy_(state[n])


def prepare(arm, run, batch):
    cfg = old_prepare('aligned', run, batch, CONFIG)
    cfg._model = JointDetector(cfg._model.detector, arm)
    cfg.yaml_cfg['optimizer'] = dict(type='AdamW', foreach=False, lr=2e-4,
        betas=[.9, .999], weight_decay=1e-4, params=[
            dict(params=r'^detector\.(backbone|thermal_backbone)\..*$', lr=1e-5),
            dict(params=r'^detector\.(encoder|decoder)\..*$', lr=2e-5),
            dict(params=r'^alignment\..*$', lr=2e-4)])
    cfg._ema = JointEMA(cfg.model, **{k:v for k,v in cfg.yaml_cfg['ema'].items() if k != 'type'})
    return cfg


def preflight(batch, arm):
    cfg = prepare(arm, REPORT/'preflight_runtime', batch)
    model = cfg.model.cuda()
    loader = cfg.train_dataloader; loader.set_epoch(0)
    x, targets = next(iter(loader)); x = x.cuda()
    targets = [{k:v.cuda() if isinstance(v, torch.Tensor) else v for k,v in t.items()} for t in targets]
    assert len(loader.dataset)==3200 and len(cfg.val_dataloader.dataset)==1820
    model.eval()
    with torch.no_grad(), torch.autocast('cuda'):
        actual = model(x[:2])
        d = model.detector
        reference = d.decoder(d.encoder(d.backbone(x[:2,:3])))
    error = max(float((actual[k]-reference[k]).abs().max()) for k in ('pred_logits','pred_boxes'))
    assert error==0, error
    criterion, optimizer, scaler = cfg.criterion.cuda(), cfg.optimizer, cfg.scaler
    covered = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(covered)==len(set(covered))==sum(p.requires_grad for p in model.parameters())
    count=[]; optimizer.register_step_post_hook(lambda *_: count.append(1))
    ema=cfg.ema.to('cuda')
    model.train(); model.set_training_epoch(0)
    groups = ['detector.backbone.', 'detector.encoder.', 'detector.decoder.']
    if arm=='fusion':
        groups += ['detector.thermal_backbone.', 'alignment.0.coarse.', 'alignment.0.query.',
                   'alignment.0.refine.', 'alignment.0.gate.', 'alignment.0.output.']
    initial = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    seen = {g:False for g in groups}
    torch.cuda.reset_peak_memory_stats()
    batch_iterator=iter(loader)
    for step in range(6):
        fresh_x,fresh_targets=next(batch_iterator)
        x=fresh_x.cuda()
        targets=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in fresh_targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            out=model(x,targets)
        with torch.autocast('cuda',enabled=False):
            losses=criterion(out,targets,epoch=0,step=step,global_step=step,epoch_step=len(loader))
            loss=sum(losses.values())
        assert bool(torch.isfinite(loss))
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        for g in groups:
            seen[g] |= any(n.startswith(g) and p.grad is not None and bool(p.grad.abs().max()>0) for n,p in model.named_parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_max_norm)
        scaler.step(optimizer); scaler.update(); ema.update(model)
        print('PREFLIGHT',arm,batch,step,float(loss.detach()),flush=True)
    assert len(count)==6 and all(seen.values()), seen
    changed={g:any(n.startswith(g) and not torch.equal(initial[n],p.detach().cpu()) for n,p in model.named_parameters() if n in initial) for g in groups}
    assert all(changed.values()), changed
    return dict(status='PASS',arm=arm,batch=batch,initial_rgb_error=error,gradient_groups=seen,
        changed_groups=changed,real_updates=len(count),peak_gib=torch.cuda.max_memory_allocated()/2**30,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        fusion_parameters=sum(p.numel() for p in model.alignment.parameters()))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['preflight','train','eval'],required=True)
    p.add_argument('--arm',choices=['fusion','rgb'],default='fusion'); p.add_argument('--batch',type=int,default=16)
    p.add_argument('--run',type=Path); args=p.parse_args()
    if args.mode=='preflight':
        result=preflight(args.batch,args.arm)
        write(REPORT/f'preflight_{args.arm}_b{args.batch}.json',result); return
    cfg=prepare(args.arm,args.run,args.batch)
    if args.mode=='eval':
        from src.solver.det_engine import evaluate
        cfg.resume=str(args.run/'best_stg1.pth')
        solver=TASKS[cfg.yaml_cfg['task']](cfg); solver.eval()
        metrics,_=evaluate(solver.ema.module,solver.criterion,solver.postprocessor,
            solver.val_dataloader,solver.evaluator,solver.device,epoch=-1,use_wandb=False)
        rows=[json.loads(s) for s in (args.run/'log.txt').read_text().splitlines() if s.strip()]
        best=max(rows,key=lambda r:r['test_coco_eval_bbox'][0])
        error=max(abs(a-b) for a,b in zip(metrics['coco_eval_bbox'],best['test_coco_eval_bbox']))
        assert error<.0002, error
        write(args.run/'independent_best_ema.json',dict(status='PASS',metrics=metrics['coco_eval_bbox'],max_abs_error=error))
        return
    assert not (args.run/'log.txt').exists()
    write(args.run/'initialization.json',dict(arm=args.arm,rgb_source=str(RGB),thermal_source=str(THERMAL),
        batch=args.batch,effective_batch=32,epochs=30,trainable_parameters=sum(p.numel() for p in cfg.model.parameters() if p.requires_grad)))
    class ProgressSolver(TASKS[cfg.yaml_cfg['task']]):
        def train(self):
            super().train(); self.actual_updates=0
            def updated(*_):
                self.actual_updates+=1
                if self.actual_updates<=3 or self.actual_updates%10==0:
                    write(args.run/'training_progress.json',dict(status='training',epoch=self.model.epoch,
                        actual_optimizer_updates=self.actual_updates,updated_at=datetime.datetime.now().isoformat(),**self.model.telemetry))
            self.optimizer.register_step_post_hook(updated)
    solver=ProgressSolver(cfg); solver.fit()
    assert solver.actual_updates==3000, solver.actual_updates
    write(args.run/'training_progress.json',dict(status='complete',actual_optimizer_updates=solver.actual_updates))


if __name__=='__main__':
    main()
