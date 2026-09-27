"""Three arms with fixed backbones and trainable encoder/decoder, RGB stability first."""
import argparse
import datetime
import json
from pathlib import Path
import torch
from run_mbudet_alignment import prepare as base_prepare,write,digest,ROOT,REPO,RGB,THERMAL
from run_local_alignment_joint import JointEMA
from local_alignment_joint import JointDetector
from src.solver import TASKS
from src.solver.det_engine import evaluate

REPORT=ROOT/'reports/160_frozen_backbone_alignment'
CONFIG=REPO/'experiments/phase_m/local_alignment_joint_30e.yml'


class FixedBackboneDetector(JointDetector):
    def __init__(self,detector,arm):
        super().__init__(detector,'rgb' if arm=='rgb' else 'fusion')
        self.experiment_arm=arm
        self.detector.backbone.requires_grad_(False)
        self.detector.thermal_backbone.requires_grad_(False)
        if arm=='local':
            for level in self.alignment:
                level.coarse_enabled=False
                level.coarse.requires_grad_(False)

    def train(self,mode=True):
        super().train(mode)
        self.detector.backbone.eval();self.detector.thermal_backbone.eval()
        return self

    def forward(self,samples,targets=None):
        result=super().forward(samples,targets)
        if self.experiment_arm=='local':
            result.pop('alignment_aux_loss',None)
            self.telemetry.pop('field_loss',None)
            self.telemetry.pop('field_error_cells',None)
        return result


def fixed_hashes(model):
    return {name:digest(getattr(model.detector,name)) for name in ('backbone','thermal_backbone')}


def prepare(arm,run,batch=16):
    cfg=base_prepare('aligned',run,batch,CONFIG)
    cfg._model=FixedBackboneDetector(cfg.model.detector,arm)
    cfg.yaml_cfg['optimizer']=dict(type='AdamW',foreach=False,lr=2e-4,betas=[.9,.999],weight_decay=1e-4,
        params=[dict(params=r'^detector\.(encoder|decoder)\..*$',lr=2e-5),dict(params=r'^alignment\..*$',lr=2e-4)])
    cfg._ema=JointEMA(cfg.model,**{k:v for k,v in cfg.yaml_cfg['ema'].items() if k!='type'})
    return cfg


def preflight(arm):
    cfg=prepare(arm,REPORT/'preflight_runtime')
    model=cfg.model.cuda();before=fixed_hashes(model)
    loader=cfg.train_dataloader;loader.set_epoch(0)
    assert len(loader.dataset)==3200 and len(cfg.val_dataloader.dataset)==1820
    iterator=iter(loader);x,targets=next(iterator);x=x.cuda()
    model.eval()
    with torch.no_grad(),torch.autocast('cuda'):
        actual=model(x[:2]);d=model.detector
        expected=d.decoder(d.encoder(d.backbone(x[:2,:3])))
    errors={k:float((actual[k]-expected[k]).abs().max()) for k in ('pred_logits','pred_boxes')}
    assert max(errors.values())==0
    ema=cfg.ema.to('cuda');criterion=cfg.criterion.cuda();optimizer=cfg.optimizer;scaler=cfg.scaler
    ids=[id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(ids)==len(set(ids))==sum(p.requires_grad for p in model.parameters())
    groups=['detector.encoder.','detector.decoder.']
    for i in range(len(model.alignment)):
        groups += [f'alignment.{i}.{part}.' for part in ('query','refine','gate','output')]
        if arm=='aligned':groups.append(f'alignment.{i}.coarse.')
    snapshots={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    gradient_seen={g:False for g in groups};updates=[]
    optimizer.register_step_post_hook(lambda *_:updates.append(1))
    model.train();model.set_training_epoch(0);torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    for step in range(12):
        x,targets=next(iterator);x=x.cuda()
        targets=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]
        with torch.autocast('cuda'):out=model(x,targets)
        with torch.autocast('cuda',enabled=False):
            losses=criterion(out,targets,epoch=0,step=step,global_step=step,epoch_step=len(loader))
            loss=sum(losses.values())
        assert bool(torch.isfinite(loss))
        assert ('alignment_aux_loss' in out)==(arm=='aligned')
        scaler.scale(loss/2).backward()
        if step%2==1:
            scaler.unscale_(optimizer)
            assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            for g in groups:
                gradient_seen[g] |= any(n.startswith(g) and p.grad is not None and bool(p.grad.abs().max()>0) for n,p in model.named_parameters())
            assert all(p.grad is None for name in ('backbone','thermal_backbone') for p in getattr(model.detector,name).parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_max_norm)
            scaler.step(optimizer);scaler.update();ema.update(model);optimizer.zero_grad(set_to_none=True)
            print('PREFLIGHT_UPDATE',arm,len(updates),float(loss.detach()),flush=True)
    assert len(updates)==6 and all(gradient_seen.values()),gradient_seen
    assert before==fixed_hashes(model)==fixed_hashes(ema.module)
    changed={g:any(n.startswith(g) and not torch.equal(snapshots[n],p.detach().cpu()) for n,p in model.named_parameters() if n in snapshots) for g in groups}
    assert all(changed.values()),changed
    write(REPORT/f'preflight_{arm}.json',dict(status='PASS',batch=16,accumulation=2,actual_updates=6,
        fixed_hashes=before,initial_errors=errors,gradient_seen=gradient_seen,changed=changed,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        peak_gib=torch.cuda.max_memory_allocated()/2**30))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','initial','train','eval'],required=True)
    p.add_argument('--arm',choices=['rgb','local','aligned'],required=True);p.add_argument('--run',type=Path)
    args=p.parse_args()
    if args.mode=='preflight':preflight(args.arm);return
    cfg=prepare(args.arm,args.run)
    if args.mode in ('initial','eval'):
        if args.mode=='eval':cfg.resume=str(args.run/'best_stg1.pth')
        solver=TASKS[cfg.yaml_cfg['task']](cfg);solver.eval()
        metrics,_=evaluate(solver.ema.module,solver.criterion,solver.postprocessor,solver.val_dataloader,
            solver.evaluator,solver.device,epoch=-1,use_wandb=False)
        if args.mode=='initial':
            expected=json.loads((ROOT/'reports/158_mbudet_rgb_alignment/initial/initial_verified.json').read_text())['metrics']
        else:
            rows=[json.loads(s) for s in (args.run/'log.txt').read_text().splitlines() if s.strip()]
            assert [r['epoch'] for r in rows]==list(range(30))
            expected=max(rows,key=lambda r:r['test_coco_eval_bbox'][0])['test_coco_eval_bbox']
        error=max(abs(a-b) for a,b in zip(metrics['coco_eval_bbox'],expected));assert error<.0002,error
        write(args.run/f'{args.mode}_verified.json',dict(status='PASS',metrics=metrics['coco_eval_bbox'],max_abs_error=error))
        return
    assert not (args.run/'log.txt').exists()
    before=fixed_hashes(cfg.model)
    write(args.run/'initialization.json',dict(arm=args.arm,batch=16,effective_batch=32,epochs=30,
        rgb_source=str(RGB),thermal_source=str(THERMAL),fixed_hashes=before))
    class ProgressSolver(TASKS[cfg.yaml_cfg['task']]):
        def train(self):
            super().train();self.actual_updates=0
            def updated(*_):
                self.actual_updates+=1
                if self.actual_updates<=3 or self.actual_updates%10==0:
                    write(args.run/'training_progress.json',dict(status='training',epoch=self.model.epoch,
                        actual_optimizer_updates=self.actual_updates,updated_at=datetime.datetime.now().isoformat(),**self.model.telemetry))
            self.optimizer.register_step_post_hook(updated)
    solver=ProgressSolver(cfg);solver.fit()
    assert solver.actual_updates==3000
    assert before==fixed_hashes(solver.model)==fixed_hashes(solver.ema.module)
    write(args.run/'fixed_state_verified.json',dict(status='PASS',fixed_hashes=before,updates=3000))
    write(args.run/'training_progress.json',dict(status='complete',actual_optimizer_updates=3000))


if __name__=='__main__':main()
