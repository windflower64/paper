"""Fresh official COCO initialization, correctly labeled IR source training."""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import torch

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('E:/two_paper')
REPORT=ROOT/'reports/158_mbudet_rgb_alignment'
RUN=ROOT/'outputs/M_IR_RAW_VERIFIED_GQ1_30E_TESTDEV/seed0'
CONFIG=REPO/'experiments/phase_m/ir_raw_verified_gq1_30e.yml'
sys.path.insert(0,str(REPO))
from src.core import YAMLConfig
from src.misc import dist_utils
from src.solver import TASKS
from run_mbudet_alignment import write


def prepare(batch, run, config=CONFIG):
    torch.set_num_threads(4)
    dist_utils.setup_distributed(0,'builtin',seed=0)
    cfg=YAMLConfig(str(config),use_amp=True,output_dir=str(run),seed=0,
                   tuning=str(ROOT/'weights/dfine_n_coco.pth'),resume=None)
    cfg.yaml_cfg['train_dataloader']['total_batch_size']=batch
    cfg.yaml_cfg['gradient_accumulation_steps']=32//batch
    # BaseSolver applies tuning BEFORE constructing EMA; preflight uses same path.
    return cfg


def preflight(batch):
    cfg=prepare(batch,REPORT/'ir_preflight_runtime')
    solver=TASKS[cfg.yaml_cfg['task']](cfg); solver.train()
    loader=solver.train_dataloader; loader.set_epoch(0)
    x,targets=next(iter(loader)); assert x.shape==(batch,3,512,640)
    assert len(loader.dataset)==3200 and len(solver.val_dataloader.dataset)==1820
    assert 'raw_verified' in loader.dataset.ann_file and 'raw_verified' in solver.val_dataloader.dataset.ann_file
    x=x.cuda(); targets=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]
    updates=[]; solver.optimizer.register_step_post_hook(lambda *_:updates.append(True))
    solver.model.train(); rows=[]; torch.cuda.reset_peak_memory_stats()
    probe=solver.model.backbone.stem.stem1.conv.weight if hasattr(solver.model.backbone.stem,'stem1') else next(solver.model.parameters())
    before=probe.detach().clone()
    for step in range(2):
        solver.optimizer.zero_grad(set_to_none=True); start=time.perf_counter()
        with torch.autocast('cuda'):
            out=solver.model(x,targets)
        with torch.autocast('cuda',enabled=False):
            losses=solver.criterion(out,targets,epoch=0,step=step,global_step=step,epoch_step=len(loader))
            loss=sum(losses.values())
        assert bool(torch.isfinite(loss))
        solver.scaler.scale(loss).backward(); solver.scaler.unscale_(solver.optimizer)
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in solver.model.parameters())
        torch.nn.utils.clip_grad_norm_(solver.model.parameters(),cfg.clip_max_norm)
        solver.scaler.step(solver.optimizer); solver.scaler.update()
        torch.cuda.synchronize()
        rows.append(dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-start))
        print('REAL_IR_PREFLIGHT_UPDATE',batch,rows[-1],flush=True)
    assert len(updates)==2 and float((probe-before).abs().max())>0
    return dict(status='PASS',batch=batch,effective_batch=32,steps=rows,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        trainable_parameters=sum(p.numel() for p in solver.model.parameters() if p.requires_grad),
        source='official COCO, no old mislabeled IR weights',actual_updates=len(updates))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=('preflight','train','eval'),required=True)
    p.add_argument('--batch',type=int,default=32); p.add_argument('--config',type=Path,default=CONFIG)
    args=p.parse_args()
    assert json.loads((REPORT/'raw_ir_label_audit.json').read_text(encoding='utf-8'))['status']=='PASS'
    if args.mode=='preflight':
        assert not (REPORT/'ir_preflight.json').exists()
        for batch in (32,16,8):
            try:
                result=preflight(batch); break
            except torch.cuda.OutOfMemoryError as e:
                write(REPORT/f'ir_oom_b{batch}.json',dict(status='OOM',error=str(e)))
                torch.cuda.empty_cache()
        else: raise RuntimeError('IR pretraining OOM at every permitted batch')
        write(REPORT/'ir_preflight.json',result); return
    cfg=prepare(args.batch,RUN,args.config)
    if args.mode=='eval':
        cfg.tuning=None; cfg.resume=str(RUN/'best_stg1.pth')
        solver=TASKS[cfg.yaml_cfg['task']](cfg); solver.eval()
        from src.solver.det_engine import evaluate
        result,_=evaluate(solver.ema.module,solver.criterion,solver.postprocessor,
            solver.val_dataloader,solver.evaluator,solver.device,epoch=-1,use_wandb=False)
        rows=[json.loads(s) for s in (RUN/'log.txt').read_text().splitlines() if s.strip()]
        assert [r['epoch'] for r in rows]==list(range(30))
        best=max(rows,key=lambda r:r['test_coco_eval_bbox'][0]); values=result['coco_eval_bbox']
        error=max(abs(a-b) for a,b in zip(values,best['test_coco_eval_bbox']))
        assert error<.0002
        write(RUN/'independent_best_ema.json',dict(status='PASS',metrics=values,best_epoch=best['epoch'],max_abs_error=error))
    else:
        assert not (RUN/'log.txt').exists()
        write(RUN/'protocol.json',dict(batch=args.batch,accumulation=32//args.batch,effective_batch=32,
            epochs=30,init=str(ROOT/'weights/dfine_n_coco.pth'),annotation_root=str(ROOT/'data/antiuav6k_ir_raw_verified'),
            note='IR source training only; not final RGB-T AP, not comparable to old shifted IR protocol'))
        class ProgressSolver(TASKS[cfg.yaml_cfg['task']]):
            def train(self):
                super().train(); self.actual_updates=0
                parameter=next(p for p in self.model.backbone.parameters() if p.requires_grad)
                before=parameter.detach().flatten()[:128].clone()
                def updated(*_):
                    self.actual_updates+=1
                    if self.actual_updates<=3 or self.actual_updates%10==0:
                        result=dict(status='training',phase='correct_IR_source',actual_optimizer_updates=self.actual_updates,
                            batch=args.batch,effective_batch=32,updated_at=datetime.datetime.now().isoformat(),
                            backbone_probe_change=float((parameter.detach().flatten()[:128]-before).abs().max()))
                        write(RUN/'training_progress.json',result); print('OPTIMIZER_UPDATE',json.dumps(result),flush=True)
                self.optimizer.register_step_post_hook(updated)
        solver=ProgressSolver(cfg); solver.fit()
        assert solver.actual_updates==3000
        write(RUN/'training_complete.json',dict(status='PASS',actual_updates=solver.actual_updates))
    dist_utils.cleanup()


if __name__=='__main__': main()
