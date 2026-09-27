"""Native Windows detector training/evaluation with training-only feature KD."""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('E:/two_paper')
sys.path.insert(0,str(REPO))
from tools.shape_feature_interface import FeatureDetector,FeatureCriterion,ARMS
from src.core import YAMLConfig
from src.misc import dist_utils
from src.solver import BaseSolver,TASKS


def prepare(config,arm,run,batch=16,log=True):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    cfg = YAMLConfig(str(config),use_amp=True,output_dir=str(run),seed=0)
    cfg.yaml_cfg['train_dataloader']['total_batch_size'] = batch
    cfg.yaml_cfg['val_dataloader']['total_batch_size'] = 16
    cfg.yaml_cfg['gradient_accumulation_steps'] = 32//batch
    assert batch in (8,16,32)
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    detector = cfg.model
    shim = BaseSolver.__new__(BaseSolver); shim.model = detector
    shim.load_tuning_state(str(ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth'))
    normalization = json.loads((ROOT/'reports/153_shape_feature_transfer/teacher/normalization.json').read_text(encoding='utf-8'))
    cfg._model = FeatureDetector(detector,arm,normalization,run/'feature_kd.jsonl' if log else None)
    cfg._criterion = FeatureCriterion(cfg.criterion)
    # Wrapper changes parameter names; preserve the anchored conditioner LR group.
    for group in cfg.yaml_cfg['optimizer']['params']:
        if group['params'].startswith('^sd2_conditioner'):
            group['params'] = '^detector.'+group['params'][1:]
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--eval',action='store_true')
    args = p.parse_args()
    dist_utils.setup_distributed(0,'builtin',seed=0)
    cfg = prepare(args.config,args.arm,args.run,args.batch,log=not args.eval)
    if args.eval:
        from src.solver.det_engine import evaluate
        cfg.resume = str(args.run/'best_stg1.pth')
        solver = TASKS[cfg.yaml_cfg['task']](cfg); solver.eval()
        assert len(solver.val_dataloader.dataset) == 1820
        assert solver.val_dataloader.dataset.sam_mask_root is None
        values,_ = evaluate(solver.ema.module,solver.criterion,solver.postprocessor,
            solver.val_dataloader,solver.evaluator,solver.device,epoch=-1,use_wandb=False)
        rows = [json.loads(s) for s in (args.run/'log.txt').read_text().splitlines() if s.strip()]
        assert [r['epoch'] for r in rows] == list(range(20))
        best = max(rows,key=lambda r:r['test_coco_eval_bbox'][0])
        metrics = values['coco_eval_bbox']
        error = max(abs(a-b) for a,b in zip(metrics,best['test_coco_eval_bbox']))
        assert error < .0002,error
        path = args.run/'independent_best_ema.json'
        if path.exists(): raise RuntimeError('Evaluation already exists')
        path.write_text(json.dumps(dict(status='PASS',images=1820,weight_source='EMA',
            best_epoch=best['epoch'],metrics=metrics,max_abs_error=error),indent=2),encoding='utf-8')
        # Raw detector weights can be loaded without teacher/projection/wrapper.
        torch.save({'model':solver.ema.module.detector.state_dict()},args.run/'detector_best_ema.pth')
    else:
        if (args.run/'log.txt').exists(): raise RuntimeError('Training output already exists')
        class ProgressSolver(TASKS[cfg.yaml_cfg['task']]):
            def train(self):
                super().train()
                self.actual_updates = 0
                first_parameter = next(self.model.detector.backbone.stages[2].parameters())
                self.initial_probe = first_parameter.detach().flatten()[:32].clone()
                def updated(optimizer,hook_args,hook_kwargs):
                    self.actual_updates += 1
                    if self.actual_updates <= 3 or self.actual_updates % 10 == 0:
                        changed = float((first_parameter.detach().flatten()[:32]-self.initial_probe).abs().max())
                        progress = dict(status='training',actual_optimizer_updates=self.actual_updates,
                            epoch=self.model.epoch,arm=args.arm,physical_batch=args.batch,
                            accumulation=32//args.batch,probe_weight_max_change=changed,
                            updated_at=datetime.datetime.now().isoformat())
                        (args.run/'training_progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
                        print('OPTIMIZER_UPDATE',json.dumps(progress),flush=True)
                self.optimizer.register_step_post_hook(updated)
        solver = ProgressSolver(cfg)
        solver.fit()
    dist_utils.cleanup()


if __name__ == '__main__':
    main()
