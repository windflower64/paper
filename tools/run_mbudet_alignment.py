"""Native Windows training/preflight for an isolated MBUDet-inspired adapter."""
import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('E:/two_paper')
REPORT = ROOT/'reports/158_mbudet_rgb_alignment'
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig
from src.misc import dist_utils
from src.optim.ema import ModelEMA
from src.solver import TASKS
from mbudet_interface import AlignedDetector, AlignmentCriterion, sample_flow, supervision

RGB = ROOT/'outputs/S_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0/best_stg1.pth'
THERMAL = ROOT/'outputs/M_IR_RAW_VERIFIED_GQ1_30E_TESTDEV/seed0/best_stg1.pth'
CONFIG = REPO/'experiments/phase_m/mbudet_rgb_reference_20e.yml'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(module):
    h = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fixed_hashes(model):
    return {name: digest(getattr(model.detector, name)) for name in
            ('backbone', 'encoder', 'decoder', 'thermal_backbone', 'thermal_encoder')}


class ExactFixedEMA(ModelEMA):
    def update(self, model):
        super().update(model)
        # Constant tensors must remain bit exact, not drift under EMA arithmetic.
        source = dist_utils.de_parallel(model).state_dict()
        with torch.no_grad():
            for name, tensor in self.module.state_dict().items():
                if name.startswith('detector.'):
                    tensor.copy_(source[name])


def prepare(arm, run, batch, config=CONFIG):
    assert arm in ('unaligned', 'aligned') and batch in (8, 16, 32)
    torch.set_num_threads(4)
    dist_utils.setup_distributed(0, 'builtin', seed=0)
    cfg = YAMLConfig(str(config), use_amp=True, output_dir=str(run), seed=0,
                     tuning=None, resume=None)
    cfg.yaml_cfg['train_dataloader']['total_batch_size'] = batch
    cfg.yaml_cfg['gradient_accumulation_steps'] = 32//batch
    cfg.yaml_cfg['val_dataloader']['total_batch_size'] = 8
    d = cfg.model
    state = torch.load(RGB, map_location='cpu')['ema']['module']
    for name in ('backbone', 'encoder', 'decoder'):
        getattr(d, name).load_state_dict({k[len(name)+1:]: v for k, v in state.items()
                                         if k.startswith(name+'.')}, strict=True)
    state = torch.load(THERMAL, map_location='cpu')['ema']['module']
    for name in ('thermal_backbone', 'thermal_encoder'):
        source = name.removeprefix('thermal_')
        getattr(d, name).load_state_dict({k[len(source)+1:]: v for k, v in state.items()
                                         if k.startswith(source+'.')}, strict=True)
    cfg._model = AlignedDetector(d, aligned=arm == 'aligned')
    cfg._criterion = AlignmentCriterion(cfg.criterion)
    cfg.yaml_cfg['optimizer'] = dict(type='AdamW', foreach=False, lr=2e-4,
        betas=[.9, .999], weight_decay=1e-4,
        params=[dict(params=r'^alignment\..*$', lr=2e-4)])
    cfg._ema = ExactFixedEMA(cfg.model, **{k: v for k, v in cfg.yaml_cfg['ema'].items() if k != 'type'})
    return cfg


def synthetic():
    source = torch.zeros(1, 1, 8, 10)
    source[0, 0, 5, 7] = 1
    flow = torch.zeros(1, 2, 8, 10, requires_grad=True)
    with torch.no_grad():
        flow[:, 0] = 4; flow[:, 1] = 3
    read = sample_flow(source, flow)
    assert abs(float(read[0, 0, 2, 3])-1) < 1e-6
    read.square().sum().backward()
    assert flow.grad is not None and bool(torch.isfinite(flow.grad).all())
    identity = sample_flow(source, torch.zeros_like(flow))
    assert float((identity-source).abs().max()) < 1e-6
    target = dict(boxes=torch.tensor([[.35, .3125, .1, .125]]),
                  infrared_boxes=torch.tensor([[.75, .6875, .1, .125]]))
    loss, _, count = supervision(flow.detach(), [target])
    assert float(loss) < 1e-10 and float(count) > 0
    empty = dict(boxes=torch.zeros(0, 4), infrared_boxes=torch.zeros(0, 4))
    loss, _, count = supervision(flow, [empty])
    assert float(loss) == 0 and float(count) == 0
    return dict(status='PASS', backward_sample_sign='IR minus RGB',
                flow_units='feature pixels', align_corners=False, empty_mask_finite=True)


def preflight(batch, arm):
    cfg = prepare(arm, REPORT/'preflight_runtime', batch)
    model = cfg.model.cuda()
    initial = fixed_hashes(model)
    loader = cfg.train_dataloader; loader.set_epoch(0)
    samples, targets = next(iter(loader))
    assert samples.shape == (batch, 6, 512, 640)
    assert len(loader.dataset) == 3200 and len(cfg.val_dataloader.dataset) == 1820
    assert loader.dataset.infrared_index_offset == 0 and loader.dataset.sam_mask_root is None
    assert cfg.val_dataloader.dataset.sam_mask_root is None and loader.collate_fn.scales is None
    samples = samples.cuda()
    targets = [{k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
    for t in targets:
        for name in ('boxes', 'infrared_boxes'):
            assert bool(torch.isfinite(t[name]).all())
            assert len(t[name]) <= 1
            if len(t[name]):
                assert bool((t[name] >= 0).all() and (t[name] <= 1).all()), (name, t[name])
    model.eval()
    with torch.no_grad(), torch.autocast('cuda'):
        initial_outputs = model(samples[:2])
        reference = model.detector(samples[:2])
    errors = {k: float((initial_outputs[k]-reference[k]).abs().max()) for k in ('pred_logits', 'pred_boxes')}
    assert max(errors.values()) == 0, errors
    criterion = cfg.criterion.cuda(); optimizer = cfg.optimizer; scaler = cfg.scaler
    covered = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(covered) == len(set(covered)) == sum(p.requires_grad for p in model.parameters())
    ema = cfg.ema.to('cuda'); real_updates = []
    optimizer.register_step_post_hook(lambda *_: real_updates.append(True))
    model.train(); model.set_training_epoch(0)
    assert not any(m.training for m in model.detector.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
    results = []; torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.autocast('cuda'):
            outputs = model(samples, targets)
        with torch.autocast('cuda', enabled=False):
            values = criterion(outputs, targets, epoch=0, step=step, global_step=step, epoch_step=len(loader))
            loss = sum(values.values())
        assert bool(torch.isfinite(loss))
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        assert all(p.grad is None for p in model.detector.parameters())
        offset_grads = sum(p.grad is not None and bool(p.grad.abs().max() > 0)
                           for level in model.alignment for p in level.offset.parameters())
        project_grads = sum(bool(level.projection.weight.grad.abs().max()>0) for level in model.alignment)
        assert project_grads == 2
        if arm == 'aligned':
            assert offset_grads > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
        scaler.step(optimizer); scaler.update(); ema.update(model)
        torch.cuda.synchronize()
        results.append(dict(step=step, loss=float(loss.detach()), offset_gradient_tensors=offset_grads,
                            seconds=time.perf_counter()-start, **model.telemetry))
        print('REAL_PREFLIGHT_UPDATE', arm, batch, json.dumps(results[-1]), flush=True)
    assert len(real_updates) == 3
    assert initial == fixed_hashes(model) == fixed_hashes(ema.module)
    return dict(status='PASS', arm=arm, physical_batch=batch, fixed_hashes=initial,
        initial_rgb_errors=errors, steps=results, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        adapter_parameters=sum(p.numel() for p in model.alignment.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        inputs_have_two_side_training_boxes=True, inference_has_no_gt=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('preflight', 'train', 'eval', 'initial'), required=True)
    p.add_argument('--arm', choices=('unaligned', 'aligned'), default='aligned')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--run', type=Path)
    p.add_argument('--config', type=Path, default=CONFIG)
    args = p.parse_args()
    if args.mode == 'preflight':
        assert not (REPORT/'corrected_preflight.json').exists()
        checks = synthetic()
        for batch in (32, 16, 8):
            try:
                runs = [preflight(batch, arm) for arm in ('unaligned', 'aligned')]
                break
            except torch.cuda.OutOfMemoryError as e:
                write(REPORT/f'oom_b{batch}.json', dict(status='OOM', error=str(e)))
                torch.cuda.empty_cache()
        else:
            raise RuntimeError('All permitted batches OOM')
        write(REPORT/'corrected_preflight.json', dict(status='PASS', selected_batch=batch, synthetic=checks, arms=runs))
        return
    cfg = prepare(args.arm, args.run, args.batch, args.config)
    from src.solver.det_engine import evaluate
    if args.mode in ('eval', 'initial'):
        if args.mode == 'eval':
            cfg.resume = str(args.run/'best_stg1.pth')
        solver = TASKS[cfg.yaml_cfg['task']](cfg); solver.eval()
        metrics, _ = evaluate(solver.ema.module, solver.criterion, solver.postprocessor,
            solver.val_dataloader, solver.evaluator, solver.device, epoch=-1, use_wandb=False)
        values = metrics['coco_eval_bbox']
        if args.mode == 'initial':
            rows = [json.loads(s) for s in (RGB.parent/'log.txt').read_text().splitlines() if s.strip()]
            expected = next(r['test_coco_eval_bbox'] for r in rows if r['epoch'] == 17)
        else:
            rows = [json.loads(s) for s in (args.run/'log.txt').read_text().splitlines() if s.strip()]
            assert [r['epoch'] for r in rows] == list(range(20))
            expected = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])['test_coco_eval_bbox']
        error = max(abs(a-b) for a, b in zip(values, expected))
        assert error < .0002, error
        write(args.run/f'{args.mode}_verified.json', dict(status='PASS', metrics=values, images=1820, max_abs_error=error))
        if args.mode == 'eval':
            solver.ema.module.bypass = True
            off, _ = evaluate(solver.ema.module, solver.criterion, solver.postprocessor,
                solver.val_dataloader, solver.evaluator, solver.device, epoch=-1, use_wandb=False)
            reference = json.loads((REPORT/'initial/initial_verified.json').read_text(encoding='utf-8'))['metrics']
            assert max(abs(a-b) for a,b in zip(off['coco_eval_bbox'], reference)) < .0002
            write(args.run/'adapter_bypass_verified.json', dict(status='PASS', metrics=off['coco_eval_bbox']))
    else:
        assert not (args.run/'log.txt').exists()
        write(args.run/'initialization.json', dict(arm=args.arm, rgb_source=str(RGB), thermal_source=str(THERMAL),
            fixed_hashes=fixed_hashes(cfg.model), batch=args.batch, effective_batch=32,
            trainable_parameters=sum(p.numel() for p in cfg.model.parameters() if p.requires_grad)))
        class ProgressSolver(TASKS[cfg.yaml_cfg['task']]):
            def train(self):
                super().train(); self.actual_updates = 0
                def updated(*_):
                    self.actual_updates += 1
                    if self.actual_updates <= 3 or self.actual_updates % 10 == 0:
                        result = dict(status='training', arm=args.arm, epoch=self.model.epoch,
                            actual_optimizer_updates=self.actual_updates, batch=args.batch,
                            updated_at=datetime.datetime.now().isoformat(), **self.model.telemetry)
                        write(args.run/'training_progress.json', result)
                        print('OPTIMIZER_UPDATE', json.dumps(result), flush=True)
                self.optimizer.register_step_post_hook(updated)
        solver = ProgressSolver(cfg); solver.fit()
        before = json.loads((args.run/'initialization.json').read_text(encoding='utf-8'))['fixed_hashes']
        assert before == fixed_hashes(solver.model) == fixed_hashes(solver.ema.module)
        assert solver.actual_updates == 2000
        write(args.run/'fixed_state_verified.json', dict(status='PASS', actual_updates=solver.actual_updates, hashes=before))
    dist_utils.cleanup()


if __name__ == '__main__':
    main()
