"""Controlled mature-RGB retention experiment; native Windows, four arms."""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('E:/two_paper')
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig
from src.misc import dist_utils
from src.optim.ema import ModelEMA
from src.solver import TASKS

ARMS = ('sam_frozen', 'c_frozen', 'sam_joint', 'c_joint')
SOURCES = {
    'c': ROOT/'outputs/C_ONLY_GQ1_B8A4_20E_TESTDEV/seed0/best_stg1.pth',
    'sam': ROOT/'outputs/S_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0/best_stg1.pth',
}
THERMAL = ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth'
REPORT = ROOT/'reports/157_sam_rgb_retention_training'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class RetentionDetector(nn.Module):
    def __init__(self, detector, frozen):
        super().__init__()
        self.detector = detector
        self.frozen = frozen
        self.epoch = -1
        if frozen:
            for name in ('backbone', 'encoder', 'decoder'):
                getattr(detector, name).requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # Identical modes in BOTH frozen and joint arms; eval still supports grads.
        for name in ('backbone', 'encoder', 'thermal_backbone', 'thermal_encoder'):
            getattr(self.detector, name).eval()
        # Decoder stays train-mode for denoising and auxiliary detection losses.
        # DFINE.train already locks every BN statistic, including the decoder.
        return self

    def set_training_epoch(self, epoch):
        self.epoch = int(epoch)
        self.detector.set_training_epoch(epoch)

    def forward(self, samples, targets=None):
        return self.detector(samples, targets=targets)


class RetentionEMA(ModelEMA):
    def update(self, model):
        super().update(model)
        # Standard EMA arithmetic can drift even a mathematically constant float.
        # Preserve all fixed parameters AND buffers bit-exactly.
        raw = dist_utils.de_parallel(model)
        fixed = ('thermal_backbone.', 'thermal_encoder.')
        if raw.frozen:
            fixed += ('backbone.', 'encoder.', 'decoder.')
        source = raw.detector.state_dict()
        fixed_names = {n for n, p in raw.detector.named_parameters() if not p.requires_grad}
        fixed_names.update(n for n, _ in raw.detector.named_buffers())
        with torch.no_grad():
            for name, value in self.module.detector.state_dict().items():
                if name.startswith(fixed) or name in fixed_names:
                    value.copy_(source[name])


def prepare(config, arm, run, batch):
    assert arm in ARMS and batch in (8, 16, 32)
    torch.set_num_threads(4)
    dist_utils.setup_distributed(0, 'builtin', seed=0)
    cfg = YAMLConfig(str(config), use_amp=True, output_dir=str(run), seed=0,
                     tuning=None, resume=None)
    cfg.yaml_cfg['train_dataloader']['total_batch_size'] = batch
    cfg.yaml_cfg['gradient_accumulation_steps'] = 32//batch
    cfg.yaml_cfg['val_dataloader']['total_batch_size'] = 8
    detector = cfg.model
    source_name = arm.split('_')[0]
    checkpoint = torch.load(SOURCES[source_name], map_location='cpu')
    state = checkpoint['ema']['module']
    assert checkpoint['last_epoch'] == 17
    for name in ('backbone', 'encoder', 'decoder'):
        part = {k[len(name)+1:]: v for k, v in state.items() if k.startswith(name+'.')}
        getattr(detector, name).load_state_dict(part, strict=True)
    thermal = torch.load(THERMAL, map_location='cpu')['model']
    for name in ('thermal_backbone', 'thermal_encoder'):
        part = {k[len(name)+1:]: v for k, v in thermal.items() if k.startswith(name+'.')}
        getattr(detector, name).load_state_dict(part, strict=True)
    frozen = arm.endswith('_frozen')
    cfg._model = RetentionDetector(detector, frozen)
    groups = [{'params': r'^detector\.sd2_conditioner\..*$', 'lr': 2e-4}]
    if not frozen:
        groups += [
            {'params': r'^detector\.backbone\.(?!.*norm|.*bn).*$', 'lr': 1e-5},
            {'params': r'^detector\.backbone\.(?=.*norm|.*bn).*$', 'lr': 1e-5, 'weight_decay': 0.0},
            {'params': r'^detector\.(encoder|decoder)\.(?!.*norm|.*bn|.*bias).*$', 'lr': 2e-5},
            {'params': r'^detector\.(encoder|decoder)\.(?=.*norm|.*bn|.*bias).*$', 'lr': 2e-5, 'weight_decay': 0.0},
        ]
    cfg.yaml_cfg['optimizer'] = dict(type='AdamW', foreach=False, params=groups,
        lr=2e-4, betas=[0.9, 0.999], weight_decay=1e-4)
    settings = {k: v for k, v in cfg.yaml_cfg['ema'].items() if k != 'type'}
    cfg._ema = RetentionEMA(cfg.model, **settings)
    return cfg


def snapshots(model):
    d = model.detector
    return {n: digest(getattr(d, n)) for n in
            ('backbone', 'encoder', 'decoder', 'thermal_backbone', 'thermal_encoder', 'sd2_conditioner')}


def preflight_one(config, arm, batch):
    import time
    cfg = prepare(config, arm, REPORT/'preflight_runtime', batch)
    model = cfg.model.cuda()
    before = snapshots(model)
    loader = cfg.train_dataloader
    loader.set_epoch(0)
    samples, targets = next(iter(loader))
    assert samples.shape == (batch, 6, 512, 640)
    assert len(loader.dataset) == 3200 and loader.collate_fn.scales is None
    assert len(cfg.val_dataloader.dataset) == 1820
    assert cfg.val_dataloader.dataset.sam_mask_root is None
    samples = samples.cuda()
    targets = [{k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
    ids = [int(t['image_id'][0]) for t in targets]
    model.eval()
    with torch.no_grad(), torch.autocast('cuda'):
        a = model(samples[:2])
        bypass = model.detector.sd2_conditioner.register_forward_hook(
            lambda module, inputs, output: inputs[0])
        b = model(samples[:2])
        bypass.remove()
    errors = {k: float((a[k]-b[k]).abs().max()) for k in ('pred_boxes', 'pred_logits')}
    assert max(errors.values()) == 0, errors
    model.train(); model.set_training_epoch(0)
    assert model.detector.decoder.training and not model.detector.encoder.training
    assert not any(m.training for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
    optimizer = cfg.optimizer
    covered = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(covered) == len(set(covered)) == sum(p.requires_grad for p in model.parameters())
    actual = []
    optimizer.register_step_post_hook(lambda *args: actual.append(True))
    criterion = cfg.criterion.cuda(); scaler = cfg.scaler
    ema = cfg.ema.to('cuda')
    losses = []; seconds = []; m_grad_counts = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.autocast('cuda'):
            outputs = model(samples, targets)
        with torch.autocast('cuda', enabled=False):
            values = criterion(outputs, targets, epoch=0, step=step, global_step=step, epoch_step=len(loader))
            loss = sum(values.values())
        assert torch.isfinite(loss)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        assert all(p.grad is None for p in model.detector.thermal_backbone.parameters())
        if model.frozen:
            assert all(p.grad is None for p in model.detector.backbone.parameters())
        m_grad_counts.append(sum(p.grad is not None and bool(p.grad.abs().max()>0)
            for p in model.detector.sd2_conditioner.parameters()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_max_norm)
        scaler.step(optimizer); scaler.update(); ema.update(model)
        torch.cuda.synchronize(); seconds.append(time.perf_counter()-start)
        losses.append(float(loss.detach()))
        print('REAL_UPDATE', arm, batch, step, losses[-1], seconds[-1], flush=True)
    assert len(actual) == 2 and m_grad_counts[-1] > 0
    after = snapshots(model)
    ema_after = snapshots(ema.module)
    for n in ('thermal_backbone', 'thermal_encoder'):
        assert before[n] == after[n] == ema_after[n]
    if model.frozen:
        for n in ('backbone', 'encoder', 'decoder'):
            assert before[n] == after[n] == ema_after[n]
    else:
        assert any(before[n] != after[n] for n in ('backbone', 'encoder', 'decoder'))
    assert before['sd2_conditioner'] != after['sd2_conditioner']
    result = dict(arm=arm, batch=batch, initial=before, after=after, inference_errors=errors,
        actual_updates=len(actual), losses=losses, m_gradient_parameters=m_grad_counts,
        seconds=seconds, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        input_ids=ids, frozen_state_exact=True, normalization_stats_fixed=True)
    return result


def validate_initial(config, source, batch):
    from src.solver.det_engine import evaluate
    cfg = prepare(config, source+'_frozen', REPORT/'initial_validation'/source, batch)
    solver = TASKS[cfg.yaml_cfg['task']](cfg); solver.eval()
    values, _ = evaluate(solver.ema.module, solver.criterion, solver.postprocessor,
        solver.val_dataloader, solver.evaluator, solver.device, epoch=-1, use_wandb=False)
    rows = [json.loads(s) for s in (SOURCES[source].parent/'log.txt').read_text().splitlines() if s.strip()]
    expected = next(r['test_coco_eval_bbox'] for r in rows if r['epoch'] == 17)
    metrics = values['coco_eval_bbox']
    error = max(abs(a-b) for a, b in zip(expected, metrics))
    assert error < .0002, (source, error)
    result = dict(source=source, images=1820, metrics=metrics, max_abs_error=error)
    write(REPORT/'initial_validation'/source/'verified.json', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=REPO/'experiments/phase_s/sam_rgb_retention_10e.yml')
    p.add_argument('--arm', choices=ARMS, default='sam_frozen')
    p.add_argument('--run', type=Path)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--eval', action='store_true')
    args = p.parse_args()
    if args.preflight:
        assert not (REPORT/'preflight.json').exists()
        runs = []
        # A single common physical batch across all four arms, fallback only on OOM.
        for batch in (32, 16, 8):
            try:
                runs = [preflight_one(args.config, arm, batch) for arm in ARMS]
                break
            except torch.cuda.OutOfMemoryError as error:
                write(REPORT/f'preflight_oom_b{batch}.json', dict(status='OOM', error=str(error)))
                import gc
                gc.collect(); torch.cuda.empty_cache()
        assert len(runs) == 4
        for n in ('sd2_conditioner', 'thermal_backbone', 'thermal_encoder'):
            assert len({r['initial'][n] for r in runs}) == 1
        for source in SOURCES:
            pair = [r for r in runs if r['arm'].startswith(source+'_')]
            for n in ('backbone', 'encoder', 'decoder'):
                assert pair[0]['initial'][n] == pair[1]['initial'][n]
        assert len({tuple(r['input_ids']) for r in runs}) == 1
        initial = [validate_initial(args.config, source, batch) for source in ('c', 'sam')]
        write(REPORT/'preflight.json', dict(status='PASS', selected_batch=batch, runs=runs,
            initial_validation=initial, effective_batch=32, accumulation=32//batch))
        print('PREFLIGHT_PASS', batch, flush=True)
        return
    cfg = prepare(args.config, args.arm, args.run, args.batch)
    if args.eval:
        from src.solver.det_engine import evaluate
        cfg.resume = str(args.run/'best_stg1.pth')
        solver = TASKS[cfg.yaml_cfg['task']](cfg); solver.eval()
        values, _ = evaluate(solver.ema.module, solver.criterion, solver.postprocessor,
            solver.val_dataloader, solver.evaluator, solver.device, epoch=-1, use_wandb=False)
        rows = [json.loads(s) for s in (args.run/'log.txt').read_text().splitlines() if s.strip()]
        assert [r['epoch'] for r in rows] == list(range(10))
        best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
        metrics = values['coco_eval_bbox']
        error = max(abs(a-b) for a, b in zip(metrics, best['test_coco_eval_bbox']))
        assert error < .0002
        write(args.run/'independent_best_ema.json', dict(status='PASS', images=1820,
            best_epoch=best['epoch'], metrics=metrics, max_abs_error=error))
        torch.save({'model': solver.ema.module.detector.state_dict()}, args.run/'detector_best_ema.pth')
        bypass = solver.ema.module.detector.sd2_conditioner.register_forward_hook(
            lambda module, inputs, output: inputs[0])
        off, _ = evaluate(solver.ema.module, solver.criterion, solver.postprocessor,
            solver.val_dataloader, solver.evaluator, solver.device, epoch=-1, use_wandb=False)
        bypass.remove()
        source = args.arm.split('_')[0]
        initial = json.loads((REPORT/'initial_validation'/source/'verified.json').read_text(encoding='utf-8'))
        off_metrics = off['coco_eval_bbox']
        off_error = max(abs(a-b) for a, b in zip(off_metrics, initial['metrics']))
        if args.arm.endswith('_frozen'):
            assert off_error < .0002, ('Frozen RGB inference drift', off_error)
        write(args.run/'same_weight_m_off.json', dict(images=1820, metrics=off_metrics,
            max_abs_difference_from_initial_rgb=off_error,
            note='Same best EMA; bypass M output, never use GT/SAM masks in inference'))
    else:
        assert not (args.run/'log.txt').exists()
        write(args.run/'initialization.json', dict(arm=args.arm, hashes=snapshots(cfg.model),
            source=str(SOURCES[args.arm.split('_')[0]]), weight_source='EMA epoch17',
            physical_batch=args.batch, accumulation=32//args.batch, epochs=10))
        class ProgressSolver(TASKS[cfg.yaml_cfg['task']]):
            def train(self):
                super().train()
                self.actual_updates = 0
                probes = [(n, p) for n, p in self.model.detector.sd2_conditioner.named_parameters()
                          if p.requires_grad]
                initial = [p.detach().flatten()[:32].clone() for _, p in probes]
                def updated(*_):
                    self.actual_updates += 1
                    if self.actual_updates <= 3 or self.actual_updates % 10 == 0:
                        change = max(float((p.detach().flatten()[:32]-v).abs().max())
                                     for (_, p), v in zip(probes, initial))
                        result = dict(status='training', actual_optimizer_updates=self.actual_updates,
                            epoch=self.model.epoch, arm=args.arm, physical_batch=args.batch,
                            accumulation=32//args.batch, m_probe_weight_max_change=change,
                            updated_at=datetime.datetime.now().isoformat())
                        write(args.run/'training_progress.json', result)
                        print('OPTIMIZER_UPDATE', json.dumps(result), flush=True)
                self.optimizer.register_step_post_hook(updated)
        solver = ProgressSolver(cfg)
        solver.fit()
        before = json.loads((args.run/'initialization.json').read_text(encoding='utf-8'))['hashes']
        after = snapshots(solver.model)
        ema_after = snapshots(solver.ema.module)
        fixed = ['thermal_backbone', 'thermal_encoder']
        if args.arm.endswith('_frozen'):
            fixed += ['backbone', 'encoder', 'decoder']
        for name in fixed:
            assert before[name] == after[name] == ema_after[name], ('Fixed stream drift', name)
        assert solver.actual_updates > 0
        write(args.run/'final_fixed_state_check.json', dict(status='PASS', fixed_modules=fixed,
            initial=before, model=after, ema=ema_after, actual_updates=solver.actual_updates))
    dist_utils.cleanup()


if __name__ == '__main__':
    main()
