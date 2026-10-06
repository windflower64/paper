"""Remove SQER readout at runtime; train paired SBR3 replacements on frozen base."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
import train_sbr1 as engine
from preflight_sqer1 import seed_all, move_targets, sha256_file

RUN = engine.ROOT / 'outputs/SBR3_REPLACE_20261004'


class FeatureTap(nn.Module):
    """Parameter-free compatibility port; no SQER attention or score head exists."""
    def __init__(self):
        super().__init__()
        self.cache = {}

    def forward(self, s4, s8, queries, logits, boxes, **kwargs):
        self.cache = {'s4': s4.detach(), 's8': s8.detach(), 'base_logits': logits.detach()}
        return {'delta_logits': torch.zeros_like(logits),
                'query_indices': logits.sigmoid().amax(-1).topk(min(64, logits.shape[1]), dim=1).indices,
                'roi_grid': None, 'attention': None}


class ReplacementSource:
    def __init__(self, model):
        self.model = model.eval().requires_grad_(False)

    def __call__(self, samples):
        with torch.no_grad():
            out = self.model(samples)
        cache = self.model.sqer.cache
        self.model.sqer.cache = {}
        assert set(cache) == {'s4', 's8', 'base_logits'}
        assert torch.equal(out['pred_logits'], out['base_pred_logits'])
        assert not list(self.model.sqer.parameters())
        return out, cache

    def close(self):
        self.model.sqer.cache = {}


def setup(batch):
    cfg, old, unused, _ = engine.setup(batch)
    base = old.model
    old.close()
    del unused
    from src.zoo.dfine.sam_side_search import SAMSideSearch
    refiner = SAMSideSearch(base.sqer).cuda()
    base.sqer = FeatureTap()
    return cfg, ReplacementSource(base), refiner, refiner.loss


def readout(model, cache, out, hw):
    return model(cache['s4'], cache['s8'], cache['base_logits'], out['pred_boxes'], hw)


def evaluate(cfg, source, model):
    return engine.evaluate(cfg, source, model, readout_fn=readout)


def preflight(args):
    cfg, source, model, lossfn = setup(args.batch_size)
    RUN.mkdir(parents=True, exist_ok=True)
    init = RUN / 'common_refiner_init.pth'
    if init.exists():
        model.load_state_dict(torch.load(init, map_location='cuda'))
    else:
        engine.save_checkpoint(init, model.state_dict())
    digest = engine.digest(source.model)
    samples, targets = next(iter(cfg.train_dataloader))
    samples = samples.cuda()
    targets = move_targets(targets, torch.device('cuda'))
    out, cache = source(samples)
    matches = engine.matching(cfg.criterion.matcher.cuda(), out, targets)
    checks = {}
    for arm, weight in [('GT_ONLY', 0.), ('SAM', 1.)]:
        model.load_state_dict(torch.load(init, map_location='cuda'))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with torch.no_grad():
            assert torch.equal(readout(model, cache, out, samples.shape[-2:])['boxes'], out['pred_boxes'])
        for step in range(4):
            seed_all(20000 + step)
            opt.zero_grad(set_to_none=True)
            result = readout(model, cache, out, samples.shape[-2:])
            loss, stats = lossfn(result, out['pred_boxes'], targets, matches, weight)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(loss) and torch.isfinite(norm)
            opt.step()
        for prefix in ['shape_head', 's4_projection', 's8_projection', 'fusion', 'score']:
            assert sum(float(p.grad.norm()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None) > 0
        checks[arm] = {'loss': float(loss), 'stats': stats, 'gradient_norm': float(norm)}
    assert engine.digest(source.model) == digest
    assert not any(p.grad is not None for p in source.model.parameters())
    del samples, targets, out, cache, result, loss, opt
    model.load_state_dict(torch.load(init, map_location='cuda'))
    baseline = evaluate(cfg, source, model)
    report = {'status': 'PASS', 'batch_size': args.batch_size, 'precision': 'float32',
              'baseline': baseline, 'sqer_runtime_parameters': 0, 'checks': checks,
              'base_sha': engine.BASE_SHA, 'init_sha': sha256_file(init)}
    engine.write_json(RUN / 'preflight.json', report)
    print(json.dumps(report), flush=True)
    source.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--arm', choices=['R2_SAM', 'R1_GT_ONLY'])
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=30)
    args = p.parse_args()
    seed_all(0)
    torch.set_num_threads(4)
    if args.preflight:
        preflight(args)
    elif args.arm:
        files = [Path(__file__), Path(engine.__file__), engine.REPO/'src/zoo/dfine/sam_side_search.py']
        engine.train(args, setup_fn=setup, readout_fn=readout, evaluate_fn=evaluate, run_dir=RUN,
                     sam_shape_weight=1., extra_protocol={
                         'module': 'SBR3_REPLACE', 'sqer_readout_removed': True,
                         'initialization': 'historical trained base; copies of S4/S8 projection only, not scratch',
                         'precision': 'float32', 'epoch_random_seed_base': 20000,
                         'roi_size': None, 'teacher_in_forward': False,
                         'source_hashes': {str(f): sha256_file(f) for f in files}})
    else:
        p.error('Specify --preflight or --arm')


if __name__ == '__main__':
    main()
