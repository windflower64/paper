"""Evaluate trained MFAM under region/branch interventions, without retraining."""
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig
from src.solver import TASKS
from src.solver.det_engine import evaluate


def main():
    arm = sys.argv[1].upper()
    assert arm in ('SAM', 'BOX')
    torch.set_num_threads(8)
    config = ROOT/f'experiments/phase_s/s_mfam1_c_{arm.lower()}_b8a4_20e_testdev_local.yml'
    run = ROOT.parent/f'outputs/S_MFAM1_C_{arm}_B8A4_20E_TESTDEV/seed0'
    output = ROOT.parent/f'reports/98_s_mfam1/interventions/{arm}'
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = run/'best_stg1.pth'
    cfg = YAMLConfig(str(config), resume=str(checkpoint), output_dir=str(output/'runtime'))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()
    model = solver.ema.module if solver.ema else solver.model
    assert model.mfam is not None
    assert solver.val_dataloader.dataset.sam_mask_root is None
    # Preserve training-time source snapshots as the comparison authority.
    for filename in ('sam_mask_aggregation.py', 'dfine.py', 'dfine_criterion.py'):
        current = ROOT/'src/zoo/dfine'/filename
        assert current.read_bytes() == (run/'artifacts'/filename).read_bytes(), filename
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for mode in ('learned', 'constant', 'disabled', 'shifted', 'mean'):
        destination = output/f'{mode}.json'
        if destination.exists():
            existing = json.loads(destination.read_text(encoding='utf-8'))
            assert existing['checkpoint_sha256'] == digest
            continue
        model.mfam.intervention = mode if mode in ('learned','constant','disabled') else 'learned'
        hook = None
        if mode == 'shifted':
            hook = model.mfam.mask_head.register_forward_hook(
                lambda module, inputs, logits: logits.roll(
                    shifts=(logits.shape[-2]//2, logits.shape[-1]//2), dims=(-2,-1)))
        elif mode == 'mean':
            hook = model.mfam.mask_head.register_forward_hook(
                lambda module, inputs, logits: torch.logit(
                    logits.sigmoid().mean((-2,-1),keepdim=True).clamp(1e-6,1-1e-6)
                ).expand_as(logits))
        stats, _ = evaluate(model, solver.criterion, solver.postprocessor,
                            solver.val_dataloader, solver.evaluator, solver.device,
                            epoch=-1, use_wandb=False)
        if hook is not None:
            hook.remove()
        result = dict(arm=arm, mode=mode, checkpoint=str(checkpoint),
                      checkpoint_sha256=digest, weight_source='ema' if solver.ema else 'model',
                      coco_eval_bbox=stats['coco_eval_bbox'])
        destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('MFAM_RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
