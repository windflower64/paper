"""Independent best EMA forward using the exact BOX training sources."""
import hashlib
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/148_sgc2_half_box'
FROZEN = REPORT / 'frozen_repo'
sys.path.insert(0, str(FROZEN))
from src.core import YAMLConfig
from src.solver import TASKS
from src.solver.det_engine import evaluate


def main():
    torch.set_num_threads(4)
    run = ROOT / 'outputs/C_PLUS_M_SD22_HALF_SGC2_BOX_DECAY9_14_B8A4_20E_TESTDEV/seed0'
    checkpoint = run / 'best_stg1.pth'
    cfg = YAMLConfig(str(FROZEN / 'experiments/phase_s/s_sgc2_box_c_plus_m_sd22_half_b8a4_20e.yml'),
                     resume=str(checkpoint), output_dir=str(REPORT / 'evaluation_runtime'))
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()
    assert len(solver.val_dataloader.dataset) == 1820
    assert solver.val_dataloader.dataset.sam_mask_root is None
    assert solver.ema.module.sd2_conditioner.final_residual_scale == .5
    values, _ = evaluate(solver.ema.module, solver.criterion, solver.postprocessor,
                         solver.val_dataloader, solver.evaluator, solver.device, epoch=-1, use_wandb=False)
    rows = [json.loads(s) for s in (run / 'log.txt').read_text().splitlines() if s.strip()]
    best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
    actual = values['coco_eval_bbox']
    error = max(abs(a-b) for a,b in zip(actual,best['test_coco_eval_bbox']))
    result = {'status': 'PASS' if error < .0002 else 'FAIL', 'images': 1820, 'best_epoch': best['epoch'],
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'coco_eval_bbox': actual, 'logged_coco_eval_bbox': best['test_coco_eval_bbox'],
              'max_abs_error': error, 'source': str(FROZEN / 'src'), 'sam_validation_cache': None}
    (REPORT / 'best_ema_evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    assert result['status'] == 'PASS'


if __name__ == '__main__':
    main()
