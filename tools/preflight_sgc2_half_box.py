"""Verify missing full RGB-T BOX arm without updating historical weights."""
import copy
import json
import sys
from pathlib import Path
import torch
# Reuse the exact source snapshot used by the reference training.
HISTORICAL = Path(__file__).resolve().parents[2] / 'outputs/C_PLUS_M_SD22_HALF_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0/artifacts'
sys.path.insert(0, str(HISTORICAL))
import src.core
import preflight_s_sgc2_c_plus_m as base

CONFIG = base.REPO / 'experiments/phase_s/s_sgc2_box_c_plus_m_sd22_half_b8a4_20e.yml'
REFERENCE = base.REPO / 'experiments/phase_s/s_sgc2_sam_c_plus_m_sd22_half_b8a4_20e.yml'
REPORT = base.ROOT / 'reports/148_sgc2_half_box/preflight.json'


def main():
    torch.set_num_threads(4)
    configs = [base.YAMLConfig(str(p)) for p in (REFERENCE, CONFIG)]
    values = [copy.deepcopy(c.yaml_cfg) for c in configs]
    assert values[0]['DFINE'].pop('sgc_supervision') == 'sam'
    assert values[1]['DFINE'].pop('sgc_supervision') == 'box'
    for value in values:
        value.pop('__include__', None)
        value.pop('output_dir')
    assert values[0] == values[1]
    digests = []
    for cfg in configs:
        torch.manual_seed(0)
        model = cfg.model
        shim = base.BaseSolver.__new__(base.BaseSolver)
        shim.model = model
        shim.load_tuning_state(str(base.CHECKPOINT))
        digests.append(base.model_digest(model))
    assert digests[0] == digests[1]
    reference = json.loads((base.ROOT / 'reports/112_sgc2_rgbt_half/preflight.json').read_text())
    assert digests[1] == reference['pre_forward_initial_sha256'], 'Historical initialization changed'
    cfg = configs[1]
    assert model.sd2_conditioner.final_residual_scale == .5
    assert model.sd2_conditioner.thermal_contrastive
    assert not model.decoder.hrqs_enabled
    assert cfg.yaml_cfg['gradient_accumulation_steps'] == 4
    samples, targets = next(iter(cfg.train_dataloader))
    assert tuple(samples.shape) == (8, 6, 512, 640)
    assert cfg.val_dataloader.dataset.sam_mask_root is None
    samples = samples.cuda()
    targets = base.move_targets(targets, torch.device('cuda'))
    model.cuda().eval()
    with torch.no_grad():
        a = model(samples[:2])
        model.sgc_aux_weight = 0.
        b = model(samples[:2])
        model.sgc_aux_weight = 10.
    errors = {k: float((a[k]-b[k]).abs().max()) for k in ('pred_logits', 'pred_boxes')}
    assert max(errors.values()) == 0
    model.train()
    model.set_training_epoch(0)
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        output = model(samples, targets)
    assert torch.isfinite(output['sgc_group_loss']) and float(output['sgc_group_loss']) > 0
    m_parameters = [p for p in model.sd2_conditioner.parameters() if p.requires_grad]
    grads = torch.autograd.grad(output['sgc_group_loss'], m_parameters, allow_unused=True, retain_graph=True)
    assert all(g is None or float(g.abs().max()) == 0 for g in grads)
    criterion = cfg.criterion.cuda()
    with torch.autocast('cuda', enabled=False):
        losses = criterion(output, targets, epoch=0, step=0, global_step=0, epoch_step=len(cfg.train_dataloader))
        loss = sum(losses.values()) / 4
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {p: base.grad_summary(model, p) for p in ('backbone.', 'sd2_conditioner.', 'thermal_backbone.')}
    assert all(g['all_finite'] for g in gradients.values())
    assert gradients['backbone.']['with_nonzero_gradient'] > 0
    assert gradients['sd2_conditioner.']['with_nonzero_gradient'] > 0
    assert gradients['thermal_backbone.']['with_gradient'] == 0
    result = dict(status='PASS', config=str(CONFIG), only_region_definition_changed=True,
                  same_initial_weights=True, initial_sha256=digests[1],
                  inference_exact_without_sgc=errors, sgc_direct_m_gradient_max=0.,
                  gradients=gradients, loss=float(loss.detach()), physical_batch=8, accumulation=4,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  historical_src=str(HISTORICAL / 'src'),
                  caveat='Uses reference historical source; inference test changes auxiliary weight only to preserve S8 routing')
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
