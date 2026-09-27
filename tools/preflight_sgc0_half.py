"""严格零SAM梯度对照：配置、初值、前向与真实batch8反向核验。"""
import copy
import hashlib
import json
import torch
import preflight_s_sgc2_c_plus_m as base

ROOT, REPO = base.ROOT, base.REPO
CONFIG = REPO / 'experiments/phase_s/c_plus_m_sd22_half_sgc0_b8a4_20e.yml'
REFERENCE = REPO / 'experiments/phase_s/s_sgc2_sam_c_plus_m_sd22_half_b8a4_20e.yml'
REPORT = ROOT / 'reports/113_sgc0_half_control/preflight.json'


def main():
    torch.set_num_threads(4)
    configs = [base.YAMLConfig(str(p)) for p in (REFERENCE, CONFIG)]
    values = [copy.deepcopy(c.yaml_cfg) for c in configs]
    assert values[0]['DFINE'].pop('sgc_aux_weight') == 10.0
    assert values[1]['DFINE'].pop('sgc_aux_weight') == 0.0
    for c in values:
        c.pop('__include__', None); c.pop('output_dir')
    assert values[0] == values[1]
    manifest = json.loads((ROOT/'outputs/C_PLUS_M_SD22_HALF_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0/artifacts/manifest.json').read_text(encoding='utf-8'))
    protected = [p for p in manifest if any(x in p.replace('\\','/') for x in
                 ('/data/', '/reports/104_sam3_role_control/', '/weights/'))]
    for path in protected:
        assert hashlib.sha256(base.Path(path).read_bytes()).hexdigest() == manifest[path], path
    digests = []
    for cfg in configs:
        torch.manual_seed(0)
        model = cfg.model
        shim = base.BaseSolver.__new__(base.BaseSolver); shim.model = model
        shim.load_tuning_state(str(base.CHECKPOINT))
        digests.append(base.model_digest(model))
    assert digests[0] == digests[1]
    prior = json.loads((ROOT/'reports/112_sgc2_rgbt_half/preflight.json').read_text())
    assert digests[1] == prior['pre_forward_initial_sha256']
    cfg = configs[1]
    assert model.sgc_enabled and model.sgc_aux_weight == 0.0
    assert model.sd2_conditioner.final_residual_scale == 0.5
    assert cfg.yaml_cfg['gradient_accumulation_steps'] == 4
    assert len(cfg.val_dataloader.dataset) == 1820 and cfg.val_dataloader.dataset.sam_mask_root is None
    samples, targets = next(iter(cfg.train_dataloader))
    assert tuple(samples.shape) == (8,6,512,640)
    samples = samples.cuda(); targets = base.move_targets(targets, torch.device('cuda'))
    model.cuda().train(); model.set_training_epoch(0)
    # Same model, same batch and same RNG; only auxiliary weight changes.
    rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        model.sgc_aux_weight = 10.0
        reference = model(samples, copy.deepcopy(targets))
        torch.set_rng_state(rng); torch.cuda.set_rng_state_all(cuda_rng)
        model.sgc_aux_weight = 0.0
        control = model(samples, copy.deepcopy(targets))
    errors = {k:float((reference[k]-control[k]).abs().max()) for k in ('pred_logits','pred_boxes')}
    assert max(errors.values()) == 0.0
    assert float(reference['sgc_group_loss']) > 0
    assert float(control['sgc_group_loss']) == 0.0
    del reference, control
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        output = model(samples, targets)
    visible_parameter = model.backbone.stages[1].blocks[0].layers[0].conv.weight
    m_parameter = next(p for p in model.sd2_conditioner.parameters() if p.requires_grad)
    aux_grads = torch.autograd.grad(output['sgc_group_loss'], (visible_parameter,m_parameter),
                                    retain_graph=True, allow_unused=True)
    assert all(g is None or float(g.abs().max()) == 0 for g in aux_grads)
    criterion = cfg.criterion.cuda()
    with torch.autocast('cuda', enabled=False):
        losses = criterion(output, targets, epoch=0, step=0, global_step=0,
                           epoch_step=len(cfg.train_dataloader))
        loss = sum(losses.values())/4
    assert torch.isfinite(loss)
    loss.backward()
    grads = {p:base.grad_summary(model,p) for p in ('backbone.','sd2_conditioner.','thermal_backbone.')}
    assert all(v['all_finite'] for v in grads.values())
    assert grads['backbone.']['with_nonzero_gradient'] > 0
    assert grads['sd2_conditioner.']['with_nonzero_gradient'] > 0
    assert grads['thermal_backbone.']['with_gradient'] == 0
    model.eval()
    with torch.no_grad():
        a = model(samples[:2]); model.sgc_aux_weight = 10.0
        b = model(samples[:2]); model.sgc_aux_weight = 0.0
    eval_errors = {k:float((a[k]-b[k]).abs().max()) for k in ('pred_logits','pred_boxes')}
    assert max(eval_errors.values()) == 0.0
    result = dict(status='PASS',config=str(CONFIG),only_sgc_weight_changed=True,
                  same_initial_weights=True,initial_sha256=digests[1],
                  checked_protected_files=len(protected),final_residual_scale=0.5,
                  sgc_aux_weight=0.0,zero_sam_gradient=True,
                  train_forward_errors=errors,inference_exact_without_sgc=eval_errors,
                  sgc_direct_m_gradient_max=0.0,gradient=grads,loss=float(loss.detach()),
                  physical_batch=8,accumulation=4,val_images=1820)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
