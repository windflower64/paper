"""S-MFAM1 data provenance, inference isolation and real batch preflight."""
import json
import sys
from pathlib import Path
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core import YAMLConfig
from src.solver import BaseSolver
from tools.preflight_q_rank1 import checkpoint_weights, load_compatible, move_targets


def main():
    torch.set_num_threads(8)
    torch.manual_seed(0)
    cfg = YAMLConfig(str(ROOT / 'experiments/phase_s/s_mfam1_c_sam_b8a4_20e_testdev_local.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    model = cfg.model
    source, source_name = checkpoint_weights(ROOT.parent/'weights/m_sd2_joint_coco_thermal_identity_init.pth')
    # Use the actual training loader, including dataset head adjustment.
    shim = BaseSolver.__new__(BaseSolver)
    shim.model = model
    shim.load_tuning_state(str(ROOT.parent/'weights/m_sd2_joint_coco_thermal_identity_init.pth'))
    loader = cfg.train_dataloader
    dataset = loader.dataset
    assert cfg.val_dataloader.dataset.sam_mask_root is None
    records = json.loads((dataset.sam_mask_root/'records.json').read_text(encoding='utf-8'))
    by_id = {int(r['image_id']): r for r in records}
    accepted, rejected, empty = 0, 0, 0
    for image_id in dataset.ids:
        info = dataset.coco.imgs[image_id]
        anns = dataset.coco.imgToAnns.get(image_id, [])
        if not anns:
            empty += 1
            continue
        assert image_id in by_id, ('record missing', image_id)
        record = by_id[image_id]
        assert record['file_name'] == info['file_name'], ('wrong image', image_id)
        assert len(anns) == 1, ('multi-object unsupported by cache', image_id)
        x, y, w, h = anns[0]['bbox']
        expected = torch.tensor([x, y, x+w, y+h])
        assert torch.allclose(torch.tensor(record['bbox_xyxy_annotation']), expected, atol=.01), image_id
        if record['accepted']:
            path = dataset.sam_mask_root/'masks'/f'{image_id:06d}.png'
            with Image.open(path) as mask:
                assert mask.size == (info['width'], info['height']), image_id
                assert mask.getbbox() is not None, ('accepted empty mask', image_id)
            accepted += 1
        else:
            rejected += 1
    print('MASK_AUDIT', accepted, rejected, empty, flush=True)
    model = model.cuda()
    criterion = cfg.criterion.cuda()
    optimizer = cfg.optimizer
    covered = {id(p) for g in optimizer.param_groups for p in g['params']}
    assert all(id(p) in covered for p in model.mfam.parameters())
    samples, targets = next(iter(loader))
    assert samples.shape == (8, 3, 512, 640), samples.shape
    samples, targets = samples.cuda(), move_targets(targets, 'cuda')
    model.eval()
    with torch.no_grad():
        model.mfam.intervention = 'disabled'
        disabled = model(samples)
        baseline_cfg = YAMLConfig(str(ROOT/'experiments/phase_m/c_only_gq1_b8a4_20e_testdev_local.yml'))
        baseline_cfg.yaml_cfg['HGNetv2']['pretrained'] = False
        baseline_cfg.yaml_cfg['HGNetv2']['return_idx'] = [2, 3]
        baseline_cfg.yaml_cfg['DFINE']['mfam_enabled'] = False
        baseline = baseline_cfg.model
        # Compare identical common weights; C attention and dataset heads are
        # intentionally initialized by the training loader rather than all
        # being present in the COCO source checkpoint.
        shared = {k: v for k, v in model.state_dict().items() if not k.startswith('mfam.')}
        baseline.load_state_dict(shared, strict=True)
        baseline = baseline.cuda().eval()
        reference = baseline(samples)
        baseline_error = max((disabled[k]-reference[k]).abs().max().item() for k in ('pred_logits','pred_boxes'))
        assert baseline_error == 0, baseline_error
        del baseline, reference, disabled
        model.mfam.intervention = 'learned'
        learned = model(samples)
        model.mfam.intervention = 'constant'
        constant = model(samples)
        intervention_error = (learned['pred_logits']-constant['pred_logits']).abs().max().item()
        assert intervention_error > 0
        del learned, constant
    model.mfam.intervention = 'learned'
    model.train()
    torch.cuda.reset_peak_memory_stats()
    before = model.mfam.output_proj.weight.detach().clone()
    scaler = torch.cuda.amp.GradScaler(init_scale=128)
    optimizer.zero_grad(set_to_none=True)
    records_step = []
    for step, (samples, targets) in enumerate(loader):
        if step == 4:
            break
        samples, targets = samples.cuda(), move_targets(targets, 'cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            outputs = model(samples, targets)
        losses = criterion(outputs, targets, epoch=0, step=step, global_step=step, epoch_step=len(loader))
        assert 'loss_mfam_region' in losses
        total = sum(losses.values())
        assert torch.isfinite(total)
        if step == 0:
            detector_only = sum(v for k, v in losses.items() if k != 'loss_mfam_region')
            detector_mask_gradient = torch.autograd.grad(
                detector_only, model.mfam.mask_head[-1].weight, retain_graph=True
            )[0].float().norm().item()
            assert detector_mask_gradient > 0
        scaler.scale(total/4).backward()
        row = {'step': step, 'total_loss': total.item(), 'region_loss': losses['loss_mfam_region'].item()}
        records_step.append(row)
        print('STEP', row, flush=True)
        del outputs, losses, total
    scaler.unscale_(optimizer)
    grads = {name: p.grad.float().norm().item() if p.grad is not None else 0.
             for name, p in model.mfam.named_parameters()}
    assert all(torch.isfinite(p.grad).all() for p in model.mfam.parameters() if p.grad is not None)
    assert grads['mask_head.1.weight'] > 0 and grads['output_proj.weight'] > 0, grads
    torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
    scaler.step(optimizer)
    scaler.update()
    update = (model.mfam.output_proj.weight-before).abs().max().item()
    assert update > 0
    report = dict(status='PASS', physical_batch=8, gradient_accumulation_steps=4,
                  effective_batch=32, epochs=20, seed=0, source=source_name,
                  mask_audit=dict(accepted=accepted,rejected=rejected,empty=empty),
                  baseline_disabled_error=baseline_error, constant_intervention_error=intervention_error,
                  detector_only_mask_gradient=detector_mask_gradient,
                  module_parameters=sum(p.numel() for p in model.mfam.parameters()),
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  gradients=grads, update=update, steps=records_step)
    output = ROOT.parent/'reports/98_s_mfam1/preflight.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
