"""SAM grouping contrast pilot; no optimizer, model edits, or test labels.

Inspired by SAMFeat's WSC concept, NOT its loss implementation or reproduction.
Positive references exclude the anchor; only same-image local negatives used.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO))


def raster(boxes, height, width, expansion=1.):
    y = (torch.arange(height, device=boxes.device) + .5) / height
    x = (torch.arange(width, device=boxes.device) + .5) / width
    out = torch.zeros((height, width), device=boxes.device, dtype=torch.bool)
    for cx, cy, bw, bh in boxes:
        out |= ((y[:, None] - cy).abs() <= bh * expansion / 2) & (
            (x[None, :] - cx).abs() <= bw * expansion / 2)
    return out.float()


def selection(target, size, source):
    masks = target['masks'].float()
    h, w = masks.shape[-2:]
    boxes = target['boxes']
    if len(boxes) != 1 or float(target['sam_quality']) <= 0:
        return None
    field = masks.amax(0) if source == 'sam' else raster(boxes, h, w)
    occupancy = F.interpolate(field[None, None], size=size, mode='area')[0, 0]
    # The same GT-supported local neighbourhood in both arms. Keep ambiguous
    # mixed cells out; do not mistake every non-object region for a new class.
    region = raster(boxes, h, w, expansion=2.)
    local = F.interpolate(region[None, None], size=size, mode='area')[0, 0]
    positives = (occupancy >= .75).flatten().nonzero().flatten()
    negatives = ((occupancy <= .05) & (local >= .5)).flatten().nonzero().flatten()
    return positives, negatives


def choose(indices, maximum):
    if indices.numel() <= maximum:
        return indices
    return indices[torch.linspace(0, indices.numel()-1, maximum,
                                  device=indices.device).round().long()]


def contrast(feature, indices):
    positive, negative = indices
    if len(positive) < 2 or len(negative) < 4:
        return feature.float().sum() * 0., None
    tokens = F.normalize(feature.float().flatten(1).T, dim=-1)
    p = tokens[choose(positive, 32)]
    n = tokens[choose(negative, 64)]
    # No self-pair. Averages of OTHER foreground tokens are positive references.
    reference = F.normalize(p.sum(0, keepdim=True) - p, dim=-1)
    sim_positive = (p * reference).sum(-1)
    sim_negative = p @ n.T
    loss = F.softplus((sim_negative - sim_positive[:, None]) / .2).mean()
    return loss, dict(positive_cells=len(positive), negative_cells=len(negative),
                      separation=float((sim_positive.mean()-sim_negative.mean()).detach()))


def run(state_name, batches):
    from src.core import YAMLConfig
    from src.solver import BaseSolver
    from tools.preflight_q_rank1 import move_targets

    torch.set_num_threads(4)
    seed = 20260907
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    cfg = YAMLConfig(str(REPO/'experiments/phase_m/c_only_gq1_b8a4_20e_testdev_local.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    cfg.yaml_cfg['train_dataloader']['dataset']['sam_mask_root'] = str(
        ROOT/'reports/104_sam3_role_control/masks_train')
    model = cfg.model
    if state_name == 'initial':
        checkpoint = ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth'
        shim = BaseSolver.__new__(BaseSolver); shim.model = model
        shim.load_tuning_state(str(checkpoint))
    else:
        checkpoint = ROOT/'outputs/C_ONLY_GQ1_B8A4_20E_TESTDEV/seed0/best_stg1.pth'
        model.load_state_dict(torch.load(checkpoint, map_location='cpu')['ema']['module'], strict=True)
    model = model.cuda().train()
    # This is a feature-gradient diagnostic, not an AP or optimizer experiment.
    # Freeze BN statistics so repeated pilot calls never change the checkpoint.
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    criterion = cfg.criterion.cuda()
    captured = {}
    hook = model.backbone.stages[1].register_forward_hook(
        lambda module, inputs, output: captured.update(s8=output))
    loader = cfg.train_dataloader
    # Fixed augmented batches shared by initial and best, despite state loading.
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    records, coverage = [], []
    for step, (samples, targets) in enumerate(loader):
        if step >= batches:
            break
        samples = samples.cuda(); targets = move_targets(targets, 'cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            output = model(samples, targets)
        losses = criterion(output, targets, epoch=0, step=step,
                           global_step=step, epoch_step=len(loader))
        detection = sum(losses.values())
        feature = captured['s8']
        assert tuple(feature.shape[-2:]) == (64, 80), feature.shape
        dgrad = torch.autograd.grad(detection, feature, retain_graph=True)[0].float()
        selection_by_arm = {a: [selection(t, feature.shape[-2:], a) for t in targets]
                            for a in ('sam', 'box')}
        usable = {a: [i for i, ind in enumerate(inds)
                       if ind is not None and len(ind[0]) >= 2 and len(ind[1]) >= 4]
                  for a, inds in selection_by_arm.items()}
        common = sorted(set(usable['sam']) & set(usable['box']))
        for i, t in enumerate(targets):
            coverage.append(dict(step=step, image_id=int(t['image_id'].item()),
                positive=len(t['boxes']) > 0, accepted=float(t['sam_quality']) > 0,
                **{a: None if inds[i] is None else [len(inds[i][0]), len(inds[i][1])]
                   for a, inds in selection_by_arm.items()},
                sam_usable=i in usable['sam'], box_usable=i in usable['box']))
        row = dict(step=step, common_images=len(common), detection_loss=float(detection.detach()))
        for arm in ('sam', 'box'):
            items = [contrast(feature[i], selection_by_arm[arm][i]) for i in common]
            aux = sum((p[0] for p in items), feature.float().sum()*0) / max(1, len(items))
            agrad = torch.autograd.grad(aux, feature, retain_graph=True)[0].float()
            cosine = F.cosine_similarity(dgrad.flatten(), agrad.flatten(), dim=0).item()
            ratio = (agrad.norm()/dgrad.norm().clamp_min(1e-12)).item()
            assert torch.isfinite(aux) and torch.isfinite(agrad).all()
            row[arm] = dict(loss=float(aux.detach()), gradient_cosine=cosine,
                            raw_gradient_ratio=ratio,
                            separation=[p[1]['separation'] for p in items])
        records.append(row)
        print(json.dumps(row), flush=True)
        captured.clear()
        del output, losses, detection, feature, dgrad, agrad, aux, items
    hook.remove()
    destination = ROOT/'reports/105_sam_group_contrast_pilot'
    destination.mkdir(parents=True, exist_ok=True)
    result = dict(state=state_name, seed=seed, batches=batches, physical_batch=8,
                  checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  protocol='train augmentation; BN eval; no optimizer; shared SAM/BOX images; S8 gradients',
                  records=records, coverage=coverage)
    with (destination/f'{state_name}.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('state', choices=['initial', 'best'])
    parser.add_argument('--batches', type=int, default=16)
    args = parser.parse_args()
    run(args.state, args.batches)
