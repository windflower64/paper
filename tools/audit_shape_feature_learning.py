"""Matched post-training KD and internal-shape readout; no detector updates."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = Path('E:/two_paper')
FROZEN = ROOT/'reports/153_shape_feature_transfer/attempt02/frozen_repo'
sys.path.insert(0, str(FROZEN))
from src.core import YAMLConfig
from tools.shape_feature_interface import FeatureDetector, roi_grid, rectangle_occupancy

OUT = ROOT/'reports/154_shape_feature_learning_audit'
ARMS = ('none', 'uniform', 'filled', 'sam')


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def readout(features, metadata, held_sets):
    details = []
    for fold, held in enumerate(held_sets):
        train = [i for i, r in enumerate(metadata) if r['group'] not in held]
        test = [i for i, r in enumerate(metadata) if r['group'] in held]
        x = np.concatenate([features[i] for i in train]).astype(np.float64)
        y = np.concatenate([metadata[i]['label'] for i in train]).astype(float)
        mu, std = x.mean(0), np.maximum(x.std(0), 1e-6)
        z = np.c_[(x-mu)/std, np.ones(len(x))]
        penalty = np.eye(z.shape[1]); penalty[-1, -1] = 0
        beta = np.linalg.solve(z.T@z+10*penalty, z.T@y)
        for i in test:
            predicted = np.c_[(features[i]-mu)/std, np.ones(len(features[i]))]@beta >= .5
            label = metadata[i]['label']
            bacc = .5*(predicted[label].mean()+(~predicted[~label]).mean())
            details.append(dict(image_id=metadata[i]['image_id'], fold=fold,
                                scale=metadata[i]['scale'], bacc=float(bacc)))
    assert len(details) == len(metadata)
    summary = {}
    for scale in ('all', 'small', 'other'):
        for fold in (None, 0, 1, 2, 3):
            chosen = [r for r in details if (scale == 'all' or r['scale'] == scale)
                      and (fold is None or r['fold'] == fold)]
            summary[f'{scale}/fold{fold}'] = dict(n=len(chosen), bacc=float(np.mean(
                [r['bacc'] for r in chosen])) if chosen else None)
    return dict(summary=summary, rows=details)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'conclusion.json').exists():
        raise RuntimeError('Completed output exists; do not overwrite')
    torch.set_num_threads(4); torch.manual_seed(0)
    selected = json.loads((ROOT/'reports/150_sam_local_geometry/attempt02/teacher_geometry_pilot.json').read_text())['selected_ids']
    geometry = json.loads((ROOT/'reports/153_shape_feature_transfer/teacher/geometry.json').read_text())
    teacher_rows = {r['image_id']: r for r in geometry['rows']}
    ann = json.loads((ROOT/'data/antiuav6k_common/annotations/instances_visible_common_train.json').read_text())
    boxes = {r['image_id']: r for r in ann['annotations']}
    normalization = json.loads((ROOT/'reports/153_shape_feature_transfer/teacher/normalization.json').read_text())
    teacher_mean = torch.tensor(normalization['mean']).view(1, 256, 1, 1).cuda()
    teacher_std = torch.tensor(normalization['std']).view(1, 256, 1, 1).cuda()
    cfg = YAMLConfig(str(FROZEN/'experiments/phase_s/shape_feature_transfer_b16a2_20e.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    train = cfg.yaml_cfg['train_dataloader']['dataset']
    for key in ('img_folder', 'ann_file', 'infrared_folder', 'infrared_label_folder'):
        if key in train:
            cfg.yaml_cfg['val_dataloader']['dataset'][key] = train[key]
    dataset = cfg.val_dataloader.dataset
    loader = DataLoader(Subset(dataset, [dataset.ids.index(i) for i in selected]),
                        batch_size=16, shuffle=False, num_workers=0,
                        collate_fn=cfg.val_dataloader.collate_fn)
    metadata, maps, excluded = [], {}, []
    teacher_features, coordinate_features = [], []
    for image_id in selected:
        r = teacher_rows[image_id]
        mask = torch.tensor(np.asarray(Image.open(ROOT/f'reports/104_sam3_role_control/masks_train/masks/{image_id:06d}.png')).copy() > 0).float()[None, None]
        grid = roi_grid(torch.tensor(r['roi']), 32)
        shape = F.grid_sample(mask, grid, align_corners=False)[0, 0].numpy().clip(0, 1)
        label = shape.ravel() >= .5
        inside = rectangle_occupancy(torch.tensor(r['roi']), torch.tensor(r['extent'])).numpy().ravel() >= .999
        pos, neg = np.flatnonzero(label & inside), np.flatnonzero(~label & inside)
        n = min(32, len(pos), len(neg))
        if n < 8:
            excluded.append(dict(image_id=image_id, foreground=len(pos), internal_background=len(neg)))
            continue
        rng = np.random.default_rng(20260917+image_id)
        points = np.r_[rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)]
        group = '_'.join(Path(r['file_name']).stem.split('_')[:2])
        meta = dict(image_id=image_id, group=group, scale='small' if boxes[image_id]['area'] < 1024 else 'other', label=label[points])
        metadata.append(meta)
        extent = torch.tensor(r['extent'])
        filled = rectangle_occupancy(torch.tensor(r['roi']), extent).cuda().clamp(0, 1)
        sam = torch.tensor(shape).cuda()
        weights = {'uniform': torch.ones_like(sam)/sam.numel()}
        for name, m in (('sam', sam), ('filled', filled)):
            weights[name] = .5*m/m.sum()+.5*(1-m)/(1-m).sum()
        maps[image_id] = dict(grid=grid.cuda(), points=points, weights=weights)
        t = torch.from_numpy(np.load(ROOT/f'reports/153_shape_feature_transfer/teacher/features/{image_id:06d}.npy').copy()).cuda().float()[None]
        t = (t-teacher_mean)/teacher_std
        teacher_features.append(t[0].flatten(1).T[points].cpu().numpy())
        xy = grid[0].reshape(-1, 2).numpy()
        # Relative coordinates are a privileged spatial-prior control.
        center = (np.array(r['extent'][:2])+np.array(r['extent'][2:]))/2
        size = np.array(r['extent'][2:])-np.array(r['extent'][:2])
        xy = ((xy+1)/2-center)/size
        coordinate_features.append(np.c_[xy, xy**2][points])
    groups = sorted({r['group'] for r in metadata}); np.random.default_rng(20260916).shuffle(groups)
    held_sets = [set(groups[f::4]) for f in range(4)]
    save('protocol.json', dict(selected=selected, eligible=len(metadata), excluded=excluded,
        small=sum(r['scale']=='small' for r in metadata), held_groups=[sorted(h) for h in held_sets],
        grid=32, points_per_class_max=32, ridge_lambda=10, threshold=.5,
        updates=0, image_augmentation='deterministic validation resize, no flip/photometric',
        caveats=['SAM pseudo-label agreement, not manual boundaries or detection AP',
                 'held groups apply to probe fitting; detectors saw these training images',
                 'GT-prompt masks and SAM-derived extent are privileged, not deployable',
                 'selected internal-background balanced points; not all pixels']))
    output = dict(status='complete_no_detector_updates', readout={}, kd={}, telemetry={})
    output['readout']['coordinates'] = readout(coordinate_features, metadata, held_sets)
    output['readout']['teacher'] = readout(teacher_features, metadata, held_sets)
    del teacher_features
    model = FeatureDetector(cfg.model, 'none', normalization).cuda().eval().requires_grad_(False)
    captured = {}
    handle = model.detector.backbone.register_forward_hook(lambda m, a, o: captured.__setitem__('s16', o[0].detach()))
    ordering = {r['image_id']: i for i, r in enumerate(metadata)}
    for arm in ARMS:
        run = ROOT/f'outputs/S_FEATURE_{arm.upper()}_B32A1_20E_TESTDEV/seed0'
        traces = [json.loads(s) for s in (run/'feature_kd.jsonl').read_text().splitlines()]
        output['telemetry'][arm] = [dict(epoch=e, qualified_per_batch=float(np.mean([r['valid'] for r in traces if r['epoch']==e])),
            raw_mse=float(np.mean([r['raw_mse'] for r in traces if r['epoch']==e]))) for e in range(10)]
        output['kd'][arm] = {}
        for ckname in ('checkpoint0004.pth', 'checkpoint0009.pth', 'best_stg1.pth', 'checkpoint0019.pth'):
            checkpoint = run/ckname
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(state['ema']['module'], strict=True)
            del state
            errors, source_features, projected_features = [], [None]*len(metadata), [None]*len(metadata)
            with torch.no_grad():
                for samples, targets in loader:
                    with torch.autocast('cuda', dtype=torch.float16):
                        model.detector(samples.cuda())
                    source = captured.pop('s16').float()
                    norm = F.layer_norm(source.permute(0,2,3,1), (512,)).permute(0,3,1,2)
                    projected = model.kd_projection(norm)
                    for j, target in enumerate(targets):
                        image_id = int(target['image_id'])
                        if image_id not in maps: continue
                        row = maps[image_id]
                        pred = F.grid_sample(projected[j:j+1], row['grid'], padding_mode='border', align_corners=False)
                        teacher = torch.from_numpy(np.load(ROOT/f'reports/153_shape_feature_transfer/teacher/features/{image_id:06d}.npy').copy()).cuda().float()[None]
                        teacher = (teacher-teacher_mean)/teacher_std
                        error = (pred-teacher).square().mean(1)[0]
                        zero_error = teacher.square().mean(1)[0]
                        detail = dict(image_id=image_id, scale=metadata[ordering[image_id]]['scale'])
                        for weight_name, weight in row['weights'].items():
                            detail[weight_name] = float((error*weight).sum())
                            detail[f'zero/{weight_name}'] = float((zero_error*weight).sum())
                        errors.append(detail)
                        if ckname == 'best_stg1.pth':
                            idx = ordering[image_id]; points = row['points']
                            sampled = F.grid_sample(source[j:j+1], row['grid'], padding_mode='border', align_corners=False)
                            source_features[idx] = sampled[0].flatten(1).T[points].cpu().numpy()
                            projected_features[idx] = pred[0].flatten(1).T[points].cpu().numpy()
            assert len(errors) == len(metadata)
            summary = {scale: {k: float(np.mean([r[k] for r in errors if scale=='all' or r['scale']==scale]))
                              for k in ('uniform','filled','sam','zero/uniform','zero/filled','zero/sam')}
                       for scale in ('all','small','other')}
            output['kd'][arm][ckname] = dict(summary=summary, rows=errors, sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            if ckname == 'best_stg1.pth':
                output['readout'][f'{arm}/S16'] = readout(source_features, metadata, held_sets)
                output['readout'][f'{arm}/projection'] = readout(projected_features, metadata, held_sets)
            print(arm, ckname, json.dumps(summary['small']), flush=True)
            save('partial_results.json', output)
    handle.remove()
    save('conclusion.json', output)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
