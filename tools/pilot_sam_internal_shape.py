"""Privileged fixed-support SAM label readout controls, not detector training."""
import hashlib
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'reports/148_sgc2_half_box/frozen_repo'))
from src.core import YAMLConfig
OUT = ROOT/'reports/151_sam_internal_shape'


def balanced_indices(y, rng):
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    n = min(32, len(pos), len(neg))
    return np.r_[rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)]


def metric(logit, label, support):
    y, p = label[support], logit[support] >= .5
    if not y.any() or y.all():
        return None
    return float(.5*(p[y].mean()+(~p[~y]).mean()))


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone-control', choices=('SAM', 'NONE'), default='SAM')
    parser.add_argument('--include-s16', action='store_true')
    parser.add_argument('--export-gradient', action='store_true')
    parser.add_argument('--four-fold-export', action='store_true')
    args = parser.parse_args()
    if args.backbone_control == 'NONE':
        OUT = OUT/'none_backbone'
    if args.include_s16:
        OUT = OUT/'scale_audit'
    if args.export_gradient:
        assert args.backbone_control == 'NONE' and args.include_s16
        OUT = OUT/'gradient_export'
        if args.four_fold_export:
            OUT=OUT/'four_folds'
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(0)
    pilot = json.loads((ROOT/'reports/150_sam_local_geometry/attempt02/teacher_geometry_pilot.json').read_text(encoding='utf-8'))
    selected = pilot['selected_ids']
    ann = json.loads((ROOT/'data/antiuav6k_common/annotations/instances_visible_common_train.json').read_text(encoding='utf-8'))
    images = {i['id']: i for i in ann['images']}
    annotations = {a['image_id']: a for a in ann['annotations']}
    cfg = YAMLConfig(str(ROOT/'reports/148_sgc2_half_box/frozen_repo/experiments/phase_s/s_sgc2_sam_c_plus_m_sd22_half_b8a4_20e.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    train = cfg.yaml_cfg['train_dataloader']['dataset']
    for key in ('img_folder', 'ann_file', 'infrared_folder', 'infrared_label_folder'):
        cfg.yaml_cfg['val_dataloader']['dataset'][key] = train[key]
    dataset = cfg.val_dataloader.dataset
    loader = DataLoader(Subset(dataset, [dataset.ids.index(i) for i in selected]), batch_size=16,
                        shuffle=False, collate_fn=cfg.val_dataloader.collate_fn, num_workers=0)
    run = 'C_PLUS_M_SD22_HALF_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV' if args.backbone_control == 'SAM' else 'C_PLUS_M_SD22_HALF_SGC0_B8A4_20E_TESTDEV'
    checkpoint = ROOT/'outputs'/run/'seed0/best_stg1.pth'
    model = cfg.model.cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['ema']['module'], strict=True)
    captured = {}
    levels = (2,) if args.export_gradient else (0, 1, 2) if args.include_s16 else (0, 1)
    handles = [model.backbone.stages[k].register_forward_hook(
        lambda module, args, output, level=k: captured.__setitem__(level, output.detach())) for k in levels]
    rows, excluded = [], []
    with torch.no_grad():
        for batch, (input_images, targets) in enumerate(loader):
            model(input_images.cuda())
            for j, target in enumerate(targets):
                image_id = int(target['image_id']); info = images[image_id]
                h, w = info['height'], info['width']
                mask = np.asarray(Image.open(ROOT/f'reports/104_sam3_role_control/masks_train/masks/{image_id:06d}.png')) > 0
                yy, xx = np.where(mask)
                assert mask.shape == (h, w) and len(xx)
                box = np.array([xx.min(), yy.min(), xx.max()+1, yy.max()+1], dtype=float)
                center, size = (box[:2]+box[2:])/2, box[2:]-box[:2]
                lo, hi = np.maximum(center-size*.75, 0), np.minimum(center+size*.75, [w, h])
                gx = lo[0]+(np.arange(24)+.5)/24*(hi[0]-lo[0])
                gy = lo[1]+(np.arange(24)+.5)/24*(hi[1]-lo[1])
                X, Y = np.meshgrid(gx, gy)
                coords = np.c_[((X-center[0])/size[0]).ravel(), ((Y-center[1])/size[1]).ravel()]
                geometry = np.c_[coords, coords**2].astype(np.float32)
                inside = ((X>=box[0]) & (X<box[2]) & (Y>=box[1]) & (Y<box[3])).ravel()
                label = mask[np.clip(Y.astype(int), 0, h-1), np.clip(X.astype(int), 0, w-1)].ravel()
                scale = 'small' if annotations[image_id]['area'] < 1024 else 'other'
                if int(label.sum()) < 8 or int((inside & ~label).sum()) < 8:
                    excluded.append({'image_id': image_id, 'scale': scale, 'reason': 'insufficient_foreground_or_inside_background',
                                     'foreground': int(label.sum()), 'inside_background': int((inside & ~label).sum())})
                    continue
                assert not (label & ~inside).any()
                rng = np.random.default_rng(20260916+image_id)
                points = np.flatnonzero(label)
                # Preserve all points on the four discrete extreme coordinates.
                pxy = coords[points]
                anchors = points[(pxy[:,0]==pxy[:,0].min()) | (pxy[:,0]==pxy[:,0].max()) |
                                 (pxy[:,1]==pxy[:,1].min()) | (pxy[:,1]==pxy[:,1].max())]
                # A sampled shape may not touch the original extent's extreme
                # grid rows. Restrict permutations to its actual discrete box.
                discrete_box = inside & (coords[:,0]>=pxy[:,0].min()) & (coords[:,0]<=pxy[:,0].max()) & (coords[:,1]>=pxy[:,1].min()) & (coords[:,1]<=pxy[:,1].max())
                remaining = np.setdiff1d(np.flatnonzero(discrete_box), anchors)
                scrambled = np.zeros_like(label); scrambled[anchors] = True
                scrambled[rng.choice(remaining, len(points)-len(anchors), replace=False)] = True
                assert scrambled.sum() == label.sum()
                assert np.all(coords[scrambled].min(0) == coords[label].min(0)) and np.all(coords[scrambled].max(0) == coords[label].max(0))
                grid = torch.tensor(np.stack([X/w*2-1, Y/h*2-1], -1), dtype=torch.float32, device='cuda')[None]
                features = {'coordinates': geometry}
                shapes = {}
                for level in levels:
                    feature = captured[level][j:j+1]
                    sampled = F.grid_sample(feature.float(), grid, align_corners=False, mode='bilinear')[0].flatten(1).T.cpu().numpy()
                    features[f'S{4*2**level}'] = np.c_[geometry, sampled]
                    shapes[f'S{4*2**level}'] = list(feature.shape[1:])
                labels = {'SAM': label, 'filled_extent': inside, 'scrambled': scrambled}
                indices = {name: balanced_indices(y, np.random.default_rng(20260916+image_id)) for name, y in labels.items()}
                rows.append({'image_id': image_id, 'sequence': '_'.join(Path(info['file_name']).stem.split('_')[:2]),
                             'scale': scale, 'features': features, 'labels': labels, 'indices': indices, 'inside': inside,
                             'shapes': shapes, 'grid': grid[0].cpu().numpy(), 'changed_fraction': float((scrambled!=label).mean())})
            if batch % 4 == 0:
                print('images', min((batch+1)*16, len(selected)), 'eligible', len(rows), 'excluded', len(excluded), flush=True)
    for handle in handles:
        handle.remove()
    assert rows
    groups = sorted({r['sequence'] for r in rows}); np.random.default_rng(20260916).shuffle(groups)
    held_sets = [set(groups[f::4]) for f in range(4)]
    if args.export_gradient:
        for fold,held in enumerate(held_sets if args.four_fold_export else held_sets[:1]):
            dest=OUT/f'fold{fold}' if args.four_fold_export else OUT
            dest.mkdir(parents=True,exist_ok=True)
            fit = [r for r in rows if r['sequence'] not in held]
            chosen = []
            for scale in ('small', 'other'):
                used = set()
                for r in rows:
                    if r['sequence'] in held and r['scale']==scale and r['sequence'] not in used:
                        chosen.append(r); used.add(r['sequence'])
                        if len(used)==16: break
            x = np.concatenate([r['features']['S16'][r['indices']['SAM']] for r in fit]).astype(np.float64)
            y = np.concatenate([r['labels']['SAM'][r['indices']['SAM']] for r in fit]).astype(float)
            mu, std = x.mean(0), np.maximum(x.std(0),1e-6)
            z = np.c_[(x-mu)/std,np.ones(len(x))]
            penalty = np.eye(z.shape[1]); penalty[-1,-1]=0
            beta = np.linalg.solve(z.T@z+10*penalty,z.T@y)
            assert not ({r['sequence'] for r in fit} & {r['sequence'] for r in chosen})
            torch.save({'rows': chosen, 'mu': mu, 'std': std, 'beta': beta,
                        'checkpoint':str(checkpoint), 'fit_groups': sorted({r['sequence'] for r in fit}),
                        'held_groups':sorted(held)}, dest/'fixed_reader.pt')
            (dest/'export_complete.json').write_text(json.dumps({'status':'fixed_S16_reader_exported',
                'fit_images':len(fit),'probe_images':len(chosen),'probe_ids':[r['image_id'] for r in chosen],
                'held_groups':sorted(held),'only_stage2_sampled':True,'detector_steps':0},indent=2),encoding='utf-8')
            print('gradient_export',fold,len(fit),len(chosen),flush=True)
        return
    results = {}
    for feature_name in ('coordinates',)+tuple(f'S{4*2**level}' for level in levels):
        for supervision in ('SAM', 'filled_extent', 'scrambled'):
            details = []
            for fold, held in enumerate(held_sets):
                fit = [r for r in rows if r['sequence'] not in held]
                test = [r for r in rows if r['sequence'] in held]
                assert fit and test and not ({r['sequence'] for r in fit} & {r['sequence'] for r in test})
                x = np.concatenate([r['features'][feature_name][r['indices'][supervision]] for r in fit]).astype(np.float64)
                y = np.concatenate([r['labels'][supervision][r['indices'][supervision]] for r in fit]).astype(float)
                mu, std = x.mean(0), np.maximum(x.std(0), 1e-6)
                z = np.c_[(x-mu)/std, np.ones(len(x))]
                penalty = np.eye(z.shape[1]); penalty[-1,-1] = 0
                beta = np.linalg.solve(z.T@z+10*penalty, z.T@y)
                for r in test:
                    x = r['features'][feature_name]
                    logit = np.c_[(x-mu)/std, np.ones(len(x))]@beta
                    details.append({'image_id': r['image_id'], 'fold': fold, 'scale': r['scale'],
                                    'all_bacc': metric(logit, r['labels']['SAM'], np.ones(len(logit), dtype=bool)),
                                    'inside_bacc': metric(logit, r['labels']['SAM'], r['inside'])})
            assert len(details) == len({r['image_id'] for r in details}) == len(rows)
            summary = {}
            for scale in ('all', 'small', 'other'):
                for fold in (None, 0, 1, 2, 3):
                    rr = [r for r in details if (scale=='all' or r['scale']==scale) and (fold is None or r['fold']==fold)]
                    summary[f'{scale}/fold{fold}'] = {'n': len(rr), **{k: float(np.mean([r[k] for r in rr])) if rr else None for k in ('all_bacc', 'inside_bacc')}}
            results[f'{feature_name}/{supervision}'] = {'summary': summary, 'rows': details}
            print(feature_name, supervision, json.dumps(summary['small/foldNone']), flush=True)
    report = {'status': 'frozen_readout_complete_not_detection_training', 'selected': selected, 'eligible': len(rows),
              'backbone_control': args.backbone_control,
              'include_s16': args.include_s16,
              'excluded': excluded, 'held_parent_groups': [sorted(h) for h in held_sets], 'ridge_lambda': 10,
              'readout_threshold': .5, 'feature_shapes': rows[0]['shapes'],
              'mean_scramble_changed_fraction': float(np.mean([r['changed_fraction'] for r in rows])),
              'checkpoint': str(checkpoint), 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'results': results,
              'caveats': ['GT-prompt masks and privileged SAM extent shared by all arms',
                          'held parent groups apply only to readout fitting; frozen detector has seen all images',
                          'SAM pseudo-label agreement, not manual boundary accuracy, detection AP or edge restoration',
                          'balanced low-capacity readout; eligibility excludes masks without enough internal background',
                          'bilinear sampling does not recreate lost high-frequency information']}
    (OUT/'internal_shape_readout.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
