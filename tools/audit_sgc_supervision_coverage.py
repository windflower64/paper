"""Read-only training-label coverage audit; no detector/SAM inference or training.

Reuse production SGC selection, check all images and both horizontal orientations.
S4/S16 rows are counterfactual sampling diagnostics, not new model results.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / 'D-FINE/src/zoo/dfine/sam_group_contrast.py'
ANNOTATION = ROOT / 'data/antiuav6k_common/annotations/instances_visible_common_train.json'
MASK_ROOT = ROOT / 'reports/104_sam3_role_control/masks_train'


def fingerprint(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def group_for(area):
    for bound, label in [(64, 'lt8'), (256, '8to16'), (576, '16to24'),
                         (1024, '24to32'), (9216, '32to96')]:
        if area < bound:
            return label
    return 'ge96'


def summarize(rows):
    output = {}
    for name in ['all_positive', 'small', 'medium', 'lt8', '8to16', '16to24',
                 '24to32', '32to96', 'ge96']:
        selected = [r for r in rows if r['objects'] == 1 and
                    (name == 'all_positive' or
                     name == 'small' and r['area'] < 1024 or
                     name == 'medium' and 1024 <= r['area'] < 9216 or
                     name == r['size_bin'])]
        result = {'positive_images': len(selected),
                  'teacher_accepted': sum(r['teacher_accepted'] for r in selected),
                  'teacher_rejected': sum(not r['teacher_accepted'] for r in selected)}
        for level in ['S4', 'S8', 'S16']:
            eligible = [r for r in selected if level in r]
            values = [r[level] for r in eligible]
            result[level] = {
                'evaluated_accepted': len(values),
                'sam_usable': sum(v['sam_usable'] for v in values),
                'box_usable': sum(v['box_usable'] for v in values),
                'joint_usable': sum(v['joint_usable'] for v in values),
                'joint_rate_all_positive': (sum(v['joint_usable'] for v in values)
                                            / len(selected) if selected else None),
                'sam_fg_lt2': sum(v['sam_positive'] < 2 for v in values),
                'sam_fg_zero': sum(v['sam_positive'] == 0 for v in values),
                'sam_bg_lt4': sum(v['sam_negative'] < 4 for v in values),
                'box_fg_lt2': sum(v['box_positive'] < 2 for v in values),
                'box_bg_lt4': sum(v['box_negative'] < 4 for v in values),
                'exclusive_reason_counts': dict(Counter(v['reason'] for v in values)),
                'sam_positive_quantiles': (np.quantile([v['sam_positive'] for v in values],
                                            [0, .25, .5, .75, 1]).tolist() if values else []),
                'mixed_mass_fraction_mean': (float(np.mean([v['mixed_mass_fraction']
                                                        for v in values])) if values else None),
            }
        output[name] = result
    return output


def measure(prod, target, size):
    sam = prod.selection(target, size, 'sam')
    box = prod.selection(target, size, 'box')
    if sam is None or box is None:
        raise RuntimeError('Accepted target unexpectedly rejected by production')
    sp, sn = map(len, sam)
    bp, bn = map(len, box)
    su, bu = sp >= 2 and sn >= 4, bp >= 2 and bn >= 4
    field = target['masks'].float().amax(0)
    occ = F.interpolate(field[None, None], size=size, mode='area')[0, 0]
    assert torch.isfinite(occ).all() and occ.min() >= 0 and occ.max() <= 1
    mixed = (occ > .05) & (occ < .75)
    return {
        'sam_positive': sp, 'sam_negative': sn, 'box_positive': bp, 'box_negative': bn,
        'sam_usable': su, 'box_usable': bu, 'joint_usable': su and bu,
        'reason': ('usable' if su and bu else 'both_sam_box_insufficient' if not su and not bu
                   else 'sam_only_insufficient' if not su else 'box_only_insufficient'),
        'occupied_cells': int((occ > 0).sum()), 'mixed_cells': int(mixed.sum()),
        'mixed_mass_fraction': float(occ[mixed].sum() / occ.sum().clamp_min(1e-12)),
        'mask_mass_grid_units': float(occ.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.relative_to((ROOT / 'reports').resolve())
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('sgc_production_audit', PRODUCTION)
    prod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prod)
    data = json.loads(ANNOTATION.read_text(encoding='utf-8'))
    raw_records = json.loads((MASK_ROOT / 'records.json').read_text(encoding='utf-8'))
    records = {int(r['image_id']): r for r in raw_records}
    assert len(records) == len(raw_records), 'Duplicate mask records'
    assert len({x['id'] for x in data['images']}) == len(data['images'])
    assert len({x['id'] for x in data['annotations']}) == len(data['annotations'])
    assert set(records) == {x['id'] for x in data['images']}
    by_image = {x['id']: [] for x in data['images']}
    for ann in data['annotations']:
        assert ann['image_id'] in by_image
        assert not ann.get('iscrowd', 0)
        by_image[ann['image_id']].append(ann)
    rows, flip_differences, mask_manifest = [], [], []
    images = data['images'][:args.limit] if args.limit else data['images']
    for index, im in enumerate(images):
        image_id = im['id']
        assert (im['height'], im['width']) == (512, 640), im
        ann = by_image[image_id]
        record = records[image_id]
        assert record['file_name'] == im['file_name']
        row = {'image_id': image_id, 'file_name': im['file_name'], 'objects': len(ann),
               'teacher_accepted': bool(record['accepted']),
               'quality_record': record}
        if not ann:
            assert not record['accepted']
            row['reason'] = 'empty_gt_detection_only'
            rows.append(row)
            continue
        assert len(ann) == 1, 'Multi-object handling must be separately designed'
        a = ann[0]
        x, y, w, h = a['bbox']
        assert all(math.isfinite(v) for v in [x, y, w, h, a['area']])
        assert w >= 1 and h >= 1 and x >= 0 and y >= 0
        assert x + w <= 640 + 1e-4 and y + h <= 512 + 1e-4
        assert abs(a['area'] - w*h) < 1e-4
        row.update(area=a['area'], bbox=a['bbox'], size_bin=group_for(a['area']))
        if not record['accepted']:
            row['reason'] = 'teacher_rejected'
            rows.append(row)
            continue
        assert float(record['supervision_weight']) == 1.0
        mask_path = MASK_ROOT / 'masks' / f'{image_id:06d}.png'
        mask_manifest.append(fingerprint(mask_path))
        with Image.open(mask_path) as image:
            assert image.size == (640, 512)
            array = np.array(image.convert('L'), copy=True)
        assert set(np.unique(array)).issubset({0, 1, 255})
        mask = torch.from_numpy(array > 0).to(torch.uint8)[None]
        assert mask.sum() > 0, 'Accepted empty mask'
        row['mask_pixels'] = int(mask.sum())
        # Mirror float32 XYXY then convert, matching torchvision box transforms.
        xyxy = torch.tensor([[x, y, x+w, y+h]], dtype=torch.float32)
        boxes = torch.cat(((xyxy[:, :2] + xyxy[:, 2:]) / 2,
                           xyxy[:, 2:] - xyxy[:, :2]), dim=1) / torch.tensor([640, 512, 640, 512])
        target = {'masks': mask, 'boxes': boxes, 'sam_quality': torch.tensor([1.])}
        for stride in [4, 8, 16]:
            row[f'S{stride}'] = measure(prod, target, (512//stride, 640//stride))
        mirrored = xyxy.clone()
        mirrored[:, 0], mirrored[:, 2] = 640-xyxy[:, 2], 640-xyxy[:, 0]
        flip_boxes = torch.cat(((mirrored[:, :2] + mirrored[:, 2:])/2,
                                mirrored[:, 2:] - mirrored[:, :2]), dim=1) / torch.tensor([640,512,640,512])
        flipped = measure(prod, {'masks': mask.flip(-1), 'boxes': flip_boxes,
                                'sam_quality': torch.tensor([1.])}, (64,80))
        row['S8_flipped'] = flipped
        count_keys = ['sam_positive','sam_negative','box_positive','box_negative','joint_usable']
        if any(flipped[k] != row['S8'][k] for k in count_keys):
            flip_differences.append({'image_id': image_id, 'normal': row['S8'], 'flip': flipped})
        rows.append(row)
        if (index+1) % 200 == 0:
            print(json.dumps({'processed': index+1, 'total': len(images)}, ensure_ascii=False), flush=True)
    summary = {
        'status': 'complete', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': 'all train labels, production selection; no model forward; native and flipped S8',
        'caveats': ['teacher acceptance is heuristic, not pixel accuracy',
                    'S4/S16 coverage is counterfactual, not detection performance',
                    'no historical stochastic batch order or gradient magnitude reconstruction',
                    'coverage cannot explain C versus C+M differences by itself'],
        'software': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__},
        'sources': [fingerprint(ANNOTATION), fingerprint(MASK_ROOT/'records.json'), fingerprint(PRODUCTION), fingerprint(Path(__file__))],
        'images': len(rows), 'positives': sum(r['objects'] == 1 for r in rows),
        'empty': sum(r['objects'] == 0 for r in rows), 'accepted_mask_files': len(mask_manifest),
        'accepted_mask_total_bytes': sum(m['bytes'] for m in mask_manifest),
        'flip_count_differences': len(flip_differences),
        'flip_eligibility_differences': sum(r['normal']['joint_usable'] != r['flip']['joint_usable'] for r in flip_differences),
        'groups': summarize(rows),
    }
    for name, obj in [('summary.json', summary), ('per_image.json', rows),
                      ('mask_manifest.json', mask_manifest), ('flip_differences.json', flip_differences)]:
        with (destination/name).open('x', encoding='utf-8') as handle:
            json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
