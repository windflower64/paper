"""Post-hoc nested descriptor audit; no new detector inference or threshold search."""
import json
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/150_sam_local_geometry/attempt02'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-video', action='store_true')
    args = parser.parse_args()
    data = json.loads((OUT / 'teacher_geometry_pilot.json').read_text(encoding='utf-8'))
    samples = data['samples']
    if args.parent_video:
        for s in samples:
            s['sequence'] = '_'.join(s['sequence'].split('_')[:2])
        groups = sorted({s['sequence'] for s in samples})
        np.random.default_rng(20260916).shuffle(groups)
        data['held_sequences_by_fold'] = [groups[f::4] for f in range(4)]
    assert len(samples) == len({s['image_id'] for s in samples}) == 600
    assert max(abs(s['candidates'][0]['delta_iou']) for s in samples) < 1e-10
    variants = {
        'extent_only': ['extent'],
        'extent_RGB': ['extent', 'RGB'],
        'extent_mask': ['extent', 'mask'],
        'extent_contour': ['extent', 'contour'],
        'extent_RGB_contour': ['extent', 'RGB', 'contour'],
        'extent_full_SAM': ['extent', 'SAM'],
        'shape_only': ['SAM'],
        'RGB_only': ['RGB'],
        'RGB_shape': ['RGB', 'SAM'],
    }
    results = {}
    for label, fields in variants.items():
        def vector(c):
            extra = []
            for field in fields:
                extra += c['SAM'][:7] if field == 'mask' else c['SAM'][7:] if field == 'contour' else c[field]
            return c['geometry'] + [v*v for v in c['geometry'][:4]] + extra
        rows = []
        for fold, held in enumerate(data['held_sequences_by_fold']):
            held = set(held)
            train = [s for s in samples if s['sequence'] not in held]
            test = [s for s in samples if s['sequence'] in held]
            assert not ({s['sequence'] for s in train} & {s['sequence'] for s in test})
            X = np.asarray([vector(c) for s in train for c in s['candidates']])
            Y = np.asarray([c['delta_iou'] for s in train for c in s['candidates']])
            mu, std = X.mean(0), np.maximum(X.std(0), 1e-6)
            Z = np.c_[(X-mu)/std, np.ones(len(X))]
            penalty = np.eye(Z.shape[1]); penalty[-1, -1] = 0
            beta = np.linalg.solve(Z.T@Z + 10*penalty, Z.T@Y)
            for s in test:
                z = np.asarray([vector(c) for c in s['candidates']])
                predicted = np.c_[(z-mu)/std, np.ones(len(z))]@beta
                predicted -= predicted[0]
                choice = int(predicted.argmax()) if predicted.max() > .005 else 0
                rows.append({'image_id': s['image_id'], 'fold': fold, 'scale': s['scale'],
                             'gain': s['candidates'][choice]['delta_iou'], 'chosen': choice})
        results[label] = {'rows': rows}
    if not args.parent_video:
        for label, v in data['results'].items():
            results['original_' + label] = {'rows': v['rows']}
    for label, result in results.items():
        rows = result['rows']
        assert len(rows) == len({r['image_id'] for r in rows}) == 600
        result['groups'] = {}
        for scale in ('all', 'small', 'other'):
            for fold in (None, 0, 1, 2, 3):
                rr = [r for r in rows if (scale == 'all' or r['scale'] == scale) and (fold is None or r['fold'] == fold)]
                result['groups'][f'{scale}/fold{fold}'] = {
                    'n': len(rr), 'mean_gain': float(np.mean([r['gain'] for r in rr])),
                    'moved': sum(r['chosen'] != 0 for r in rr),
                    'improved_gt_001': sum(r['gain'] > .01 for r in rr),
                    'worsened_lt_minus001': sum(r['gain'] < -.01 for r in rr),
                }
    output = {'scope': 'post-hoc nested descriptor audit, fixed lambda/threshold; not AP or student',
              'grouping': 'date_time parent video' if args.parent_video else 'filename segment prefix',
              'held_groups': data['held_sequences_by_fold'],
              'sanity': {'unique_samples': 600, 'zero_candidate_exact': True,
                         'sequence_disjoint': True, 'each_evaluated_once': True}, 'results': results}
    name = 'parent_video_audit.json' if args.parent_video else 'nested_descriptor_audit.json'
    (OUT / name).write_text(json.dumps(output, indent=2), encoding='utf-8')
    for label, result in results.items():
        g = result['groups']
        print(label, json.dumps({'all': g['all/foldNone'], 'small': g['small/foldNone'],
                                'small_fold_gains': [g[f'small/fold{f}']['mean_gain'] for f in range(4)]}))


if __name__ == '__main__':
    main()
