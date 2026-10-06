"""Audit real training masks with current SGC eligibility; no detector updates."""
import json
import torch
import train_sbr1 as engine
from preflight_sqer1 import build_config, move_targets


def main():
    torch.set_num_threads(4)
    cfg = build_config(engine.REPO, engine.REPO/'experiments/phase_m/m_layout_recovery_r2_main.yml')
    dl = cfg.yaml_cfg['train_dataloader']
    dl.update(total_batch_size=16, num_workers=0, drop_last=False, shuffle=False, persistent_workers=False)
    dl['collate_fn'] = {'type': 'BatchImageCollateFunction'}
    trans = dl['dataset']['transforms']
    trans['ops'] = [x for x in trans['ops'] if not x['type'].startswith('Random')]
    trans.pop('policy', None)
    from src.zoo.dfine.sam_group_contrast import selection
    report = {'protocol': 'all training images; deterministic training resize; SAM and BOX common eligibility; no model forward', 'images': []}
    for samples, targets in cfg.train_dataloader:
        hw = samples.shape[-2:]
        targets = move_targets(targets, torch.device('cuda'))
        for t in targets:
            row = {'image_id': int(t['image_id'].item()), 'objects': len(t['boxes']), 'scales': {}}
            areas = t['boxes'][:, 2:].prod(-1) * hw[0] * hw[1]
            row['size_group'] = 'empty' if not len(areas) else ('small' if bool((areas < 1024).all()) else 'other')
            for stride in (4, 8, 16, 32):
                size = (hw[0]//stride, hw[1]//stride)
                choices = {arm: selection(t, size, arm) for arm in ('sam', 'box')}
                valid = all(c is not None and len(c[0]) >= 2 and len(c[1]) >= 4 for c in choices.values())
                row['scales'][str(stride)] = {'eligible': valid, 'sam_cells': None if choices['sam'] is None else [len(x) for x in choices['sam']]}
            report['images'].append(row)
        if len(report['images']) % 320 == 0:
            print('audited', len(report['images']), flush=True)
    report['summary'] = {}
    for group in ('all', 'small', 'other', 'empty'):
        rows = [r for r in report['images'] if group == 'all' or r['size_group'] == group]
        report['summary'][group] = {'count': len(rows), 'eligible': {str(s): sum(r['scales'][str(s)]['eligible'] for r in rows) for s in (4,8,16,32)},
            's4_rescue': sum(r['scales']['4']['eligible'] and not r['scales']['8']['eligible'] for r in rows),
            'joint_s4_s8': sum(r['scales']['4']['eligible'] and r['scales']['8']['eligible'] for r in rows)}
    out = engine.ROOT/'reports/SGC_MULTISCALE_20261004'
    out.mkdir(parents=True, exist_ok=True)
    engine.write_json(out/'coverage.json', report)
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
