"""Reproducible descriptive statistics, not a significance/AP test."""
import json
from pathlib import Path
import statistics as stats

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'reports/105_sam_group_contrast_pilot'


def main():
    runs = {s: json.loads((REPORT/f'{s}.json').read_text()) for s in ('initial', 'best')}
    assert runs['initial']['coverage'] == runs['best']['coverage']
    annotations = json.loads((ROOT/'data/antiuav6k_common/annotations/instances_visible_common_train.json').read_text())
    areas = {}
    for a in annotations['annotations']:
        areas.setdefault(a['image_id'], []).append(a['bbox'][2]*a['bbox'][3])
    coverage = runs['initial']['coverage']
    summary = dict(matched_coverage=True, images=len(coverage),
                   unique_images=len({r['image_id'] for r in coverage}), groups={}, states={})
    for group in ('all_positive', 'small_coco', 'medium_coco', 'empty'):
        def include(r):
            area = max(areas.get(r['image_id'], [0]))
            return {'all_positive': area > 0, 'small_coco': 0 < area < 32**2,
                    'medium_coco': 32**2 <= area < 96**2, 'empty': area == 0}[group]
        rows = [r for r in coverage if include(r)]
        summary['groups'][group] = dict(images=len(rows), accepted=sum(r['accepted'] for r in rows),
            sam_usable=sum(r['sam_usable'] for r in rows), box_usable=sum(r['box_usable'] for r in rows),
            common=sum(r['sam_usable'] and r['box_usable'] for r in rows))
    for state, run in runs.items():
        summary['states'][state] = {}
        for arm in ('sam', 'box'):
            rows = [r[arm] for r in run['records'] if r['common_images'] > 0]
            summary['states'][state][arm] = dict(
                loss_mean=stats.mean(r['loss'] for r in rows),
                gradient_cosine_mean=stats.mean(r['gradient_cosine'] for r in rows),
                positive_cosine_batches=sum(r['gradient_cosine'] > 0 for r in rows),
                raw_gradient_ratio_median=stats.median(r['raw_gradient_ratio'] for r in rows),
                separation_mean=stats.mean(v for r in rows for v in r['separation']))
    (REPORT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
