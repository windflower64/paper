"""Compare three completed, matched full RGB-T training arms."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = {
    'SAM': 'C_PLUS_M_SD22_HALF_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV',
    'BOX': 'C_PLUS_M_SD22_HALF_SGC2_BOX_DECAY9_14_B8A4_20E_TESTDEV',
    'NONE': 'C_PLUS_M_SD22_HALF_SGC0_B8A4_20E_TESTDEV',
}
LABELS = ['AP', 'AP50', 'AP75', 'APS', 'APM', 'APL', 'AR1', 'AR10', 'AR100', 'ARS', 'ARM', 'ARL']


def main():
    curves, runs = {}, {}
    for name, run in NAMES.items():
        rows = [json.loads(s) for s in (ROOT / 'outputs' / run / 'seed0/log.txt').read_text().splitlines() if s.strip()]
        assert [r['epoch'] for r in rows] == list(range(20))
        curves[name] = rows
        best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
        runs[name] = {'best_epoch': best['epoch'], 'best': dict(zip(LABELS, best['test_coco_eval_bbox'])),
                      'last': dict(zip(LABELS, rows[-1]['test_coco_eval_bbox'])),
                      'tail6': {k: sum(r['test_coco_eval_bbox'][i] for r in rows[-6:])/6 for i, k in enumerate(LABELS)},
                      'aux_losses': [r.get('train_loss_sgc_group', 0) for r in rows]}
    differences = {}
    for control in ('BOX', 'NONE'):
        differences['SAM_minus_' + control] = {
            point: {k: 100*(runs['SAM'][point][k]-runs[control][point][k]) for k in LABELS}
            for point in ('best', 'last', 'tail6')}
        differences['SAM_minus_' + control]['same_epoch_wins'] = {
            k: sum(a['test_coco_eval_bbox'][i] > b['test_coco_eval_bbox'][i] for a,b in zip(curves['SAM'],curves[control]))
            for i,k in enumerate(LABELS)}
    result = {'runs': runs, 'differences_ap_points': differences,
              'caveats': ['single seed; original test used for development', 'best epochs differ',
                          'epochs correlated; wins and tail means are descriptive, not independent trials'],
              'curves': {name: [dict(epoch=r['epoch'], **dict(zip(LABELS,r['test_coco_eval_bbox']))) for r in rows] for name,rows in curves.items()}}
    dest = ROOT / 'reports/148_sgc2_half_box/result_comparison.json'
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'runs': {k: {p: v[p] for p in ('best_epoch','best','tail6')} for k,v in runs.items()}, 'differences': differences}, indent=2))


if __name__ == '__main__':
    main()
