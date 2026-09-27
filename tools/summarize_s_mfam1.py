"""Summarize matched training histories and completed same-weight checks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    summary = {}
    for arm, name in [('C','C_ONLY_GQ1_B8A4_20E_TESTDEV'),
                      ('SAM','S_MFAM1_C_SAM_B8A4_20E_TESTDEV'),
                      ('BOX','S_MFAM1_C_BOX_B8A4_20E_TESTDEV')]:
        rows = [json.loads(line) for line in (ROOT/'outputs'/name/'seed0/log.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
        assert [row['epoch'] for row in rows] == list(range(20))
        best = max(rows,key=lambda r:r['test_coco_eval_bbox'][0])
        summary[arm] = dict(best_epoch=best['epoch'],best=best['test_coco_eval_bbox'],
                            last=rows[-1]['test_coco_eval_bbox'],
                            tail5_AP=sum(r['test_coco_eval_bbox'][0] for r in rows[-5:])/5,
                            region_loss_start=rows[0].get('train_loss_mfam_region'),
                            region_loss_end=rows[-1].get('train_loss_mfam_region'))
        if arm != 'C':
            checks = {}
            for mode in ('learned','constant','disabled','shifted','mean'):
                path = ROOT/f'reports/98_s_mfam1/interventions/{arm}/{mode}.json'
                if path.exists():
                    checks[mode] = json.loads(path.read_text(encoding='utf-8'))['coco_eval_bbox']
            if 'learned' in checks:
                assert max(abs(a-b) for a,b in zip(checks['learned'],summary[arm]['best'])) < 1e-10
            summary[arm]['interventions'] = checks
    report = ROOT/'reports/98_s_mfam1/analysis_summary.json'
    report.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    for arm, row in summary.items():
        print(arm,'best',row['best'][:5], 'tail5',row['tail5_AP'])
        for mode, values in row.get('interventions',{}).items():
            print(' ',mode,values[:5])


if __name__ == '__main__':
    main()
