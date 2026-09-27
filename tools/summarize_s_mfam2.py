"""Training/EMA intervention summary with explicit completeness checks."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    rows={}
    names={'C':'C_ONLY_GQ1_B8A4_20E_TESTDEV',
           'MFAM1_SAM':'S_MFAM1_C_SAM_B8A4_20E_TESTDEV',
           'MFAM1_BOX':'S_MFAM1_C_BOX_B8A4_20E_TESTDEV',
           'SAM':'S_MFAM2_C_SAM_B8A4_20E_TESTDEV',
           'BOX':'S_MFAM2_C_BOX_B8A4_20E_TESTDEV'}
    for arm,name in names.items():
        history=[json.loads(line) for line in (ROOT/'outputs'/name/'seed0/log.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
        assert [r['epoch'] for r in history]==list(range(20))
        best=max(history,key=lambda r:r['test_coco_eval_bbox'][0])
        rows[arm]=dict(best_epoch=best['epoch'],best=best['test_coco_eval_bbox'],
                       last=history[-1]['test_coco_eval_bbox'],
                       tail5_AP=sum(r['test_coco_eval_bbox'][0] for r in history[-5:])/5)
        if arm in ('SAM','BOX'):
            checks={}
            for mode in ('learned','support_shift','shape_shift','shape_mean','disabled'):
                path=ROOT/f'reports/99_s_mfam2/interventions/{arm}/{mode}.json'
                if path.exists():checks[mode]=json.loads(path.read_text())['coco_eval_bbox']
            if 'learned' in checks:assert max(abs(a-b) for a,b in zip(rows[arm]['best'],checks['learned']))<1e-10
            rows[arm]['interventions']=checks
            rows[arm]['checks_complete']=len(checks)==5
    (ROOT/'reports/99_s_mfam2/analysis_summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    for arm,row in rows.items():
        print(arm,'best',row['best'][:5],'tail5',row['tail5_AP'])
        for mode,values in row.get('interventions',{}).items():print(mode,values[:5])


if __name__=='__main__':main()
