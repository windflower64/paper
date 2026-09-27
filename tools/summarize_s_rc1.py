"""Reproducible training and inference readout; AP values in native 0--1 units."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/104_sam3_role_control'


def main():
    result={}
    for arm,prefix in [('C','C_ONLY_GQ1'),('NONE','S_RC1_NONE'),('BOX','S_RC1_BOX'),('SAM','S_RC1_SAM'),('EDGE','S_RC1_EDGE')]:
        run=ROOT/f'outputs/{prefix}_B8A4_20E_TESTDEV/seed0'
        rows=[json.loads(line) for line in (run/'log.txt').read_text().splitlines() if line.strip()]
        assert [r['epoch'] for r in rows]==list(range(20))
        best=max(rows,key=lambda r:r['test_coco_eval_bbox'][0])
        entry=dict(best_epoch=best['epoch'],best=best['test_coco_eval_bbox'],last=rows[-1]['test_coco_eval_bbox'],
                   tail5_ap=sum(r['test_coco_eval_bbox'][0] for r in rows[-5:])/5)
        if arm!='C':
            entry['aux_first_last']=[rows[i].get('train_loss_mfam_region') for i in [0,-1]]
            entry['preflight']=json.loads((REPORT/f'preflight_{arm.lower()}.json').read_text())
            entry['interventions']={p.stem:json.loads(p.read_text()) for p in (REPORT/'interventions'/arm.lower()).glob('*.json')}
        result[arm]=entry
    (REPORT/'analysis_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for arm,r in result.items(): print(arm,'epoch',r['best_epoch'],'AP/AP75/APs/APm',[round(100*r['best'][i],4) for i in [0,2,3,4]],'tail5',round(100*r['tail5_ap'],4),flush=True)


if __name__=='__main__':main()
