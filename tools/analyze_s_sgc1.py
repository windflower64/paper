"""Log-grounded SGC1 analysis; AP fractions retained in JSON."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/105_sam_group_contrast_pilot'


def main():
    result={'protocol':'20 epochs, seed0, batch8 accumulation4; original test1820 is development validation', 'runs':{}}
    for name,folder in [('C','C_ONLY_GQ1'),('SAM','S_SGC1_SAM'),('BOX','S_SGC1_BOX')]:
        path=ROOT/f'outputs/{folder}_B8A4_20E_TESTDEV/seed0/log.txt'
        rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        assert [r['epoch'] for r in rows]==list(range(20))
        best=max(rows,key=lambda r:r['test_coco_eval_bbox'][0])
        result['runs'][name]=dict(log=str(path),log_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            epochs=20,best_epoch=best['epoch'],best_metrics=best['test_coco_eval_bbox'],
            last_ap=rows[-1]['test_coco_eval_bbox'][0],
            tail5_ap=statistics.mean(r['test_coco_eval_bbox'][0] for r in rows[-5:]),
            aux_first=rows[0].get('train_loss_sgc_group'),aux_last=rows[-1].get('train_loss_sgc_group'),
            trajectory=[dict(epoch=r['epoch'],ap=r['test_coco_eval_bbox'][0],
                             aux=r.get('train_loss_sgc_group')) for r in rows])
    for name in ('SAM','BOX'):
        r=result['runs'][name]; c=result['runs']['C']
        r['delta_best_metrics_vs_C_pp']=[100*(a-b) for a,b in zip(r['best_metrics'],c['best_metrics'])]
        r['delta_tail5_vs_C_pp']=100*(r['tail5_ap']-c['tail5_ap'])
        r['aux_reduction_pct']=100*(1-r['aux_last']/r['aux_first'])
    result['sam_minus_box_best_metrics_pp']=[100*(a-b) for a,b in zip(result['runs']['SAM']['best_metrics'],result['runs']['BOX']['best_metrics'])]
    (REPORT/'training_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for name,r in result['runs'].items():
        print(name,json.dumps({k:v for k,v in r.items() if k not in ('trajectory','log_sha256','log')}))


if __name__=='__main__': main()
