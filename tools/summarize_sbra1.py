"""Audit complete SBRA1 training logs without claiming independent inference."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
def main():
    results={}; manifests=[]; logs={}
    for arm in ('sam','none','box'):
        run=ROOT/f'outputs/S_BRA1_{arm.upper()}_B8A4_20E_TESTDEV/seed0'
        rows=[json.loads(line) for line in (run/'log.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
        assert [r['epoch'] for r in rows]==list(range(20))
        assert np.isfinite([r['test_coco_eval_bbox'] for r in rows]).all()
        best=max(rows,key=lambda r:r['test_coco_eval_bbox'][0])
        results[arm]=dict(best_epoch=best['epoch'],metrics=best['test_coco_eval_bbox'],
            tail5_ap=float(np.mean([r['test_coco_eval_bbox'][0] for r in rows[-5:]])),
            final_ap=rows[-1]['test_coco_eval_bbox'][0],first_aux=rows[0]['train_loss_sbra_relation'],last_aux=rows[-1]['train_loss_sbra_relation'])
        manifests.append(json.loads((run/'artifacts/manifest.json').read_text(encoding='utf-8')))
        logs[arm]=rows
    assert manifests[0]==manifests[1]==manifests[2]
    comparison=dict(best_ap_points=100*(results['sam']['metrics'][0]-results['none']['metrics'][0]),
        ap75_points=100*(results['sam']['metrics'][2]-results['none']['metrics'][2]),
        aps_points=100*(results['sam']['metrics'][3]-results['none']['metrics'][3]),
        tail5_ap_points=100*(results['sam']['tail5_ap']-results['none']['tail5_ap']),
        same_epoch_ap_wins=sum(a['test_coco_eval_bbox'][0]>b['test_coco_eval_bbox'][0] for a,b in zip(logs['sam'],logs['none'])))
    out=ROOT/'reports/119_sbra1/final';out.mkdir(parents=True,exist_ok=True)
    result=dict(status='complete_log_audit',all_epochs_present=True,manifest_identical=True,results=results,sam_minus_none=comparison,
       caveats=['No independent checkpoint re-evaluation or intervention performed in this audit','Single seed; epochs are correlated, not independent repetitions','Original test is development validation','Auxiliary soft cross-entropy has nonzero target entropy; flat loss alone does not prove no learning'])
    (out/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
