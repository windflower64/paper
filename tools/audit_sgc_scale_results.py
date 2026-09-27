"""Read-only source audit and fixed same-epoch comparison for report133."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/133_sgc_scale_training'

def main():
    out=REPORT/'result_audit.json'
    if out.exists():raise RuntimeError('Refusing to overwrite audit')
    manifest=json.loads((REPORT/'manifest.json').read_text(encoding='utf-8'))
    changed=[p for p,h in manifest.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    runs={};metadata=[]
    for arm in ('none','box','sam'):
        run=ROOT/f'outputs/SGC_SCALE_{arm.upper()}_B16A2_20E_TESTDEV/seed0'
        path=run/'log.txt'
        rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
        assert [r['epoch'] for r in rows]==list(range(20))
        assert all(len(r['test_coco_eval_bbox'])==12 for r in rows)
        assert all(math.isfinite(v) and (-1<=v<=1) for r in rows for v in r['test_coco_eval_bbox'])
        assert all('train_loss_sgc_group' in r and r['train_loss_sgc_group']>0 for r in rows[:14])
        assert all('train_loss_sgc_group' not in r for r in rows[14:])
        evaluation=json.loads((REPORT/arm/'evaluation/best_ema.json').read_text(encoding='utf-8'))
        best=max(rows,key=lambda r:r['test_coco_eval_bbox'][0])
        assert evaluation['status']=='PASS' and evaluation['max_abs_error']==0
        assert evaluation['coco_eval_bbox']==best['test_coco_eval_bbox']
        assert hashlib.sha256((run/'best_stg1.pth').read_bytes()).hexdigest()==evaluation['checkpoint_sha256']
        runs[arm]=rows
        metadata.append({'file':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    comparisons={}
    for a,b in [('sam','none'),('box','none'),('sam','box')]:
        row={}
        for label,lo,hi in [('all20',0,20),('active14',0,14),('last6',14,20)]:
            differences=[[100*(runs[a][e]['test_coco_eval_bbox'][i]-runs[b][e]['test_coco_eval_bbox'][i]) for i in range(12)] for e in range(lo,hi)]
            row[label]={'mean_delta_points':[sum(d[i] for d in differences)/(hi-lo) for i in range(12)],
                'positive_epochs':[sum(d[i]>0 for d in differences) for i in range(12)]}
        row['same_epoch_8_10']={str(e):{arm:runs[arm][e]['test_coco_eval_bbox'] for arm in runs} for e in (8,10)}
        comparisons[f'{a}_minus_{b}']=row
    result={'status':'PASS','manifest_files_verified':len(manifest),'log_metadata':metadata,
            'epochs_per_arm':20,'independent_evaluation_max_error':0,'schedule_verified':True,
            'comparisons':comparisons,'note':'Epochs are correlated, not independent seeds; -1 large-target metrics unavailable.'}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    for name,row in comparisons.items():
        print(name)
        for window in ('all20','active14','last6'):
            r=row[window];print(window,{k:round(r['mean_delta_points'][i],4) for k,i in [('AP',0),('AP75',2),('APS',3),('APM',4)]},'APS positive',r['positive_epochs'][3])
    print('SAME_EPOCH',json.dumps(comparisons['sam_minus_box']['same_epoch_8_10']))
    print('FILES_VERIFIED',len(manifest))

if __name__=='__main__':main()
