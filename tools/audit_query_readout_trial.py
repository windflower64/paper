import hashlib
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/'reports/137_query_readout_trial'

def main():
    dest=REPORT/'result_audit.json'
    if dest.exists():raise RuntimeError('Existing audit')
    manifest=json.loads((REPORT/'manifest.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    data={};results={}
    for arm in ('none','box','sam'):
        run=ROOT/f'outputs/QUERY_READ_{arm.upper()}_B16A2_10E_TESTDEV/seed0'
        rows=[json.loads(s) for s in (run/'log.txt').read_text().splitlines() if s.strip()]
        logs=[json.loads(s) for s in (run/'guidance_batches.jsonl').read_text().splitlines() if s.strip()]
        assert [r['epoch'] for r in rows]==list(range(10))
        assert [(t['epoch'],t['step']) for t in logs]==[(e,s) for e in range(10) for s in range(200)]
        assert all(0<=t['eligible']<=t['batch']==16 and math.isfinite(t['sam_geometry']) and math.isfinite(t['box_geometry']) for t in logs)
        evaluation=json.loads((REPORT/f'eval_{arm}.json').read_text())
        assert evaluation['status']=='PASS' and evaluation['max_abs_error']<.0002
        assert hashlib.sha256((run/'best_stg1.pth').read_bytes()).hexdigest()==evaluation['checkpoint_sha256']
        means={str(e):{key:sum(t[key] for t in logs if t['epoch']==e)/200 for key in ('sam_geometry','box_geometry','eligible')} for e in range(10)}
        results[arm]={'evaluation':evaluation,'telemetry_means':means,
            'curve':[{k:r[k] for k in ('epoch','test_coco_eval_bbox')} for r in rows]}
        data[arm]=rows
    pairs={}
    for a,b in [('sam','none'),('sam','box'),('box','none')]:
        pairs[a+'-'+b]={}
        for label,lo in [('all',0),('last3',7)]:
            diffs=[[100*(data[a][e]['test_coco_eval_bbox'][i]-data[b][e]['test_coco_eval_bbox'][i]) for i in range(12)] for e in range(lo,10)]
            pairs[a+'-'+b][label]={'mean':[sum(r[i] for r in diffs)/len(diffs) for i in range(12)],
                'positive_epochs':[sum(r[i]>0 for r in diffs) for i in range(12)]}
    dest.write_text(json.dumps({'status':'PASS','manifest_count':len(manifest),'runs':results,'pairs':pairs},indent=2),encoding='utf-8')
    for name,pair in pairs.items():
        print(name,{k:{'AP':v['mean'][0],'AP75':v['mean'][2],'APS':v['mean'][3],'APS_positive':v['positive_epochs'][3]} for k,v in pair.items()})
    for arm,r in results.items():print(arm,'geometry epoch0/9',r['telemetry_means']['0'],r['telemetry_means']['9'])
    print('verified',len(manifest))
if __name__=='__main__':main()
