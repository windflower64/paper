"""Verify matched logs/checkpoints and reproduce both best full-model metrics."""
import json
import torch
import train_scale_combination as job
from preflight_sqer1 import seed_all, sha256_file

seed_all(0)
torch.set_num_threads(4)
report_dir=job.joint.task.engine.ROOT/'reports/SCALE_COMBINATION_20261005'
report_dir.mkdir(parents=True,exist_ok=True)
report={'arms':{}}
curves={}; protocols=[]
for arm in ('S8','MULTISCALE'):
    root=job.RUN/arm
    rows=[json.loads(x) for x in (root/'log.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows]==list(range(20))
    assert all(r['validation']['images']==1820 for r in rows)
    best=max(rows,key=lambda r:r['validation']['coco_eval_bbox'][0])
    state=torch.load(root/'best.pth',map_location='cpu')
    assert state['last_row']==best and state['epoch']==best['epoch']
    protocols.append(state['protocol'])
    cfg,source,refiner,criterion=job.setup(16,arm)
    source.model.load_state_dict(state['base'],strict=True)
    refiner.load_state_dict(state['refiner'],strict=True)
    source.model.eval()
    reproduced=job.evaluate(cfg,source,refiner)
    error=max(abs(a-b) for a,b in zip(reproduced['coco_eval_bbox'],best['validation']['coco_eval_bbox']))
    assert error<.0002,error
    off_error=max(abs(a-b) for a,b in zip(reproduced['s16_disabled_diagnostic']['coco_eval_bbox'],best['validation']['s16_disabled_diagnostic']['coco_eval_bbox']))
    assert off_error<.0002,off_error
    curves[arm]=[r['validation']['coco_eval_bbox'] for r in rows]
    report['arms'][arm]={'best':best,'last':rows[-1], 'reproduced':reproduced,'max_abs_error':error,
                        'checkpoint_sha':sha256_file(root/'best.pth'), 'protocol':state['protocol']}
    source.close()
    del state,cfg,source,refiner,criterion
    torch.cuda.empty_cache()
assert {k:v for k,v in protocols[0].items() if k not in ('arm','sgc_scales')}=={k:v for k,v in protocols[1].items() if k not in ('arm','sgc_scales')}
report['same_epoch']={str(i):{'wins':sum(b[i]>a[i] for a,b in zip(curves['S8'],curves['MULTISCALE'])),
 'mean_delta_pp':sum(b[i]-a[i] for a,b in zip(curves['S8'],curves['MULTISCALE']))*5} for i in (0,2,3,4)}
report['status']='PASS'
job.joint.task.engine.write_json(report_dir/'audit.json',report)
print(json.dumps({'status':'PASS','same_epoch':report['same_epoch'],'errors':{k:v['max_abs_error'] for k,v in report['arms'].items()}}),flush=True)


