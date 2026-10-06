"""Verify matched logs/checkpoints and reproduce both best full-model metrics."""
import json
import torch
import train_m_p32 as job
job.joint.RUN=job.RUN
job.joint.setup=job.setup
job.joint.train_mode=job.train_mode
from preflight_sqer1 import seed_all, sha256_file

seed_all(0)
torch.set_num_threads(4)
report_dir=job.joint.task.engine.ROOT/'reports/M_P32_20261005'
report_dir.mkdir(parents=True,exist_ok=True)
report={'arms':{}}
curves={}; protocols=[]
for arm in ('ORIGINAL','P32'):
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
    reproduced=job.joint.task.evaluate(cfg,source,refiner)
    error=max(abs(a-b) for a,b in zip(reproduced['coco_eval_bbox'],best['validation']['coco_eval_bbox']))
    assert error<.0002,error
    curves[arm]=[r['validation']['coco_eval_bbox'] for r in rows]
    report['arms'][arm]={'best':best,'last':rows[-1], 'reproduced':reproduced,'max_abs_error':error,
                        'checkpoint_sha':sha256_file(root/'best.pth'), 'protocol':state['protocol']}
    source.close()
    del state,cfg,source,refiner,criterion
    torch.cuda.empty_cache()
assert {k:v for k,v in protocols[0].items() if k not in ('arm','ir_p32_enabled')}=={k:v for k,v in protocols[1].items() if k not in ('arm','ir_p32_enabled')}
report['same_epoch']={str(i):{'wins':sum(b[i]>a[i] for a,b in zip(curves['ORIGINAL'],curves['P32'])),
 'mean_delta_pp':sum(b[i]-a[i] for a,b in zip(curves['ORIGINAL'],curves['P32']))*5} for i in (0,2,3,4)}
report['status']='PASS'
job.joint.task.engine.write_json(report_dir/'audit.json',report)
print(json.dumps({'status':'PASS','same_epoch':report['same_epoch'],'errors':{k:v['max_abs_error'] for k,v in report['arms'].items()}}),flush=True)


