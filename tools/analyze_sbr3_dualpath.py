"""Audit completed dual-path training and independently evaluate saved best."""
import json
import torch
import train_sbr3_dualpath as job
from preflight_sqer1 import seed_all,sha256_file

seed_all(0); torch.set_num_threads(4)
root=job.RUN/'DUAL'
rows=[json.loads(x) for x in (root/'log.jsonl').read_text().splitlines()]
assert [r['epoch'] for r in rows]==list(range(20))
assert all(r['validation']['images']==1820 for r in rows)
best=max(rows,key=lambda r:r['validation']['coco_eval_bbox'][0])
state=torch.load(root/'best.pth',map_location='cpu')
assert state['last_row']==best and state['epoch']==best['epoch']
cfg,source,refiner,criterion=job.setup(16,'DUAL')
source.model.load_state_dict(state['base'],strict=True)
refiner.load_state_dict(state['refiner'],strict=True)
source.model.eval()
fresh=job.evaluate(cfg,source,refiner)
error=max(abs(a-b) for a,b in zip(fresh['coco_eval_bbox'],best['validation']['coco_eval_bbox']))
assert error<.0002,error
# Same saved checkpoint, extra score enabled: diagnostic only, not new selection.
enabled=job.context.evaluate(cfg,source,refiner)
old=json.loads((job.joint.task.engine.ROOT/'reports/SBR3_CONTEXT_20261005/context_dependence.json').read_text())['best']['context_disabled']['coco_eval_bbox']
detail=[json.loads(x) for x in (job.context.RUN/'DETAIL/log.jsonl').read_text().splitlines()]
report=dict(status='PASS',best=best,last=rows[-1],reproduced=fresh,max_abs_error=error,
            checkpoint_sha=sha256_file(root/'best.pth'),protocol=state['protocol'],
            enabled_diagnostic=enabled,
            best_minus_previous_candidate_pp=[100*(a-b) for a,b in zip(fresh['coco_eval_bbox'],old)],
            same_epoch_vs_detail={str(i):dict(wins=sum(r['validation']['coco_eval_bbox'][i]>d['validation']['coco_eval_bbox'][i] for r,d in zip(rows,detail)),
                mean_delta_pp=sum(r['validation']['coco_eval_bbox'][i]-d['validation']['coco_eval_bbox'][i] for r,d in zip(rows,detail))*5) for i in (0,2,3,4)},
            training_on_lower_than_off_epochs=sum(r['train']['geometry_on']<r['train']['geometry_off'] for r in rows))
directory=job.joint.task.engine.ROOT/'reports/SBR3_DUALPATH_20261006'
directory.mkdir(parents=True,exist_ok=True)
job.joint.task.engine.write_json(directory/'audit.json',report)
print(json.dumps(report),flush=True)
source.close()
