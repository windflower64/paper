"""Full development-set intervention on the added S16 score contribution."""
import json
import torch
import train_sbr3_context as task
from preflight_sqer1 import seed_all,sha256_file

seed_all(0); torch.set_num_threads(4)
outdir=task.joint.task.engine.ROOT/'reports/SBR3_CONTEXT_20261005'
report={}
for tag in ('best','last'):
    cfg,source,refiner,_=task.setup(16,'CONTEXT')
    path=task.RUN/'CONTEXT'/f'{tag}.pth'
    saved=torch.load(path,map_location='cpu')
    source.model.load_state_dict(saved['base'],strict=True)
    refiner.load_state_dict(saved['refiner'],strict=True)
    source.model.eval(); refiner.eval()
    reference=saved['last_row']['validation']
    if tag=='last':
        fresh=task.evaluate(cfg,source,refiner)
        assert max(abs(a-b) for a,b in zip(reference['coco_eval_bbox'],fresh['coco_eval_bbox']))<.0002
    hook=refiner.context_score.register_forward_hook(lambda m,a,o:torch.zeros_like(o))
    try: disabled=task.evaluate(cfg,source,refiner)
    finally: hook.remove()
    report[tag]={'epoch':saved['epoch'],'checkpoint_sha':sha256_file(path),'normal':reference,
                 'context_disabled':disabled,
                 'enabled_minus_disabled_pp':[(a-b)*100 for a,b in zip(reference['coco_eval_bbox'],disabled['coco_eval_bbox'])]}
    task.joint.task.engine.write_json(outdir/'context_dependence.json',report)
    print(json.dumps({'checkpoint':tag,'delta_pp':report[tag]['enabled_minus_disabled_pp'][:5]}),flush=True)
    source.close(); del cfg,source,refiner,saved
    torch.cuda.empty_cache()
