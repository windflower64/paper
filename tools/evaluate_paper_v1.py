"""Frozen PAPER_V1 evaluation: complete RGB-T checkpoint, S16 score disabled."""
import argparse
import torch
from pathlib import Path
import train_sbr3_context as task
from preflight_sqer1 import seed_all,sha256_file

SHA='351edff4f2fb1dd9010b5064505101ed63e629b5d25a4c6155a0742385801829'
EXPECTED=[.5485778850860029,.9424335540545905,.6022198021969847,.5031891403125792,.5697800173202506,-1.,.5975162625665287,.6286812536960379,.6713187463039623,.6429357798165138,.6848167539267015,-1.]

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=task.RUN/'CONTEXT/best.pth')
    args=p.parse_args()
    assert sha256_file(args.checkpoint)==SHA,'Not the frozen PAPER_V1 checkpoint'
    seed_all(0);torch.set_num_threads(4)
    cfg,source,refiner,_=task.setup(16,'CONTEXT')
    saved=torch.load(args.checkpoint,map_location='cpu')
    source.model.load_state_dict(saved['base'],strict=True)
    refiner.load_state_dict(saved['refiner'],strict=True)
    source.model.eval()
    hook=refiner.context_score.register_forward_hook(lambda m,a,o:torch.zeros_like(o))
    try: result=task.evaluate(cfg,source,refiner)
    finally: hook.remove();source.close()
    error=max(abs(a-b) for a,b in zip(EXPECTED,result['coco_eval_bbox']))
    assert error<.0002,error
    out=task.joint.task.engine.ROOT/'reports/PAPER_V1_20261006'
    out.mkdir(parents=True,exist_ok=True)
    task.joint.task.engine.write_json(out/'verification.json',dict(status='PASS',checkpoint_sha=SHA,max_abs_error=error,result=result))
    print({'status':'PASS','max_abs_error':error},flush=True)
