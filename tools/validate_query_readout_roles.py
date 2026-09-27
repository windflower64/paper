"""Fixed SAM checkpoint: remove one readout role without renormalizing others."""
import json
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
SNAP=ROOT/'reports/137_query_readout_trial/frozen_repo'
sys.path.insert(0,str(SNAP))
from src.core import YAMLConfig
from src.solver import TASKS
from src.solver.det_engine import evaluate
from tools.query_readout_runtime import install
from tools.sam_query_readout import QueryReadout


class RoleProbe(QueryReadout):
    def forward(self,feature,query,references):
        boxes=references[:,:,0,:].detach();self.references=boxes
        offsets=self.offsets(query).reshape(*query.shape[:2],12,2).tanh()
        self.locations=boxes[:,:,None,:2]+offsets*boxes[:,:,None,2:]
        projected=self.project(feature)
        sampled=F.grid_sample(projected,self.locations*2-1,mode='bilinear',padding_mode='zeros',align_corners=False)
        weights=self.weights(query).softmax(-1)
        if self.disabled_role is not None:
            mask=torch.ones_like(weights);mask[:,:,self.disabled_role*4:self.disabled_role*4+4]=0
            weights=weights*mask
        return self.output((sampled.permute(0,2,3,1)*weights[...,None]).sum(2))


def main():
    out=ROOT/'reports/139_query_readout_roles/attempt02';out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    cfg=YAMLConfig(str(SNAP/'experiments/phase_s/sgc_scale_none_b16a2_20e.yml'),output_dir=str(out/'runtime'))
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    model=install(cfg.model)
    checkpoint=ROOT/'outputs/QUERY_READ_SAM_B16A2_10E_TESTDEV/seed0/best_stg1.pth'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['ema']['module'],strict=True)
    solver=TASKS[cfg.yaml_cfg['task']](cfg);solver.eval()
    solver.model.eval()
    reader=solver.model.decoder.decoder.layers[0].cross_attn.reader
    # Check equivalent implementation before intervening; same inputs/weights.
    samples,_=next(iter(solver.val_dataloader));samples=samples[:2].to(solver.device)
    with torch.no_grad():reference=solver.model(samples)
    reader.__class__=RoleProbe;reader.disabled_role=None
    with torch.no_grad():control=solver.model(samples)
    assert all(torch.equal(reference[k],control[k]) for k in ('pred_boxes','pred_logits'))
    results={}
    for role,name in enumerate(('interior','boundary','context')):
        reader.disabled_role=role
        stats,_=evaluate(solver.model,solver.criterion,solver.postprocessor,solver.val_dataloader,
            solver.evaluator,solver.device,epoch=-1,use_wandb=False)
        results[name]=stats['coco_eval_bbox']
        (out/f'without_{name}.json').write_text(json.dumps({'metrics':results[name]},indent=2),encoding='utf-8')
        print('ROLE_RESULT',name,results[name],flush=True)
    (out/'summary.json').write_text(json.dumps({'status':'complete','results':results,
        'scope':'SAM best EMA, 1820 images, one role zeroed at a time; no softmax renormalization, no training; role labels do not guarantee points learned semantic roles'},indent=2),encoding='utf-8')

if __name__=='__main__':main()
