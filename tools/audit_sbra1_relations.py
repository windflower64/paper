"""Measure actual teacher entropy and student KL on fixed probe frames."""
import json
import math
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from audit_sam_boundary_readability import ROOT,REPO,MASKS,YAMLConfig
from src.zoo.dfine.sam_group_contrast import raster

def main():
    torch.set_num_threads(4)
    rows=[r for r in json.loads((ROOT/'reports/114_sam_boundary_mechanism_pilot/samples.json').read_text(encoding='utf-8')) if r['heldout']]
    results={}
    for arm in ('sam','none','box'):
        cfg=YAMLConfig(str(REPO/f'experiments/phase_s/s_sbra1_{arm}_b8a4_20e.yml'))
        cfg.yaml_cfg['HGNetv2']['pretrained']=False
        model=cfg.model.eval().requires_grad_(False)
        state=torch.load(ROOT/f'outputs/S_BRA1_{arm.upper()}_B8A4_20E_TESTDEV/seed0/best_stg1.pth',map_location='cpu',weights_only=False)
        model.load_state_dict(state['ema']['module'],strict=True);del state
        model.cuda();cache={}; module=model.backbone.stages[2].sbra
        hook=module.relation.register_forward_hook(lambda m,a,o:cache.__setitem__('logits',o.detach()))
        measurements={source:[] for source in ('sam','box')}
        with torch.no_grad():
            for row in rows:
                image=np.asarray(Image.open(ROOT/'data/antiuav6k_common/images/train'/row['file_name']).convert('RGB')).copy()
                model.backbone(torch.from_numpy(image).permute(2,0,1)[None].float().cuda()/255)
                logits=cache['logits'][0].permute(1,2,0).reshape(32,40,4,4).float()
                logpred=logits.log_softmax(-1);pred=logpred.exp()
                mask=torch.tensor((np.asarray(Image.open(MASKS/'masks'/f"{row['image_id']:06d}.png"))>0).copy(),device='cuda').float()
                x1,y1,x2,y2=row['bbox_xyxy_annotation']
                boxes=torch.tensor([[(x1+x2)/1280,(y1+y2)/1024,(x2-x1)/640,(y2-y1)/512]],device='cuda')
                support=F.interpolate(raster(boxes,512,640,2)[None,None],size=(32,40),mode='area')[0,0]>.1
                for source in measurements:
                    field=mask if source=='sam' else raster(boxes,512,640)
                    p=F.pixel_unshuffle(F.interpolate(field[None,None],size=(64,80),mode='area'),2)[0].permute(1,2,0)
                    t=p[...,None]*p[...,None,:]+(1-p[...,None])*(1-p[...,None,:])
                    t=t/t.sum(-1,keepdim=True).clamp_min(1e-8)
                    logt=t.clamp_min(1e-12).log()
                    entropy=-(t*logt).sum(-1).mean(-1)
                    kl=(t*(logt-logpred)).sum(-1).mean(-1)
                    predentropy=-(pred*logpred).sum(-1).mean(-1)
                    for a,b,c in zip(entropy[support].cpu().tolist(),kl[support].cpu().tolist(),predentropy[support].cpu().tolist()):
                        measurements[source].append((a,math.log(4)-a,b,c))
        results[arm]={}
        for source,values in measurements.items():
            a=np.asarray(values); info=a[:,1]>.01
            results[arm][source]=dict(blocks=len(a),teacher_entropy=float(a[:,0].mean()),uniform_kl=float(a[:,1].mean()),student_kl=float(a[:,2].mean()),student_entropy=float(a[:,3].mean()),
                near_uniform_fraction=float((a[:,1]<.001).mean()),informative_blocks=int(info.sum()),informative_uniform_kl=float(a[info,1].mean()),informative_student_kl=float(a[info,2].mean()))
        hook.remove();del model,module,cache;torch.cuda.empty_cache()
        print(arm,results[arm],flush=True)
    out=ROOT/'reports/119_sbra1/verification';out.mkdir(parents=True,exist_ok=True)
    (out/'relations.json').write_text(json.dumps(dict(images=len(rows),results=results,caveats=['Frames are probe-heldout but detector-training data, not independent test','No augmentation; this does not reproduce training-epoch average loss','Block weighted metrics; correlated blocks, no significance claim','Informative means teacher KL to uniform >0.01 nats; near uniform <0.001 nats']),indent=2),encoding='utf-8')

if __name__=='__main__':main()
