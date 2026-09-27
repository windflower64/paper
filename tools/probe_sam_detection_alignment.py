"""Exploratory S8 feature-gradient alignment; no parameter optimization."""
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from audit_sam_boundary_readability import ROOT,REPO,MASKS,CHECKPOINT,YAMLConfig
from src.zoo.dfine.sam_group_contrast import group_loss,raster

OUT=ROOT/'reports/118_sam_detection_alignment'

def relation_loss(features,targets,source):
    total=features.sum()*0
    for f,t in zip(features,targets):
        field=t['masks'].float().amax(0) if source=='sam' else raster(t['boxes'],512,640)
        p=F.interpolate(field[None,None],size=f.shape[-2:],mode='area')[0,0]
        local=F.interpolate(raster(t['boxes'],512,640,2)[None,None],size=f.shape[-2:],mode='area')[0,0]>.1
        z=F.normalize(f.float(),dim=0)
        losses=[]
        for axis in (0,1):
            a,b=(p[:-1],p[1:]) if axis==0 else (p[:,:-1],p[:,1:])
            za,zb=(z[:,:-1],z[:,1:]) if axis==0 else (z[:,:,:-1],z[:,:,1:])
            valid=(local[:-1]&local[1:]) if axis==0 else (local[:,:-1]&local[:,1:])
            target=2*(a*b+(1-a)*(1-b))-1
            error=((za*zb).sum(0)-target).square()
            if valid.any(): losses.append(error[valid].mean())
        if losses: total=total+sum(losses)/len(losses)
    return total/len(targets)

def main():
    torch.set_num_threads(4); torch.manual_seed(20260912); random.seed(20260912)
    rows=json.loads((ROOT/'reports/114_sam_boundary_mechanism_pilot/samples.json').read_text(encoding='utf-8'))
    unique={r['sequence']:r for r in rows if not r['heldout']}
    selected=random.sample(list(unique.values()),64)
    cfg=YAMLConfig(str(REPO/'experiments/phase_m/c_only_gq1_b8a4_20e_testdev_local.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    model=cfg.model
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['ema']['module'],strict=True)
    model.cuda().train().requires_grad_(False)
    for m in model.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
    model.decoder.num_denoising=0
    criterion=cfg.criterion.cuda().train()
    cache={}
    def hook(m,a,o):
        leaf=o.detach().requires_grad_(True); cache['s8']=leaf; return leaf
    handle=model.backbone.stages[1].register_forward_hook(hook)
    records=[]; loss_keys=None
    for start in range(0,64,8):
        images=[]; targets=[]
        for row in selected[start:start+8]:
            im=np.asarray(Image.open(ROOT/'data/antiuav6k_common/images/train'/row['file_name']).convert('RGB')).copy()
            images.append(torch.from_numpy(im).permute(2,0,1).float()/255)
            mask=np.asarray(Image.open(MASKS/'masks'/f"{row['image_id']:06d}.png"))>0
            x1,y1,x2,y2=row['bbox_xyxy_annotation']
            targets.append(dict(boxes=torch.tensor([[(x1+x2)/1280,(y1+y2)/1024,(x2-x1)/640,(y2-y1)/512]],device='cuda'),
                labels=torch.zeros(1,dtype=torch.long,device='cuda'),masks=torch.tensor(mask.copy(),device='cuda')[None],
                sam_quality=torch.tensor(row['supervision_weight'],device='cuda')))
        outputs=model(torch.stack(images).cuda(),targets)
        losses=criterion(outputs,targets)
        loss_keys=list(losses)
        assert all(k in losses for k in ('loss_vfl','loss_bbox','loss_giou'))
        f=cache['s8']
        objectives={'classification':losses['loss_vfl'],'localization':losses['loss_bbox']+losses['loss_giou'],
                    'group_sam':group_loss(f,targets,'sam'),'group_box':group_loss(f,targets,'box'),
                    'relation_sam':relation_loss(f,targets,'sam'),'relation_box':relation_loss(f,targets,'box')}
        gradients={k:torch.autograd.grad(v,f,retain_graph=True)[0].detach().flatten(1) for k,v in objectives.items()}
        for i,row in enumerate(selected[start:start+8]):
            r=dict(image_id=row['image_id'],sequence=row['sequence'],measurements={})
            for aux in ('group_sam','group_box','relation_sam','relation_box'):
                a=gradients[aux][i]; an=a.norm()
                result={'gradient_norm':float(an)}
                for task in ('classification','localization'):
                    b=gradients[task][i]; bn=b.norm()
                    result[task+'_cosine']=float((a@b)/(an*bn).clamp_min(1e-20)) if an>0 and bn>0 else None
                    result[task+'_norm_ratio']=float(an/bn.clamp_min(1e-20))
                r['measurements'][aux]=result
            records.append(r)
        print('BATCH',start//8,{k:float(v.detach()) for k,v in objectives.items()},flush=True)
        del outputs,losses,objectives,gradients,f
    handle.remove()
    summary={}
    for aux in ('group_sam','group_box','relation_sam','relation_box'):
        summary[aux]={}
        for task in ('classification','localization'):
            values=[r['measurements'][aux][task+'_cosine'] for r in records if r['measurements'][aux][task+'_cosine'] is not None]
            summary[aux][task]=dict(valid=len(values),mean_cosine=float(np.mean(values)),median_cosine=float(np.median(values)),positive=sum(v>0 for v in values))
    assert all(p.grad is None for p in model.parameters())
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(status='complete_no_parameter_updates',images=64,sequences=64,batch=8,checkpoint=str(CHECKPOINT),summary=summary,records=records,criterion_keys=loss_keys,
        caveats=['S8 feature gradients, not parameter-gradient conflict or causal training outcome','Frozen model parameters/BN; decoder training outputs, no denoising, no augmentation, FP32','Only final loss_vfl and loss_bbox+loss_giou; excludes auxiliary heads/FGL/DDF from task gradient','Aux losses are unweighted; norms do not select final training weights','Relation objective is exploratory cosine matching, not a validated trainable module','Positive gradient cosine is only local first-order agreement, not proof of detection improvement','SAM pseudo-labels, all images from detector training set; earlier probe-heldout sequences excluded'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
