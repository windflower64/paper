"""Privileged grouping capacity diagnostic; no detector updates or student."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from audit_sam_boundary_readability import ROOT,REPO,MASKS,CHECKPOINT,YAMLConfig

OUT=ROOT/'reports/116_sam_group_compression'

def encode(v,w):
    mean=v.mean(1)
    slots=[]
    for a in (w,1-w):
        denom=a.sum(1,keepdims=True)
        slots.append(np.where(denom>1e-6,(v*a[...,None]).sum(1)/np.maximum(denom,1e-6),mean))
    return np.concatenate(slots,axis=1)

def main(extra_encoders=None, output_dir=None):
    torch.set_num_threads(4)
    rng=np.random.default_rng(20260911)
    # Constant feature test: grouping must not invent structure from mask alone.
    v=np.ones((3,4,128),dtype=np.float32)
    assert np.allclose(encode(v,rng.random((3,4))),1)
    rows=json.loads((ROOT/'reports/114_sam_boundary_mechanism_pilot/samples.json').read_text(encoding='utf-8'))
    cfg=YAMLConfig(str(REPO/'experiments/phase_m/c_only_gq1_b8a4_20e_testdev_local.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    model=cfg.model.eval().requires_grad_(False)
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['ema']['module'],strict=True)
    model.cuda(); capture={}
    h=model.backbone.stages[1].register_forward_hook(lambda m,a,o:capture.__setitem__('x',o.detach()))
    q,_=np.linalg.qr(rng.normal(size=(256,128)))
    projection=torch.tensor(q,dtype=torch.float32,device='cuda')
    codes={k:[] for k in ('mean','fixed_vertical','box','sam','sam_shift8')}
    if extra_encoders:
        codes.update({k:[] for k in extra_encoders})
    targets=[]; held=[]; ids=[]
    for index,row in enumerate(rows):
        mask=np.asarray(Image.open(MASKS/'masks'/f"{row['image_id']:06d}.png"))>0
        yy,xx=np.mgrid[:512,:640]; x1,y1,x2,y2=row['bbox_xyxy_annotation']
        box=(xx+.5>=x1)&(xx+.5<x2)&(yy+.5>=y1)&(yy+.5<y2)
        shifted=np.zeros_like(mask); shifted[8:,8:]=mask[:-8,:-8]
        def blocks(a):
            t=F.avg_pool2d(torch.tensor(a.astype(np.float32))[None,None],8)
            return F.unfold(t,2,stride=2)[0].T.numpy()
        weights={k:blocks(a) for k,a in [('sam',mask),('box',box),('sam_shift8',shifted)]}
        boundary=(weights['sam'].max(1)>0)&(weights['sam'].min(1)<1)
        pick=np.flatnonzero(boundary)
        if len(pick)>64: pick=rng.choice(pick,64,False)
        if not len(pick):continue
        im=np.asarray(Image.open(ROOT/'data/antiuav6k_common/images/train'/row['file_name']).convert('RGB')).copy()
        with torch.no_grad():
            model.backbone(torch.from_numpy(im).permute(2,0,1)[None].float().cuda()/255)
            f=(capture['x'].permute(0,2,3,1)@projection).permute(0,3,1,2)
            v=F.unfold(f,2,stride=2)[0].T.reshape(-1,128,4).transpose(1,2).cpu().numpy()[pick]
        targets.append(v.reshape(len(pick),-1))
        for name in codes:
            if extra_encoders and name in extra_encoders:
                codes[name].append(extra_encoders[name](v,{k:a[pick] for k,a in weights.items()}))
                continue
            w=np.full((len(pick),4),.5) if name=='mean' else np.tile([1,0,1,0],(len(pick),1)) if name=='fixed_vertical' else weights[name][pick]
            codes[name].append(encode(v,w))
        held.extend([row['heldout']]*len(pick)); ids.extend([row['image_id']]*len(pick))
        if index%40==0: print('EXTRACT',index,len(rows),flush=True)
    y=np.concatenate(targets).astype(np.float64); test=np.asarray(held); ids=np.asarray(ids)
    assert set(ids[test]).isdisjoint(set(ids[~test]))
    mu=y[~test].mean(0); sd=np.maximum(y[~test].std(0),1e-6); y=(y-mu)/sd
    results={}
    for name,values in codes.items():
        x=np.concatenate(values).astype(np.float64)
        x=(x-x[~test].mean(0))/np.maximum(x[~test].std(0),1e-6)
        x=np.c_[x,np.ones(len(x))]
        # Same fixed ridge strength, fitted only on training sequences.
        regularizer=np.eye(x.shape[1]); regularizer[-1,-1]=0
        coef=np.linalg.solve(x[~test].T@x[~test]+len(x[~test])*.01*regularizer,x[~test].T@y[~test])
        error=((x[test]@coef-y[test])**2).mean(1)
        per={str(i):float(error[ids[test]==i].mean()) for i in np.unique(ids[test])}
        results[name]=dict(mse=float(error.mean()),image_macro_mse=float(np.mean(list(per.values()))),per_image_mse=per)
    destination=OUT if output_dir is None else output_dir
    h.remove(); destination.mkdir(parents=True,exist_ok=True)
    summary=dict(status='complete_privileged_capacity_diagnostic',constant_feature_check=True,cells=len(y),heldout_cells=int(test.sum()),heldout_images=len(np.unique(ids[test])),code_dimensions=256,target_dimensions=512,results=results,
       caveats=['Teacher masks/boxes used at encoding time: NOT deployable results','Decoder receives only encoded features, no mask or per-cell membership','Grouping can encode teacher location indirectly; this is NOT proof of true edge restoration','Mean baseline is rank-deficient despite equal output size; fixed spatial grouping is the stronger capacity control','Target is fixed random128 projection of S8, not original pixels or ground-truth edges','Boundary cells selected using same SAM mask for all conditions; evaluation is pseudo-mask-conditioned','No student and no detection AP evaluation'])
    if extra_encoders:
        summary['additional_encoder_names']=list(extra_encoders)
        summary['caveats'].append('Additional position encoders retain first64 projected channels at all four positions; comparison with two-slot encoders changes channel allocation')
    (destination/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='per_image_mse'} for k,v in results.items()},indent=2),flush=True)

if __name__=='__main__':main()
