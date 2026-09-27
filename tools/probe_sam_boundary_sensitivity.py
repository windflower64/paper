"""Exploratory frozen-feature probe; not a detector or deployable module."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion, binary_dilation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from audit_sam_boundary_readability import ROOT, REPO, MASKS, CHECKPOINT, YAMLConfig

OUT=ROOT/'reports/115_sam_boundary_sensitivity'

def main():
    torch.set_num_threads(4)
    rng=np.random.default_rng(20260910)
    rows=json.loads((ROOT/'reports/114_sam_boundary_mechanism_pilot/samples.json').read_text(encoding='utf-8'))
    cfg=YAMLConfig(str(REPO/'experiments/phase_m/c_only_gq1_b8a4_20e_testdev_local.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    model=cfg.model.eval().requires_grad_(False)
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['ema']['module'],strict=True)
    model.cuda()
    capture={}
    modules={'S8':model.backbone.stages[1], 'S16_entry':model.backbone.stages[2].downsample}
    handles=[m.register_forward_hook(lambda m,a,o,k=k:capture.__setitem__(k,o.detach())) for k,m in modules.items()]
    modes=('bilinear','nearest','shift_half_cell')
    parts={(layer,mode):[] for layer in modules for mode in modes}
    labels=[]; held=[]; counts=[]
    for index,row in enumerate(rows):
        mask=np.asarray(Image.open(MASKS/'masks'/f"{row['image_id']:06d}.png"))>0
        pos=np.argwhere(mask & ~binary_erosion(mask,iterations=2))
        neg=np.argwhere(binary_dilation(mask,iterations=2) & ~mask)
        n=min(64,len(pos),len(neg))
        assert n>=8
        points=np.concatenate([pos[rng.choice(len(pos),n,False)],neg[rng.choice(len(neg),n,False)]])
        labels.extend([1]*n+[0]*n); held.extend([row['heldout']]*(2*n)); counts.append(2*n)
        im=np.asarray(Image.open(ROOT/'data/antiuav6k_common/images/train'/row['file_name']).convert('RGB')).copy()
        grid=torch.tensor(np.c_[(points[:,1]+.5)/640*2-1,(points[:,0]+.5)/512*2-1],dtype=torch.float32,device='cuda')[None,None]
        with torch.no_grad():
            model.backbone(torch.from_numpy(im).permute(2,0,1)[None].float().cuda()/255)
            for layer,value in capture.items():
                for mode in modes:
                    g=grid.clone()
                    if mode=='shift_half_cell':
                        g[...,0]+=1/value.shape[-1]; g[...,1]+=1/value.shape[-2]
                    feat=F.grid_sample(value,g,mode='nearest' if mode=='nearest' else 'bilinear',align_corners=False)
                    parts[layer,mode].append(feat[0,:,0].T.cpu().numpy())
        if index%40==0: print('EXTRACT',index,len(rows),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    y=np.asarray(labels); test=np.asarray(held); results=[]
    for (layer,mode),values in parts.items():
        x=np.concatenate(values).astype(np.float64)
        z=(x-x[~test].mean(0))/np.maximum(x[~test].std(0),1e-6)
        _,basis=np.linalg.eigh(z[~test].T@z[~test]/sum(~test))
        for dim in (8,16,32,64):
            p=z@basis[:,-dim:]; p/=np.maximum(p[~test].std(0),1e-6)
            probe=LogisticRegression(C=1,max_iter=1000,random_state=0).fit(p[~test],y[~test])
            scores=probe.predict_proba(p)[:,1]
            offset=0; per=[]
            for row,count in zip(rows,counts):
                if row['heldout']: per.append(float(roc_auc_score(y[offset:offset+count],scores[offset:offset+count])))
                offset+=count
            result=dict(layer=layer,mode=mode,dimension=dim,auc=float(roc_auc_score(y[test],scores[test])),image_macro_auc=float(np.mean(per)),per_image_auc=per)
            results.append(result)
            print(layer,mode,dim,result['auc'],flush=True)
    comparisons=[]
    for mode in modes:
        for dim in (8,16,32,64):
            a,b=[next(r for r in results if r['layer']==l and r['mode']==mode and r['dimension']==dim) for l in modules]
            delta=np.asarray(a['per_image_auc'])-np.asarray(b['per_image_auc'])
            comparisons.append(dict(mode=mode,dimension=dim,s8_minus_s16_auc=a['auc']-b['auc'],macro_difference=float(delta.mean()),positive_images=int((delta>0).sum())))
    for h in handles:h.remove()
    summary=dict(status='complete_exploratory',images=len(rows),heldout_images=sum(r['heldout'] for r in rows),results=results,comparisons=comparisons,
        caveats=['SAM pseudo-label evaluation, not true contour accuracy or detection AP','Same sequence split as pilot; new fixed boundary points shared by all conditions','Positive half-cell is sensitivity intervention, not claimed geometric correction','PCA trained on probe training partition only; nearest reads naturally tie adjacent pixels'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(comparisons,indent=2),flush=True)

if __name__=='__main__': main()
