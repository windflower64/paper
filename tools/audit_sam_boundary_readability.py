"""Pilot: pseudo-mask geometry and sequence-held-out equal-capacity probes.

No detector updates. A probe measures decodability, not information-theoretic
loss or human-verified contour accuracy. All source frames are training data.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0,str(REPO))
from src.core import YAMLConfig

OUT = ROOT/'reports/114_sam_boundary_mechanism_pilot'
MASKS = ROOT/'reports/104_sam3_role_control/masks_train'
CHECKPOINT = ROOT/'outputs/C_ONLY_GQ1_B8A4_20E_TESTDEV/seed0/best_stg1.pth'


def sequence(row):
    return Path(row['file_name']).stem.rsplit('_',1)[0]


def boundary(mask):
    return mask & ~binary_erosion(mask)


def boundary_f1(a,b,tolerance=2):
    x,y=boundary(a),boundary(b)
    if not x.any() or not y.any():
        return 0.
    precision=float((distance_transform_edt(~y)[x]<=tolerance).mean())
    recall=float((distance_transform_edt(~x)[y]<=tolerance).mean())
    return 2*precision*recall/max(precision+recall,1e-12)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    rng=np.random.default_rng(20260909)
    OUT.mkdir(parents=True,exist_ok=True)
    records=json.loads((MASKS/'records.json').read_text(encoding='utf-8'))
    accepted=[r for r in records if r['accepted']]
    groups=sorted({sequence(r) for r in accepted})
    rng.shuffle(groups)
    heldout=set(groups[:max(1,len(groups)//4)])
    # At most two frames per sequence; no selection on probe performance.
    selected=[]
    for group in groups:
        candidates=[r for r in accepted if sequence(r)==group]
        selected.extend(candidates[i] for i in rng.choice(len(candidates),min(2,len(candidates)),replace=False))
    capture={}
    cfg=YAMLConfig(str(REPO/'experiments/phase_m/c_only_gq1_b8a4_20e_testdev_local.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    model=cfg.model.eval().requires_grad_(False)
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    model.load_state_dict(state['ema']['module'],strict=True)
    model.cuda(); del state
    modules={'S4':model.backbone.stages[0], 'S8_entry':model.backbone.stages[1].downsample,
             'S8':model.backbone.stages[1], 'S16_entry':model.backbone.stages[2].downsample,
             'S16':model.backbone.stages[2]}
    handles=[m.register_forward_hook(lambda m,a,o,k=k:capture.__setitem__(k,o.detach())) for k,m in modules.items()]
    features={k:[] for k in modules}
    labels=[]; splits=[]; metadata=[]; geometry=[]; skipped=[]
    for index,row in enumerate(selected):
        mask=np.asarray(Image.open(MASKS/'masks'/f"{row['image_id']:06d}.png"))>0
        if mask.shape!=(512,640):
            raise ValueError(f'Unexpected mask geometry: {mask.shape}')
        inside=mask & ~binary_erosion(mask,iterations=2)
        outside=binary_dilation(mask,iterations=2) & ~mask
        pos,neg=np.argwhere(inside),np.argwhere(outside)
        if min(len(pos),len(neg))<8:
            skipped.append(row['image_id']); continue
        count=min(64,len(pos),len(neg))
        points=np.concatenate([pos[rng.choice(len(pos),count,False)],neg[rng.choice(len(neg),count,False)]])
        y=np.r_[np.ones(count),np.zeros(count)]
        x1,y1,x2,y2=row['bbox_xyxy_annotation']
        edge=np.sqrt(max(0,(x2-x1)*(y2-y1)))
        bin_name='lt16' if edge<16 else '16to32' if edge<32 else 'ge32'
        mask_tensor=torch.from_numpy(mask.astype(np.float32))[None,None]
        geom=dict(image_id=row['image_id'],size_bin=bin_name,foreground_pixels=int(mask.sum()))
        for stride in (4,8,16):
            coarse=F.avg_pool2d(mask_tensor,stride)
            reconstructed=F.interpolate(coarse,scale_factor=stride,mode='bilinear',align_corners=False)[0,0].numpy()>=.5
            geom[f'S{stride}']=dict(iou=float((mask&reconstructed).sum()/max((mask|reconstructed).sum(),1)),
                                      boundary_f1=boundary_f1(reconstructed,mask),
                                      mixed_cells=int(((coarse>0)&(coarse<1)).sum()),
                                      vanished=not bool(reconstructed.any()))
        geometry.append(geom)
        image=np.asarray(Image.open(ROOT/'data/antiuav6k_common/images/train'/row['file_name']).convert('RGB')).copy()
        assert image.shape[:2]==mask.shape
        tensor=torch.from_numpy(image).permute(2,0,1)[None].float().cuda()/255
        grid=torch.tensor(np.c_[(points[:,1]+.5)/640*2-1,(points[:,0]+.5)/512*2-1],dtype=torch.float32,device='cuda')[None,None]
        with torch.no_grad():
            model.backbone(tensor)
            for name,value in capture.items():
                features[name].append(F.grid_sample(value,grid,align_corners=False)[0,:,0].T.cpu().numpy())
        labels.append(y); splits.append(np.full(2*count,sequence(row) in heldout))
        metadata.append(dict(**row,sequence=sequence(row),heldout=sequence(row) in heldout,size_bin=bin_name,points=count*2))
        if index%40==0:
            print('EXTRACT',index,len(selected),flush=True)
    labels=np.concatenate(labels); test=np.concatenate(splits)
    results={}
    for name,parts in features.items():
        x=np.concatenate(parts).astype(np.float64)
        mean=x[~test].mean(0); std=np.maximum(x[~test].std(0),1e-6)
        z=(x-mean)/std
        # Train-only PCA; each probe has 16 coefficients plus intercept.
        _,v=np.linalg.eigh(z[~test].T@z[~test]/int((~test).sum()))
        basis=v[:,-16:]; z=z@basis
        scale=np.maximum(z[~test].std(0),1e-6); z=z/scale
        probe=LogisticRegression(C=1.,max_iter=1000,random_state=0)
        probe.fit(z[~test],labels[~test])
        score=probe.predict_proba(z)[:,1]
        sizes=np.concatenate([np.full(r['points'],r['size_bin']) for r in metadata])
        result=dict(channels=x.shape[1],probe_parameters=17,auc=float(roc_auc_score(labels[test],score[test])),
                    balanced_accuracy=float(balanced_accuracy_score(labels[test],score[test]>=.5)))
        result['size_bins']={}
        for size in ('lt16','16to32','ge32'):
            pick=test&(sizes==size)
            result['size_bins'][size]=dict(points=int(pick.sum()),auc=float(roc_auc_score(labels[pick],score[pick])) if pick.any() else None)
        results[name]=result
        np.savez_compressed(OUT/f'probe_{name}.npz',scores=score,labels=labels,heldout=test,
                            mean=mean,std=std,basis=basis,scale=scale,coef=probe.coef_,intercept=probe.intercept_)
    for h in handles:h.remove()
    summary=dict(status='pilot_complete_not_human_boundary_validation',seed=20260909,
        checkpoint=str(CHECKPOINT),checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        records=len(records),accepted=len(accepted),images=len(metadata),skipped=skipped,
        sequence_count=len(groups),heldout_sequences=len(heldout),
        heldout_images=sum(r['heldout'] for r in metadata),probes=results,
        geometry={f'S{s}':dict(mean_iou=float(np.mean([g[f'S{s}']['iou'] for g in geometry])),
                               mean_boundary_f1=float(np.mean([g[f'S{s}']['boundary_f1'] for g in geometry])),
                               vanished=sum(g[f'S{s}']['vanished'] for g in geometry)) for s in (4,8,16)},
        caveats=['SAM pseudo-labels, not human ground truth',
                 'Probe-held-out sequences were seen by detector training; not independent detector test',
                 'PCA16 equalizes readout capacity but can discard different information across stages',
                 'Mask resampling is a geometric diagnostic, not the actual learned convolution',
                 'Cross-stage scores alone do not establish causal or irreversible information loss'])
    for name,data in [('summary',summary),('samples',metadata),('geometry',geometry)]:
        (OUT/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
