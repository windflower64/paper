"""CPU target-only audit: occupancy independence versus aligned mask patches."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
PAIRS=((0,1),(0,2),(1,3),(2,3))

def patch_targets(mask):
    h,w=mask.shape
    z=mask.reshape(h//16,2,8,w//16,2,8).transpose(0,3,1,4,2,5).reshape(h//16,w//16,4,64).astype(np.float64)
    p=z.mean(-1)
    old=p[...,None]*p[...,None,:]+(1-p[...,None])*(1-p[...,None,:])
    same=1-np.abs(z[..., :,None,:]-z[...,None,:,:]).mean(-1)
    return old,same

def kl_uniform(affinity):
    t=affinity/affinity.sum(-1,keepdims=True)
    return (t*np.log(np.maximum(t,1e-12)*4)).sum(-1).mean(-1)

def main():
    # Equal occupancy cannot distinguish identical from opposite half-patches.
    a=np.zeros((16,16),dtype=bool);a[:4,:8]=1;a[:4,8:]=1
    b=a.copy();b[:8,8:]=~b[:8,8:]
    olda,newa=patch_targets(a);oldb,newb=patch_targets(b)
    assert np.array_equal(olda,oldb)
    assert newa[0,0,0,1]==1 and newb[0,0,0,1]==0
    assert np.all(np.diagonal(newa,axis1=-2,axis2=-1)==1)
    rows=json.loads((ROOT/'reports/114_sam_boundary_mechanism_pilot/samples.json').read_text(encoding='utf-8'))
    records=[]; all_old=[];all_new=[];all_pairs=[]
    yy,xx=np.mgrid[:512,:640]
    for row in rows:
        mask=np.asarray(Image.open(ROOT/f"reports/104_sam3_role_control/masks_train/masks/{row['image_id']:06d}.png"))>0
        x1,y1,x2,y2=row['bbox_xyxy_annotation'];cx=(x1+x2)/2;cy=(y1+y2)/2
        region=(np.abs(xx+.5-cx)<=x2-x1)&(np.abs(yy+.5-cy)<=y2-y1)
        support=region.reshape(32,16,40,16).mean((1,3))>.1
        old,new=patch_targets(mask)
        oldkl=kl_uniform(old)[support];newkl=kl_uniform(new)[support]
        cut=np.stack([1-new[...,i,j] for i,j in PAIRS],axis=-1)[support]
        assert np.isfinite(newkl).all() and ((cut>=0)&(cut<=1)).all()
        all_old.extend(oldkl);all_new.extend(newkl);all_pairs.extend(cut.flatten())
        records.append(dict(image_id=row['image_id'],heldout=row['heldout'],blocks=int(support.sum()),old_near_uniform=float((oldkl<.001).mean()),native_near_uniform=float((newkl<.001).mean()),nonzero_pair_fraction=float((cut>0).mean())))
    old=np.asarray(all_old);new=np.asarray(all_new);pairs=np.asarray(all_pairs)
    result=dict(status='target_audit_complete_no_training',images=len(rows),blocks=len(old),adjacent_pairs=len(pairs),synthetic_equal_occupancy_test=True,
        old_mean_kl=float(old.mean()),native_mean_kl=float(new.mean()),old_near_uniform_fraction=float((old<.001).mean()),native_near_uniform_fraction=float((new<.001).mean()),
        native_pair_nonzero_fraction=float((pairs>0).mean()),native_pair_above_0125_fraction=float((pairs>.125).mean()),mean_pair_disagreement=float(pairs.mean()),records=records,
        caveats=['Native target compares translated aligned 8x8 patches: shape disagreement, NOT exact boundary-crossing probability','All frames from detector training data; SAM masks are pseudo-labels','More nonuniform targets do not prove better supervision or detection gains','Counts weighted by blocks/pairs; no significance interpretation'])
    out=ROOT/'reports/120_sam_native_boundary_targets';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))

if __name__=='__main__':main()
