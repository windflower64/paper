"""Read-only result checks and reproducible pseudo-contour review sheets."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/114_sam_boundary_mechanism_pilot'

def main():
    samples=json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
    train={r['sequence'] for r in samples if not r['heldout']}
    test={r['sequence'] for r in samples if r['heldout']}
    assert train.isdisjoint(test)
    assert len({r['image_id'] for r in samples})==len(samples)
    metrics={}
    arrays={}
    for layer in ('S4','S8_entry','S8','S16_entry','S16'):
        a=np.load(OUT/f'probe_{layer}.npz')
        assert len(a['scores'])==sum(r['points'] for r in samples)
        assert np.isfinite(a['scores']).all()
        per_image=[]; offset=0
        for row in samples:
            end=offset+row['points']
            assert np.all(a['heldout'][offset:end]==row['heldout'])
            if row['heldout']:
                per_image.append(roc_auc_score(a['labels'][offset:end],a['scores'][offset:end]))
            offset=end
        arrays[layer]=np.asarray(per_image)
        metrics[layer]={'image_macro_auc':float(np.mean(per_image))}
    delta=arrays['S8']-arrays['S16_entry']
    result={'checks_passed':True,'heldout_images':len(delta),'metrics':metrics,
            's8_minus_s16_entry':{'mean_image_auc_difference':float(delta.mean()),
                                'images_positive':int((delta>0).sum()),
                                'images_negative':int((delta<0).sum())},
            'caveat':'Image comparisons within sequences are correlated; no significance claim.'}
    (OUT/'review_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    rng=np.random.default_rng(20260909)
    rows=[r for r in samples if r['heldout']]
    chosen=[rows[i] for i in rng.choice(len(rows),24,replace=False)]
    (OUT/'contour_review_pending.json').write_text(json.dumps([
        {'image_id':r['image_id'],'file_name':r['file_name'],'human_review':'pending'} for r in chosen
    ],indent=2),encoding='utf-8')
    for page in range(4):
        fig,axes=plt.subplots(6,3,figsize=(12,17))
        for i,row in enumerate(chosen[page*6:page*6+6]):
            im=np.asarray(Image.open(ROOT/'data/antiuav6k_common/images/train'/row['file_name']).convert('RGB'))
            mask=np.asarray(Image.open(ROOT/f"reports/104_sam3_role_control/masks_train/masks/{row['image_id']:06d}.png"))>0
            x1,y1,x2,y2=row['bbox_xyxy_annotation']
            side=max(96,2*max(x2-x1,y2-y1)); cx=(x1+x2)/2; cy=(y1+y2)/2
            left=max(0,int(cx-side/2)); right=min(640,int(cx+side/2))
            top=max(0,int(cy-side/2)); bottom=min(512,int(cy+side/2))
            axes[i,0].imshow(im)
            axes[i,0].add_patch(Rectangle((left,top),right-left,bottom-top,fill=False,edgecolor='yellow',linewidth=1))
            axes[i,0].set_title(f"Image {row['image_id']} | full RGB",fontsize=10)
            crop=im[top:bottom,left:right]
            for j in (1,2): axes[i,j].imshow(crop,interpolation='nearest')
            axes[i,1].set_title('RGB crop (no overlay)',fontsize=10)
            axes[i,2].contour(mask[top:bottom,left:right],levels=[.5],colors=['lime'],linewidths=1)
            axes[i,2].add_patch(Rectangle((x1-left,y1-top),x2-x1,y2-y1,fill=False,edgecolor='cyan',linewidth=1))
            axes[i,2].set_title('Green: SAM | cyan: annotation box',fontsize=10)
            for ax in axes[i]: ax.axis('off')
        fig.suptitle('Random held-out probe frames / SAM pseudo-contours, NOT pixel ground truth',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.975))
        fig.savefig(OUT/f'contour_review_{page+1}.jpg',dpi=130)
        plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
