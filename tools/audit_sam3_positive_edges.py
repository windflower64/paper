"""Train-only positive-box SAM3 contour stability audit, not pixel accuracy."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/103_sam3_positive_edges'
sys.path.insert(0,str(ROOT/'D-FINE'))
from tools.sam3_teacher_pilot import dump, iou


def mask_iou(a,b):
    return float((a & b).sum()/max(1,(a | b).sum()))


def bounds(mask):
    yy,xx=np.where(mask)
    return [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else [0,0,0,0]


def edge_band(mask,radius=2):
    from scipy.ndimage import binary_dilation,binary_erosion
    return binary_dilation(mask,iterations=radius)^binary_erosion(mask,iterations=radius)


def select_mask(pred,box):
    masks=pred['masks'].cpu().numpy()[:,0]
    if not len(masks): return None
    # Known-target identity association only, not best pixel mask selection.
    pb=pred['boxes'].float().cpu().tolist()
    index=max(range(len(pb)),key=lambda k:iou(pb[k],box))
    return masks[index]


def run(full=False):
    import torch
    from PIL import Image
    sys.path.insert(0,str(ROOT/'_third_party/sam3'))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.set_num_threads(4)
    if full:
        from collections import defaultdict
        data=json.loads((ROOT/'data/antiuav6k_common/annotations/instances_visible_common_train.json').read_text(encoding='utf-8'))
        aa=defaultdict(list)
        for ann in data['annotations']:
            if not ann.get('iscrowd',0): aa[ann['image_id']].append(ann)
        positives=[]
        for im in data['images']:
            if aa[im['id']]:
                area=max(a['bbox'][2]*a['bbox'][3] for a in aa[im['id']])
                positives.append(dict(**im,annotations=aa[im['id']],group='small' if area<1024 else 'medium' if area<9216 else 'large'))
    else:
        manifest=json.loads((ROOT/'reports/100_sam3_teacher/manifest.json').read_text(encoding='utf-8'))
        assert manifest['split']=='train'
        positives=[im for im in manifest['images'] if im['annotations']]
    dump(OUT/'manifest.json',dict(split='train',images=positives,source='full train positives' if full else 'fixed pilot positives',
                                note='GT boxes allowed ONLY to construct training teachers; no validation labels used'))
    model=build_sam3_image_model(checkpoint_path=str(ROOT/'weights/sam3/sam3.pt'),load_from_HF=False,compile=False)
    processor=Sam3Processor(model,confidence_threshold=.2)
    dump(OUT/'protocol.json',dict(prompt='drone',box_source='train GT',jitters='center translation (+dx,+dy) and (-dx,-dy), dx=max(1,0.05*w), dy=max(1,0.05*h)',
                                 edge_band_radius_pixels=2,resize='official 1008',selection='maximum predicted-box IoU to nominal GT to associate object',
                                 pixel_ground_truth=False,torch=torch.__version__))
    for index,im in enumerate(positives):
        path=OUT/'masks'/f"{im['id']:06d}.npz"
        if path.exists(): continue
        assert len(im['annotations'])==1
        x,y,w,h=im['annotations'][0]['bbox']; box=[x,y,x+w,y+h]
        masks=[]; available=[]
        with Image.open(ROOT/'data/antiuav6k_common/images/train'/im['file_name']) as source:
            image=source.convert('RGB')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=processor.set_image(image)
            for sign in [0,1,-1]:
                dx=sign*max(1,.05*w);dy=sign*max(1,.05*h)
                x0,x1=np.clip([x+dx,x+w+dx],0,im['width'])
                y0,y1=np.clip([y+dy,y+h+dy],0,im['height'])
                prompt=[float((x0+x1)/2/im['width']),float((y0+y1)/2/im['height']),float((x1-x0)/im['width']),float((y1-y0)/im['height'])]
                processor.reset_all_prompts(state)
                processor.set_text_prompt(prompt='drone',state=state)
                pred=processor.add_geometric_prompt(box=prompt,label=True,state=state)
                mask=select_mask(pred,box)
                available.append(mask is not None)
                masks.append(mask if mask is not None else np.zeros((im['height'],im['width']),dtype=bool))
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,masks=np.stack(masks),available=available)
        print('EDGE_IMAGE',index+1,len(positives),im['id'],available,flush=True)
    summary()


def summary():
    from PIL import Image
    import torch
    from torch.nn.functional import interpolate
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    oldroot=ROOT/'reports/20_spatial_importance/S_TNDP2_RETENTION_DISTILLATION/masks_train'
    oldrecords={r['image_id']:r for r in json.loads((oldroot/'records.json').read_text(encoding='utf-8'))}
    rows=[]
    for im in manifest['images']:
        with np.load(OUT/'masks'/f"{im['id']:06d}.npz") as archive:
            mm=archive['masks']; available=archive['available']
        x,y,w,h=im['annotations'][0]['bbox']; gt=[x,y,x+w,y+h]
        row=dict(image_id=im['id'],group=im['group'],available=available.tolist(),mask_pixels=int(mm[0].sum()),
                 bbox_iou=iou(bounds(mm[0]),gt),mask_jitter_iou=min(mask_iou(mm[0],m) for m in mm[1:]),
                 edge_jitter_iou=min(mask_iou(edge_band(mm[0]),edge_band(m)) for m in mm[1:]),
                 sam2_accepted=bool(oldrecords[im['id']]['accepted']))
        oldpath=oldroot/'masks'/f"{im['id']:06d}.png"
        if oldpath.exists() and row['sam2_accepted']:
            assert np.allclose(oldrecords[im['id']]['bbox_xyxy_annotation'],gt,atol=1e-3), 'SAM2 GT mismatch'
            old=np.asarray(Image.open(oldpath).convert('L'))>0
            assert old.shape==mm[0].shape
            row['sam2_sam3_mask_agreement']=mask_iou(old,mm[0])
            row['sam2_sam3_edge_agreement']=mask_iou(edge_band(old),edge_band(mm[0]))
        band=edge_band(mm[0]).astype(np.float32)
        for stride in [4,8,16]:
            low=interpolate(torch.from_numpy(band)[None,None],size=(im['height']//stride,im['width']//stride),mode='area')[0,0]
            row[f'edge_s{stride}_soft_area']=float(low.sum())
            row[f'edge_s{stride}_half_cells']=int((low>=.5).sum())
        rows.append(row)
    stats={}
    for group in sorted(set(r['group'] for r in rows)):
        subset=[r for r in rows if r['group']==group]
        keys=['bbox_iou','mask_pixels','mask_jitter_iou','edge_jitter_iou','sam2_sam3_mask_agreement','sam2_sam3_edge_agreement']
        detail={k:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p10=float(np.quantile(v,.1))) for k in keys if (v:=[r[k] for r in subset if k in r])}
        detail['images']=len(subset)
        detail['all_prompts_available']=sum(all(r['available']) for r in subset)
        detail['zero_half_edge_cells']={str(s):sum(r[f'edge_s{s}_half_cells']==0 for r in subset) for s in [4,8,16]}
        # Predefined screening heuristic, explicitly not a pixel-quality label.
        detail['heuristic_eligible']=sum(all(r['available']) and r['bbox_iou']>=.5 and r['mask_jitter_iou']>=.7 for r in subset)
        stats[group]=detail
    dump(OUT/'rows.json',rows)
    dump(OUT/'summary.json',dict(complete=len(rows)==len(manifest['images']),stats=stats,note='Stability/agreement only; no pixel GT, no student detection gain measured'))
    print(json.dumps(stats,indent=2),flush=True)


def figures():
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    selected=[im for group in ['small','medium'] for im in [x for x in manifest['images'] if x['group']==group][:2]]
    fig,axes=plt.subplots(4,4,figsize=(12,12))
    oldroot=ROOT/'reports/20_spatial_importance/S_TNDP2_RETENTION_DISTILLATION/masks_train/masks'
    for row,im in enumerate(selected):
        source=np.asarray(Image.open(ROOT/'data/antiuav6k_common/images/train'/im['file_name']).convert('RGB'))
        with np.load(OUT/'masks'/f"{im['id']:06d}.npz") as archive: masks=archive['masks']
        old=np.asarray(Image.open(oldroot/f"{im['id']:06d}.png").convert('L'))>0
        x,y,w,h=im['annotations'][0]['bbox']; half=max(40,w,h);cx=x+w/2;cy=y+h/2
        for col in range(4):
            ax=axes[row,col];ax.imshow(source)
            overlay=np.zeros((*source.shape[:2],4))
            if col in [1,2]: overlay[old if col==1 else masks[0]]=[1,0,1,.4]
            if col==3:
                edges=np.stack([edge_band(m) for m in masks],-1)
                overlay[:,:,:3]=edges.astype(float);overlay[:,:,3]=edges.any(-1)*.85
            ax.imshow(overlay)
            ax.set_xlim(max(0,cx-half),min(im['width'],cx+half));ax.set_ylim(min(im['height'],cy+half),max(0,cy-half))
            ax.set_title([f"{im['id']} {im['group']} RGB",'SAM2 cached mask','SAM3 GT-prompt mask','Edges: nominal / +shift / -shift'][col],fontsize=8)
            ax.axis('off')
    fig.suptitle('Fixed train samples; crop is for inspection only, not teacher input',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/'fixed4_edge_review.png',dpi=150);plt.close(fig)
    dump(OUT/'fixed4_edge_review_ids.json',[im['id'] for im in selected])
    print('EDGE FIGURE SAVED')


if __name__=='__main__':
    full='--full' in sys.argv
    if full: OUT=ROOT/'reports/104_sam3_role_control/teacher_raw'
    if sys.argv[1]=='run': run(full)
    elif sys.argv[1]=='summary': summary()
    elif sys.argv[1]=='figures': figures()
    else:
        mask=np.zeros((16,16),bool);mask[4:12,4:12]=True
        assert mask_iou(mask,mask)==1 and bounds(mask)==[4,4,12,12]
        assert edge_band(mask).sum()>0
        print('EDGE TEST PASS')
