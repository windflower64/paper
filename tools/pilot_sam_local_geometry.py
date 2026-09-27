"""Sequence-held-out training-side pilot for SAM local box-change evidence."""
import json
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'reports/148_sgc2_half_box/frozen_repo'))
from src.core import YAMLConfig

OUT = ROOT/'reports/150_sam_local_geometry'


def iou(box, gt):
    inter = np.maximum(0,np.minimum(box[2:],gt[2:])-np.maximum(box[:2],gt[:2])).prod()
    return float(inter/max(1e-8,(box[2:]-box[:2]).prod()+(gt[2:]-gt[:2]).prod()-inter))


def stats(field, box):
    h,w=field.shape
    x0,y0,x1,y1=np.round(box).astype(int)
    x0,x1=np.clip([x0,x1],0,w); y0,y1=np.clip([y0,y1],0,h)
    patch=field[y0:y1,x0:x1]
    if patch.size==0:
        return [0.]*7
    thickness=max(1,round(min(x1-x0,y1-y0)*.15))
    return [float(patch.mean()),float(patch.std()),float(patch[:thickness].mean()),
            float(patch[-thickness:].mean()),float(patch[:,:thickness].mean()),
            float(patch[:,-thickness:].mean()),float(patch.sum()/max(field.sum(),1e-8))]


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--per-scale',type=int,default=100)
    parser.add_argument('--folds',type=int,default=1)
    parser.add_argument('--nonlinear',action='store_true')
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.output
    torch.set_num_threads(4); torch.manual_seed(0)
    rng=np.random.default_rng(20260916)
    OUT.mkdir(parents=True,exist_ok=True)
    ann=json.loads((ROOT/'data/antiuav6k_common/annotations/instances_visible_common_train.json').read_text())
    images={x['id']:x for x in ann['images']}
    annotations={x['image_id']:x for x in ann['annotations']}
    records={x['image_id']:x for x in json.loads((ROOT/'reports/104_sam3_role_control/masks_train/records.json').read_text())}
    selected=[]
    for small in (True,False):
        eligible=[i for i,a in annotations.items() if records[i].get('accepted') and (a['area']<1024)==small]
        selected.extend(int(i) for i in rng.choice(eligible,min(args.per_scale,len(eligible)),replace=False))
    cfg=YAMLConfig(str(ROOT/'reports/148_sgc2_half_box/frozen_repo/experiments/phase_s/s_sgc2_sam_c_plus_m_sd22_half_b8a4_20e.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    train=cfg.yaml_cfg['train_dataloader']['dataset']
    val=cfg.yaml_cfg['val_dataloader']['dataset']
    for key in ('img_folder','ann_file','infrared_folder','infrared_label_folder'):
        val[key]=train[key]
    dataset=cfg.val_dataloader.dataset
    indices=[dataset.ids.index(i) for i in selected]
    loader=DataLoader(Subset(dataset,indices),batch_size=8,shuffle=False,collate_fn=cfg.val_dataloader.collate_fn,num_workers=0)
    model=cfg.model.cuda().eval().requires_grad_(False)
    checkpoint=ROOT/'outputs/C_PLUS_M_SD22_HALF_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0/best_stg1.pth'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['ema']['module'],strict=True)
    bases={}
    with torch.no_grad():
        for batch,(samples,targets) in enumerate(loader):
            predictions=cfg.postprocessor(model(samples.cuda()),torch.stack([t['orig_size'] for t in targets]).cuda())
            for p,t in zip(predictions,targets):
                bases[int(t['image_id'])]=p['boxes'][p['scores'].argmax()].cpu().numpy()
            if batch%8==0: print('bases',len(bases),flush=True)
    samples=[]
    for image_id in selected:
        info=images[image_id]; a=annotations[image_id]
        x,y,w,h=a['bbox']; gt=np.array([x,y,x+w,y+h])
        base=bases[image_id].astype(float)
        if args.nonlinear:
            base=np.clip(base,0,[info['width'],info['height'],info['width'],info['height']])
        bw,bh=np.maximum(base[2:]-base[:2],1.)
        mask=np.asarray(Image.open(ROOT/f'reports/104_sam3_role_control/masks_train/masks/{image_id:06d}.png'))>0
        assert mask.shape==(info['height'],info['width'])
        gray=np.asarray(Image.open(Path(train['img_folder'])/info['file_name']).convert('L'),dtype=np.float32)/255
        edge=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
        edge=edge/max(float(np.percentile(edge,95)),1e-6)
        contour=(mask.astype(float)-cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))).astype(float)
        yy,xx=np.where(mask)
        mb=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1],dtype=float)
        base_iou=iou(base,gt)
        offsets=[(0,0,1,1),(-.1,0,1,1),(.1,0,1,1),(0,-.1,1,1),(0,.1,1,1),
                 (0,0,.9,1),(0,0,1.1,1),(0,0,1,.9),(0,0,1,1.1)]
        candidates=[]
        base_mask=stats(mask.astype(float),base); base_contour=stats(contour,base); base_edge=stats(edge,base)
        for dx,dy,sx,sy in offsets:
            center=(base[:2]+base[2:])/2+np.array([dx*bw,dy*bh])
            size=np.array([bw*sx,bh*sy])
            box=np.r_[center-size/2,center+size/2]
            box=np.maximum(box,0); box=np.minimum(box,[info['width'],info['height'],info['width'],info['height']])
            geom=[dx,dy,sx-1,sy-1,bw/info['width'],bh/info['height']]
            sm=stats(mask.astype(float),box); sc=stats(contour,box); se=stats(edge,box)
            candidates.append({'box':box.tolist(),'delta_iou':iou(box,gt)-base_iou,
                'geometry':geom,'extent':[iou(box,mb)-iou(base,mb)],
                'SAM':[u-v for u,v in zip(sm+sc,base_mask+base_contour)],
                'RGB':[u-v for u,v in zip(se,base_edge)]})
        samples.append({'image_id':image_id,'sequence':Path(info['file_name']).stem.rsplit('_',1)[0],
                        'scale':'small' if a['area']<1024 else 'other','base_iou':base_iou,'candidates':candidates})
    groups=sorted({s['sequence'] for s in samples}); rng.shuffle(groups)
    held_sets=[set(groups[:max(1,round(len(groups)*.25))])] if args.folds==1 else [set(groups[f::args.folds]) for f in range(args.folds)]
    splits=[{'train':[s for s in samples if s['sequence'] not in held],
             'held':[s for s in samples if s['sequence'] in held]} for held in held_sets]
    assert all(s['train'] and s['held'] for s in splits)
    results={}
    for label,fields in {'geometry_only':[], 'mask_extent':['extent'],'RGB_edges':['RGB'],
                         'SAM_shape':['SAM'],'RGB_plus_SAM':['RGB','SAM']}.items():
        def vector(c):return c['geometry']+([v*v for v in c['geometry'][:4]] if args.nonlinear else [])+sum([c[k] for k in fields],[])
        detail=[]
        for fold,split in enumerate(splits):
            X=np.asarray([vector(c) for s in split['train'] for c in s['candidates']]); Y=np.asarray([c['delta_iou'] for s in split['train'] for c in s['candidates']])
            mu=X.mean(0); std=np.maximum(X.std(0),1e-6)
            Z=np.c_[(X-mu)/std,np.ones(len(X))]
            penalty=np.eye(Z.shape[1]);penalty[-1,-1]=0
            beta=np.linalg.solve(Z.T@Z+10*penalty,Z.T@Y)
            for s in split['held']:
                candidates=s['candidates']; z=np.asarray([vector(c) for c in candidates])
                predicted=np.c_[(z-mu)/std,np.ones(len(z))]@beta
                predicted-=predicted[0]
                choice=int(predicted.argmax()) if predicted.max()>.005 else 0
                gain=candidates[choice]['delta_iou']
                detail.append({'image_id':s['image_id'],'scale':s['scale'],'chosen':choice,'gain':gain,'fold':fold,
                               'oracle_gain':max(c['delta_iou'] for c in candidates),'base_iou':s['base_iou']})
        assert len({r['image_id'] for r in detail})==len(detail)
        results[label]={}
        for scale in ('all','small','other'):
            rows=[r for r in detail if scale=='all' or r['scale']==scale]
            results[label][scale]={'n':len(rows),'mean_gain':float(np.mean([r['gain'] for r in rows])) if rows else None,
                                  'improved':sum(r['gain']>.01 for r in rows),'worsened':sum(r['gain']<-.01 for r in rows),
                                  'mean_oracle_gain':float(np.mean([r['oracle_gain'] for r in rows])) if rows else None}
        results[label]['rows']=detail
    result={'selected_ids':selected,'held_sequences_by_fold':[sorted(h) for h in held_sets],
            'split_sizes':[{'train':len(s['train']),'held':len(s['held'])} for s in splits],
            'nonlinear_geometry':args.nonlinear,'folds':args.folds,
            'ridge_lambda':10,'decision_gain_threshold':.005,'results':results,'samples':samples,
            'caveats':['exploratory fixed single sequence split, privileged GT-prompt SAM masks',
                       'frozen detector has already seen these training images; split tests regressor not detector generalization',
                       'GT used only for response labels and evaluation, never SAM/RGB feature descriptors',
                       'selected accepted masks; coverage limited, not AP or deployable student',
                       'GT rectangle is label oracle, not a fair teacher descriptor baseline']}
    (OUT/'teacher_geometry_pilot.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:{s:v[s] for s in ('all','small','other')} for k,v in results.items()},indent=2),flush=True)


if __name__=='__main__':main()
