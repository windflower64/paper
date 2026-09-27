"""Paired full-test prediction audit for C, SGC1-SAM and SGC1-BOX."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO=Path(__file__).resolve().parents[1]; ROOT=REPO.parent
OUT=ROOT/'reports/107_sgc1_per_image_diagnosis'
sys.path.insert(0,str(REPO))

RUNS={
    'c': ('experiments/phase_m/c_only_gq1_b8a4_20e_testdev_local.yml',
          'outputs/C_ONLY_GQ1_B8A4_20E_TESTDEV/seed0/best_stg1.pth'),
    'sam': ('experiments/phase_s/s_sgc1_sam_b8a4_20e.yml',
            'outputs/S_SGC1_SAM_B8A4_20E_TESTDEV/seed0/best_stg1.pth'),
    'box': ('experiments/phase_s/s_sgc1_box_b8a4_20e.yml',
            'outputs/S_SGC1_BOX_B8A4_20E_TESTDEV/seed0/best_stg1.pth'),
}


def predict(arm):
    from src.core import YAMLConfig
    cfg_path=REPO/RUNS[arm][0]; checkpoint=ROOT/RUNS[arm][1]
    destination=OUT/f'{arm}_predictions.npz'; metadata=OUT/f'{arm}_metadata.json'
    OUT.mkdir(parents=True,exist_ok=True)
    if destination.exists() or metadata.exists(): raise RuntimeError(f'Existing output for {arm}')
    cfg=YAMLConfig(str(cfg_path)); cfg.yaml_cfg['HGNetv2']['pretrained']=False
    model=cfg.model.cuda().eval(); state=torch.load(checkpoint,map_location='cpu')
    assert 'ema' in state; model.load_state_dict(state['ema']['module'],strict=True)
    loader=cfg.val_dataloader
    assert loader.dataset.sam_mask_root is None and len(loader.dataset)==1820
    all_ids=[]; all_boxes=[]; all_scores=[]; all_labels=[]
    torch.set_num_threads(4)
    with torch.inference_mode():
        for step,(samples,targets) in enumerate(loader):
            samples=samples.cuda()
            sizes=torch.stack([t['orig_size'] for t in targets]).cuda()
            results=cfg.postprocessor(model(samples),sizes)
            for target,result in zip(targets,results):
                all_ids.append(int(target['image_id'].item()))
                all_boxes.append(result['boxes'].cpu().float().numpy())
                all_scores.append(result['scores'].cpu().float().numpy())
                all_labels.append(result['labels'].cpu().numpy())
            if step%50==0: print(arm,step,len(all_ids),flush=True)
    assert len(set(all_ids))==len(all_ids)==1820
    boxes=np.stack(all_boxes); scores=np.stack(all_scores); labels=np.stack(all_labels)
    assert boxes.shape==(1820,300,4) and scores.shape==(1820,300)
    np.savez_compressed(destination,image_ids=np.array(all_ids,dtype=np.int64),
                        boxes=boxes,scores=scores,labels=labels)
    metadata.write_text(json.dumps(dict(arm=arm,config=str(cfg_path),checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),images=1820,
        predictions_per_image=300,weight_source='ema',split='original test used as development validation',
        validation_sam_cache=None),indent=2),encoding='utf-8')


def ious(boxes,gt):
    left=np.maximum(boxes[:,0],gt[0]); top=np.maximum(boxes[:,1],gt[1])
    right=np.minimum(boxes[:,2],gt[2]); bottom=np.minimum(boxes[:,3],gt[3])
    inter=np.maximum(0,right-left)*np.maximum(0,bottom-top)
    area=np.maximum(0,boxes[:,2]-boxes[:,0])*np.maximum(0,boxes[:,3]-boxes[:,1])
    gt_area=max(0,gt[2]-gt[0])*max(0,gt[3]-gt[1])
    return inter/np.maximum(area+gt_area-inter,1e-12)


def image_stats(boxes,scores,gt):
    overlap=ious(boxes,gt)
    visible=scores>=.2
    out={'top_score':float(scores.max()),'best_iou_at_02':float(overlap[visible].max(initial=0))}
    for threshold in np.arange(.5,1.,.05):
        key=f'{threshold:.2f}'
        valid=overlap>=threshold
        out[f'matched_score_{key}']=float(scores[valid].max(initial=0))
        out[f'hit02_{key}']=bool(np.any(valid&visible))
    return out


def analyze():
    # Eagerly materialize compressed arrays once. Repeated lazy NPZ indexing
    # would decompress a full member for every image and appear to hang.
    predictions={}
    for arm in RUNS:
        with np.load(OUT/f'{arm}_predictions.npz') as archive:
            predictions[arm]={key:archive[key] for key in archive.files}
    ids=predictions['c']['image_ids']
    assert all(np.array_equal(ids,p['image_ids']) for p in predictions.values())
    annotation_path=ROOT/'data/antiuav6k_common/annotations/instances_visible_common_test.json'
    coco=json.loads(annotation_path.read_text(encoding='utf-8'))
    images={i['id']:i for i in coco['images']}; anns={i:[] for i in images}
    for a in coco['annotations']:
        if not a.get('iscrowd',0): anns[a['image_id']].append(a)
    rows=[]
    for index,image_id in enumerate(ids.tolist()):
        assert len(anns[image_id])<=1
        annotation=anns[image_id]
        area=0 if not annotation else annotation[0]['bbox'][2]*annotation[0]['bbox'][3]
        group='empty' if not annotation else 'small' if area<32**2 else 'medium' if area<96**2 else 'large'
        row=dict(image_id=image_id,file_name=images[image_id]['file_name'],group=group,area=area)
        if annotation:
            x,y,w,h=annotation[0]['bbox']; row['gt']=[x,y,x+w,y+h]
            row['models']={arm:image_stats(p['boxes'][index],p['scores'][index],row['gt']) for arm,p in predictions.items()}
        else:
            row['models']={arm:{'top_score':float(p['scores'][index].max())} for arm,p in predictions.items()}
        rows.append(row)
    summary={'source_annotation':str(annotation_path),'images':len(rows),'groups':{},'paired':{}}
    for group in ('small','medium','empty'):
        subset=[r for r in rows if r['group']==group]; summary['groups'][group]={'images':len(subset)}
        if group=='empty':
            for arm in RUNS:
                scores=[r['models'][arm]['top_score'] for r in subset]
                summary['groups'][group][arm]=dict(fp02=sum(s>=.2 for s in scores),fp05=sum(s>=.5 for s in scores),
                    top_score_mean=float(np.mean(scores)),top_score_p95=float(np.quantile(scores,.95)))
        else:
            for arm in RUNS:
                summary['groups'][group][arm]={}
                for t in (.5,.75,.9):
                    k=f'{t:.2f}'
                    summary['groups'][group][arm][k]=dict(hit02=sum(r['models'][arm][f'hit02_{k}'] for r in subset),
                        matched_score_mean=float(np.mean([r['models'][arm][f'matched_score_{k}'] for r in subset])),
                        best_iou_at_02_mean=float(np.mean([r['models'][arm]['best_iou_at_02'] for r in subset])))
    for challenger in ('sam','box'):
        summary['paired'][challenger]={}
        for group in ('small','medium'):
            subset=[r for r in rows if r['group']==group]; item={}
            for t in (.5,.75,.9):
                k=f'{t:.2f}'; ckey=f'hit02_{k}'
                both=[r for r in subset if r['models']['c'][ckey] and r['models'][challenger][ckey]]
                item[k]=dict(recovered=sum(not r['models']['c'][ckey] and r['models'][challenger][ckey] for r in subset),
                    lost=sum(r['models']['c'][ckey] and not r['models'][challenger][ckey] for r in subset),
                    unchanged_hit=sum(r['models']['c'][ckey] and r['models'][challenger][ckey] for r in subset),
                    matched_score_delta_mean=float(np.mean([r['models'][challenger][f'matched_score_{k}']-r['models']['c'][f'matched_score_{k}'] for r in subset])),
                    both_hit_score_delta_mean=float(np.mean([r['models'][challenger][f'matched_score_{k}']-r['models']['c'][f'matched_score_{k}'] for r in both])) if both else None)
            deltas=[r['models'][challenger]['best_iou_at_02']-r['models']['c']['best_iou_at_02'] for r in subset]
            item['localization']=dict(improved_gt_002=sum(d>.02 for d in deltas),degraded_lt_minus002=sum(d<-.02 for d in deltas),
                                       mean_delta=float(np.mean(deltas)),median_delta=float(np.median(deltas)))
            summary['paired'][challenger][group]=item
        transitions={}
        for r in [x for x in rows if x['group']=='small']:
            c=r['models']['c']['hit02_0.75']; v=r['models'][challenger]['hit02_0.75']
            label='recovered' if not c and v else 'lost' if c and not v else None
            if label:
                sequence=r['file_name'].rsplit('_',1)[0]
                transitions.setdefault(sequence,{'recovered':0,'lost':0})[label]+=1
        summary['paired'][challenger]['small_iou75_sequences']=dict(
            sequences_with_transition=len(transitions),
            top=sorted((dict(sequence=k,**v) for k,v in transitions.items()),
                       key=lambda x:x['recovered']+x['lost'],reverse=True)[:10])
    # Fixed ranking for inspection, never used to estimate aggregate performance.
    small=[r for r in rows if r['group']=='small']
    ranked=sorted(small,key=lambda r:r['models']['sam']['best_iou_at_02']-r['models']['c']['best_iou_at_02'],reverse=True)
    review=ranked[:6]+ranked[-6:]
    summary['review_ids_improved']=[r['image_id'] for r in ranked[:6]]
    summary['review_ids_degraded']=[r['image_id'] for r in ranked[-6:]]
    (OUT/'per_image.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # One compact audit sheet: GT yellow, C red, SGC1-SAM green, highest-IoU >= score .2.
    sheet=Image.new('RGB',(640*3,300*4),'white'); draw=ImageDraw.Draw(sheet)
    for cell,row in enumerate(review):
        src=Image.open(ROOT/'data/antiuav6k_common/images/test'/row['file_name']).convert('RGB').resize((320,256))
        x0=(cell%3)*640; y0=(cell//3)*300
        sheet.paste(src,(x0,y0+32)); sheet.paste(src,(x0+320,y0+32))
        draw.text((x0+4,y0+4),f"{row['image_id']} C  IoU {row['models']['c']['best_iou_at_02']:.3f}",fill='black')
        draw.text((x0+324,y0+4),f"SAM  IoU {row['models']['sam']['best_iou_at_02']:.3f}",fill='black')
        idx=int(np.where(ids==row['image_id'])[0][0]); gt=np.array(row['gt'])*.5
        for offset,arm,color in [(0,'c','red'),(320,'sam','green')]:
            boxes=predictions[arm]['boxes'][idx]; scores=predictions[arm]['scores'][idx]
            ov=ious(boxes,row['gt']); valid=np.where(scores>=.2)[0]
            best=int(valid[np.argmax(ov[valid])]) if len(valid) else int(np.argmax(ov))
            box=boxes[best]*.5
            draw.rectangle([x0+offset+gt[0],y0+32+gt[1],x0+offset+gt[2],y0+32+gt[3]],outline='yellow',width=2)
            draw.rectangle([x0+offset+box[0],y0+32+box[1],x0+offset+box[2],y0+32+box[3]],outline=color,width=2)
    sheet.save(OUT/'small_localization_changes.png')
    print(json.dumps(summary['paired'],indent=2),flush=True)


def validate():
    """Prove saved arrays reproduce the authoritative COCO metrics."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    expected={
        'c':[0.5337824641983735,0.9354903448236579,0.5674758082609308,0.47150654144672743,0.5644811836973361,-1.0,0.5856298048492016,0.6325251330573625,0.66989946777055,0.6258715596330275,0.6908376963350784,-1.0],
        'sam':[0.533634531020575,0.9386886290950999,0.5765689042806637,0.4755283537908791,0.562989459448192,-1.0,0.5842696629213483,0.6363690124186873,0.6790065050266115,0.6365137614678898,0.6992146596858638,-1.0],
        'box':[0.5337753424796962,0.9372684393497484,0.577346538047248,0.4575897245579841,0.566013039917309,-1.0,0.582909520993495,0.6545830869308101,0.6917800118273212,0.6438532110091744,0.7145724258289703,-1.0],
    }
    annotation=ROOT/'data/antiuav6k_common/annotations/instances_visible_common_test.json'
    gt=COCO(str(annotation)); output={}
    for arm in RUNS:
        with np.load(OUT/f'{arm}_predictions.npz') as p:
            ids=p['image_ids']; boxes=p['boxes']; scores=p['scores']; labels=p['labels']
        records=[]
        for image_id,image_boxes,image_scores,image_labels in zip(ids,boxes,scores,labels):
            for box,score,label in zip(image_boxes,image_scores,image_labels):
                records.append(dict(image_id=int(image_id),category_id=int(label),score=float(score),
                    bbox=[float(box[0]),float(box[1]),float(box[2]-box[0]),float(box[3]-box[1])]))
        dt=gt.loadRes(records); evaluator=COCOeval(gt,dt,'bbox'); evaluator.params.imgIds=ids.tolist()
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        metrics=evaluator.stats.tolist(); maximum=max(abs(a-b) for a,b in zip(metrics,expected[arm]))
        # A fresh CUDA forward can reorder nearly tied detections by tiny amounts.
        # Keep a strict practical bound while reporting the actual discrepancy.
        assert maximum<2e-4,(arm,maximum,metrics)
        output[arm]=dict(coco_eval_bbox=metrics,expected=expected[arm],max_abs_error=maximum,
                         tolerance=2e-4,status='PASS_CLOSE_REPRODUCTION')
    (OUT/'coco_reproduction.json').write_text(json.dumps(output,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['predict','analyze','validate']); parser.add_argument('--arm',choices=list(RUNS))
    args=parser.parse_args()
    if args.mode=='predict':
        if args.arm is None: raise ValueError('--arm required')
        predict(args.arm)
    elif args.mode=='analyze': analyze()
    else: validate()
