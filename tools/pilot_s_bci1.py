"""Train-only foreground-preserving background perturbation audit.

KeepAugment-inspired principle, not a reproduction. No optimizer or new labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms.functional import pil_to_tensor

REPO=Path(__file__).resolve().parents[1]; ROOT=REPO.parent
OUT=ROOT/'reports/106_sam_background_consistency'
sys.path.insert(0,str(REPO))


def protection(mask):
    """Preserve the binary mask plus 3 px; feather only outside that core."""
    mask=mask.float()[None,None]
    core=F.max_pool2d(mask,7,1,3)
    expanded=F.max_pool2d(core,9,1,4)
    alpha=F.avg_pool2d(expanded,9,1,4,count_include_pad=False)
    return torch.where(core.bool(),torch.ones_like(alpha),alpha)[0]


def perturb(image, alpha, strength, signs):
    if not 0 <= strength <= .3: raise ValueError(strength)
    changed=((image-.5)*(1+signs[0]*strength)+.5+signs[1]*strength*.25).clamp(0,1)
    # Exact identity on protected pixels, not a floating point blend there.
    return torch.where(alpha==1,image,alpha*image+(1-alpha)*changed)


def texture_perturb(image, alpha, strength, signs=None):
    """Weak background-only normalized blur; foreground cannot bleed into blur."""
    if not 0 <= strength <= .3: raise ValueError(strength)
    if strength==0: return image.clone()
    weight=1-alpha
    numerator=F.avg_pool2d((image*weight)[None],9,1,4,count_include_pad=False)[0]
    denominator=F.avg_pool2d(weight[None],9,1,4,count_include_pad=False)[0]
    blurred=torch.where(denominator>1e-6,numerator/denominator.clamp_min(1e-6),image)
    changed=(image+strength*weight*(blurred-image)).clamp(0,1)
    return torch.where(alpha==1,image,changed)


def box_mask(annotation,h,w):
    mask=torch.zeros(h,w)
    if annotation:
        x,y,bw,bh=annotation[0]['bbox']
        mask[max(0,int(np.floor(y))):min(h,int(np.ceil(y+bh))),
             max(0,int(np.floor(x))):min(w,int(np.ceil(x+bw)))]=1
    return mask


def target_size(image, batch, device):
    # This repository stores orig_size as [width, height], unlike some DETRs.
    width,height=image.size
    return torch.tensor([[width,height]]*batch,device=device)


def measure(pred, annotation):
    from tools.sam3_teacher_pilot import iou
    scores=pred['scores'].cpu().tolist(); boxes=pred['boxes'].cpu().tolist()
    overlaps=[0.]*len(boxes)
    if annotation:
        x,y,w,h=annotation[0]['bbox']; gt=[x,y,x+w,y+h]
        overlaps=[iou(b,gt) for b in boxes]
    return dict(top_score=max(scores,default=0),
        **{f'hit{int(t*100)}':any(s>=.2 and v>=t for s,v in zip(scores,overlaps)) for t in (.5,.75)},
        best_iou_at_02=max((v for s,v in zip(scores,overlaps) if s>=.2),default=0),
        correct_score=max((s for s,v in zip(scores,overlaps) if v>=.5),default=0),
        false_positive_empty=not annotation and max(scores,default=0)>=.2)


def run(variant='photometric'):
    from src.core import YAMLConfig
    torch.set_num_threads(4)
    manifest_path=ROOT/'reports/100_sam3_teacher/manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')); assert manifest['split']=='train'
    cache=ROOT/'reports/104_sam3_role_control/masks_train'
    records={r['image_id']:r for r in json.loads((cache/'records.json').read_text())}
    cfg=YAMLConfig(str(REPO/'experiments/phase_m/c_only_gq1_b8a4_20e_testdev_local.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    model=cfg.model.cuda().eval()
    checkpoint=ROOT/'outputs/C_ONLY_GQ1_B8A4_20E_TESTDEV/seed0/best_stg1.pth'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['ema']['module'],strict=True)
    assert not model.rgbt_enabled and model.mfam is None and not model.sgc_enabled
    OUT.mkdir(parents=True,exist_ok=True)
    destination=OUT/'pilot_wh_v1.json'
    if destination.exists(): raise RuntimeError('Existing pilot result; do not overwrite')
    review_ids={im['id'] for im in manifest['images'] if im['annotations'] and records[im['id']]['accepted']}
    review_ids=set(random.Random(20260907).sample(sorted(review_ids),5))
    rows=[]
    with torch.inference_mode():
        for im in manifest['images']:
            annotation=im['annotations']; assert len(annotation)<=1
            if annotation and not records[im['id']]['accepted']:
                rows.append(dict(image_id=im['id'],group=im['group'],skipped='teacher_rejected')); continue
            image=Image.open(ROOT/'data/antiuav6k_common/images/train'/im['file_name']).convert('RGB')
            assert image.size==(640,512)
            x=pil_to_tensor(image).float().div(255).cuda()
            mask=torch.zeros(512,640)
            if annotation:
                mask=torch.from_numpy(np.array(Image.open(cache/'masks'/f"{im['id']:06d}.png"),copy=True)>0).float()
                assert mask.any()
            sam=protection(mask).cuda(); box=protection(box_mask(annotation,512,640)).cuda()
            rng=random.Random(20260907+im['id']); signs=[rng.choice([-1,1]),rng.choice([-1,1])]
            views={'clean':x}; audit={}
            for source,alpha in [('sam',sam),('box',box),('global',torch.zeros_like(sam))]:
                for strength in (.1,.2,.3):
                    key=f'{source}_{strength:.1f}'
                    operator=perturb if variant=='photometric' else texture_perturb
                    views[key]=operator(x,alpha,strength,signs)
                    diff=(views[key]-x).abs()
                    protected=diff[:,alpha[0]==1]
                    error=protected.max().item() if protected.numel() else 0.
                    assert error==0. and torch.isfinite(views[key]).all()
                    audit[key]=dict(protected_max_error=error,image_mae=diff.mean().item(),
                                    protected_fraction=(alpha==1).float().mean().item())
            row=dict(image_id=im['id'],group=im['group'],signs=signs,audit=audit,predictions={})
            keys=list(views)
            for start in range(0,len(keys),8):
                kk=keys[start:start+8]
                preds=cfg.postprocessor(model(torch.stack([views[k] for k in kk])),
                                        target_size(image,len(kk),'cuda'))
                for k,p in zip(kk,preds): row['predictions'][k]=measure(p,annotation)
            rows.append(row)
            if im['id'] in review_ids:
                sheet=Image.new('RGB',(640*3,540),'white'); draw=ImageDraw.Draw(sheet)
                for j,k in enumerate(('clean','sam_0.2','box_0.2')):
                    pix=(views[k].cpu().permute(1,2,0).numpy()*255).round().astype(np.uint8)
                    sheet.paste(Image.fromarray(pix),(640*j,28)); draw.text((640*j+8,6),f'{im["id"]}  {k}',fill='black')
                sheet.save(OUT/f'review_{im["id"]:06d}.png')
            print('IMAGE',im['id'],im['group'],row['predictions']['clean']['hit50'],flush=True)
    result=dict(split='train',variant=variant,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                label_records_sha256=hashlib.sha256((cache/'records.json').read_bytes()).hexdigest(),
                review_ids=sorted(review_ids),confidence_threshold=.2,
                note='Stratified train-side sensitivity audit, NOT held-out AP or proof of augmentation benefit',rows=rows)
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')


def summary():
    data=json.loads((OUT/'pilot_wh_v1.json').read_text()); rows=[r for r in data['rows'] if 'skipped' not in r]
    result={'skipped':[r['image_id'] for r in data['rows'] if 'skipped' in r], 'groups':{}}
    for group in ('small','medium','empty'):
        subset=[r for r in rows if r['group']==group]; result['groups'][group]={}
        for key in rows[0]['predictions']:
            predictions=[r['predictions'][key] for r in subset]
            result['groups'][group][key]=dict(images=len(subset),hits50=sum(p['hit50'] for p in predictions),
                hits75=sum(p['hit75'] for p in predictions),empty_fp=sum(p['false_positive_empty'] for p in predictions),
                lost50=sum(r['predictions']['clean']['hit50'] and not r['predictions'][key]['hit50'] for r in subset),
                gained50=sum(not r['predictions']['clean']['hit50'] and r['predictions'][key]['hit50'] for r in subset),
                lost75=sum(r['predictions']['clean']['hit75'] and not r['predictions'][key]['hit75'] for r in subset),
                gained75=sum(not r['predictions']['clean']['hit75'] and r['predictions'][key]['hit75'] for r in subset),
                mean_abs_correct_score_change=float(np.mean([abs(r['predictions'][key]['correct_score']-r['predictions']['clean']['correct_score']) for r in subset])))
    (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['run','summary'])
    parser.add_argument('--variant',choices=['photometric','texture'],default='photometric')
    args=parser.parse_args()
    if args.variant=='texture': OUT=OUT/'texture'
    if args.mode=='run': run(args.variant)
    else: summary()
