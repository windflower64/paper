"""Same-image teacher / C comparison. Single-target dataset, no score calibration assumed."""
import json
import sys
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO.parent
OUT=ROOT/'reports/101_sam3_c_comparison'
sys.path.insert(0,str(REPO))
from tools.sam3_teacher_pilot import dump, iou


def run_c():
    import torch
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor
    from src.core import YAMLConfig
    torch.set_num_threads(4)
    cfg=YAMLConfig(str(REPO/'experiments/phase_m/c_only_gq1_b8a4_20e_testdev_local.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    cfg.yaml_cfg['DFINE']['mfam_enabled']=False
    model=cfg.model.cuda().eval()
    assert not model.rgbt_enabled and model.mfam is None
    checkpoint=ROOT/'outputs/C_ONLY_GQ1_B8A4_20E_TESTDEV/seed0/best_stg1.pth'
    state=torch.load(checkpoint,map_location='cpu')
    assert 'ema' in state
    model.load_state_dict(state['ema']['module'],strict=True)
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['split']=='test'
    dump(OUT/'c_runtime.json',dict(config=str(cfg.yaml_cfg.get('output_dir')),checkpoint=str(checkpoint),
                                 weight_source='ema',torch=torch.__version__,precision='float32',
                                 preprocessing='RGB original 640x512, float32 /255; matches config validation resize + convert'))
    with torch.inference_mode():
        for im in manifest['images']:
            path=OUT/'c_predictions'/f"{im['id']:06d}.json"
            if path.exists(): continue
            with Image.open(ROOT/'data/antiuav6k_common/images/test'/im['file_name']) as source:
                image=source.convert('RGB')
            assert image.size==(640,512), 'Must implement exact config resize for other image sizes'
            x=pil_to_tensor(image).float().div(255).unsqueeze(0).cuda()
            size=torch.tensor([[im['width'],im['height']]],device='cuda')
            pred=cfg.postprocessor(model(x),size)[0]
            order=pred['scores'].argsort(descending=True)
            dump(path,dict(image_id=im['id'],boxes=pred['boxes'][order].cpu().tolist(),
                           scores=pred['scores'][order].cpu().tolist()))
            print('C_IMAGE',im['id'],flush=True)


def summarize():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    rows=[]
    for im in manifest['images']:
        assert len(im['annotations'])<=1
        gt=None
        if im['annotations']:
            x,y,w,h=im['annotations'][0]['bbox']; gt=[x,y,x+w,y+h]
        row=dict(image_id=im['id'],group=im['group'],gt=gt,file_name=im['file_name'])
        for name,path in [('sam',OUT/'predictions'/f"{im['id']:06d}_0.json"),('c',OUT/'c_predictions'/f"{im['id']:06d}.json")]:
            pred=json.loads(path.read_text(encoding='utf-8'))
            assert pred['image_id']==im['id']
            assert len(pred['scores'])==len(pred['boxes'])
            assert pred['scores']==sorted(pred['scores'],reverse=True)
            row[name]=dict(scores=pred['scores'],ious=[iou(b,gt) if gt else 0 for b in pred['boxes']])
        rows.append(row)
    metrics={}
    for model in ['sam','c']:
        metrics[model]={}
        for threshold in [.2,.5,.8]:
            metrics[model][str(threshold)]={}
            for group in ['small','medium','empty']:
                subset=[r for r in rows if r['group']==group]
                hits50=hits75=predictions=empty_fp=0
                for r in subset:
                    overlaps=[v for v,s in zip(r[model]['ious'],r[model]['scores']) if s>=threshold]
                    hits50+=int(max(overlaps,default=0)>=.5)
                    hits75+=int(max(overlaps,default=0)>=.75)
                    predictions+=len(overlaps)
                    empty_fp+=int(group=='empty' and bool(overlaps))
                metrics[model][str(threshold)][group]=dict(images=len(subset),hits50=hits50,hits75=hits75,
                                                         predictions=predictions,empty_false_positive_images=empty_fp)
    overlap={}
    for cutoff in [.5,.75]:
        for sam_t,c_t in [(.2,.2),(.2,.5),(.5,.5),(.8,.8)]:
            key=f'iou{cutoff}_sam{sam_t}_c{c_t}'
            counters={g:{k:[] for k in ['both','sam_only','c_only','neither']} for g in ['small','medium']}
            for row in rows:
                if row['group']=='empty': continue
                a=any(v>=cutoff and s>=sam_t for v,s in zip(row['sam']['ious'],row['sam']['scores']))
                b=any(v>=cutoff and s>=c_t for v,s in zip(row['c']['ious'],row['c']['scores']))
                tag='both' if a and b else 'sam_only' if a else 'c_only' if b else 'neither'
                counters[row['group']][tag].append(row['image_id'])
            overlap[key]=counters
    # Teacher hits where even all retained C queries have no correct geometry.
    geometry={}
    for cutoff in [.5,.75]:
        geometry[str(cutoff)]=[r['image_id'] for r in rows if r['gt'] and
                              max(r['sam']['ious'],default=0)>=cutoff and max(r['c']['ious'],default=0)<cutoff]
    result=dict(images=len(rows),complete=True,metrics=metrics,overlap=overlap,
                sam_recovers_missing_c_query_geometry=geometry,
                caveat='test is development validation; stratified subset, not full AP; model scores uncalibrated')
    dump(OUT/'comparison_rows.json',rows)
    dump(OUT/'comparison_summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def figures():
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    selected=[im for group in ['small','medium'] for im in [x for x in manifest['images'] if x['group']==group][:3]]
    fig,axes=plt.subplots(6,3,figsize=(15,16))
    for row,im in enumerate(selected):
        source=np.asarray(Image.open(ROOT/'data/antiuav6k_common/images/test'/im['file_name']).convert('RGB'))
        x,y,w,h=im['annotations'][0]['bbox']; gt=[x,y,x+w,y+h]
        cx,cy=x+w/2,y+h/2; half=max(48,w,h)
        crop=[max(0,cx-half),max(0,cy-half),min(im['width'],cx+half),min(im['height'],cy+half)]
        c=json.loads((OUT/'c_predictions'/f"{im['id']:06d}.json").read_text())
        sam=json.loads((OUT/'predictions'/f"{im['id']:06d}_0.json").read_text())
        with np.load(OUT/'masks'/f"{im['id']:06d}_0.npz") as archive:
            masks=archive['masks']
        for col in range(3):
            ax=axes[row,col]; ax.imshow(source)
            if col==2 and len(masks):
                rgba=np.zeros((*masks[0].shape,4)); rgba[masks[0]]=[1,0,1,.4]
                ax.imshow(rgba)
            entries=[(gt,'lime','GT')]
            if col==1:
                if c['boxes']: entries.append((c['boxes'][0],'cyan',f"C {c['scores'][0]:.2f}"))
                if sam['boxes']: entries.append((sam['boxes'][0],'orange',f"SAM {sam['scores'][0]:.2f}"))
            for box,color,label in entries:
                ax.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,edgecolor=color,linewidth=1.2))
            if col==0:
                ax.add_patch(Rectangle((crop[0],crop[1]),crop[2]-crop[0],crop[3]-crop[1],fill=False,edgecolor='white',linewidth=1))
            else:
                ax.set_xlim(crop[0],crop[2]); ax.set_ylim(crop[3],crop[1])
            title=f"{im['id']} {im['group']}" if col==0 else ('GT=green C=cyan SAM=orange' if col==1 else ('SAM top1 mask' if len(masks) else 'SAM: no prediction >0.2'))
            ax.set_title(title,fontsize=9); ax.axis('off')
    fig.suptitle('Fixed first 3 small + 3 medium samples; GT crop for inspection ONLY',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.98))
    fig.savefig(OUT/'fixed6_mask_review.png',dpi=140)
    plt.close(fig)
    dump(OUT/'fixed6_mask_review_ids.json',[im['id'] for im in selected])
    print('FIGURE',OUT/'fixed6_mask_review.png')


if __name__=='__main__':
    if sys.argv[1]=='c': run_c()
    elif sys.argv[1]=='summary': summarize()
    elif sys.argv[1]=='figures': figures()
    else: raise ValueError(sys.argv[1])
