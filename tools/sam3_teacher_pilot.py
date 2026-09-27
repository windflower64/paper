"""Fixed train-side, full-image text-only SAM3 screening. No detector training."""
import argparse
import json
import random
import time
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/100_sam3_teacher'
PROMPTS = ['drone', 'unmanned aerial vehicle']


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(split='train', seed=20260906):
    source = ROOT/f'data/antiuav6k_common/annotations/instances_visible_common_{split}.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    anns = defaultdict(list)
    for ann in data['annotations']:
        if not ann.get('iscrowd', 0):
            anns[ann['image_id']].append(ann)
    pools = defaultdict(list)
    for image in sorted(data['images'], key=lambda x: x['id']):
        aa = anns[image['id']]
        area = max((a['bbox'][2]*a['bbox'][3] for a in aa), default=0)
        group = 'empty' if not aa else 'small' if area < 1024 else 'medium' if area < 9216 else 'large'
        pools[group].append(dict(**image, group=group, annotations=aa))
    rng = random.Random(seed)
    selected = []
    for group, count in [('small',80),('medium',80),('empty',40)]:
        assert len(pools[group]) >= count, (group,len(pools[group]))
        selected.extend(rng.sample(pools[group], count))
    rng.shuffle(selected)
    result = dict(seed=seed, split=split, prompts=PROMPTS,
                  note='Teacher screening only; not held-out accuracy or full AP. GT never passed to teacher.',
                  counts=dict(Counter(x['group'] for x in selected)), images=selected)
    path = OUT/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == result, 'Existing manifest differs'
    else:
        dump(path,result)
    print('MANIFEST',result['counts'],flush=True)


def iou(a,b):
    inter = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
    return inter/max(union,1e-12)


def summarize():
    manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    summary = {'expected_images':len(manifest['images']), 'prompts':{}}
    for pi,prompt in enumerate(PROMPTS):
        records = []
        for im in manifest['images']:
            path = OUT/'predictions'/f"{im['id']:06d}_{pi}.json"
            if path.exists():
                records.append((im,json.loads(path.read_text(encoding='utf-8'))))
        metrics = {}
        for threshold in [.2,.5,.8]:
            groups = {}
            for group in ['small','medium','empty']:
                n=ngt=tp50=tp75=npred=emptyfp=0
                for im,pred in records:
                    if im['group'] != group:
                        continue
                    n+=1
                    boxes = [b for b,s in zip(pred['boxes'],pred['scores']) if s>=threshold]
                    gt = [[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in im['annotations']]
                    ngt+=len(gt); npred+=len(boxes); emptyfp+=int(not gt and bool(boxes))
                    # Score-ordered one-to-one matching, independently at each IoU.
                    for cutoff in [.5,.75]:
                        used=set(); matched=0
                        for box in boxes:
                            candidates=[(iou(box,g),j) for j,g in enumerate(gt) if j not in used]
                            best=max(candidates,default=(0,-1))
                            if best[0]>=cutoff:
                                used.add(best[1]); matched+=1
                        if cutoff==.5: tp50+=matched
                        else: tp75+=matched
                    
                groups[group]=dict(images=n,targets=ngt,predictions=npred,matched50=tp50,matched75=tp75,
                                   recall50=tp50/ngt if ngt else None,recall75=tp75/ngt if ngt else None,
                                   precision50=tp50/npred if npred else None,empty_false_positive_images=emptyfp)
            metrics[str(threshold)]=groups
        summary['prompts'][prompt]=dict(completed_images=len(records),metrics=metrics)
    summary['complete']=all(v['completed_images']==summary['expected_images'] for v in summary['prompts'].values())
    dump(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def run(limit, candidate_root=None):
    import sys
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0,str(ROOT/'_third_party/sam3'))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.set_num_threads(4)
    torch.manual_seed(20260906)
    manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    model = build_sam3_image_model(checkpoint_path=str(ROOT/'weights/sam3/sam3.pt'),load_from_HF=False,compile=False)
    processor = Sam3Processor(model,confidence_threshold=.2)
    dump(OUT/'runtime.json',dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
                               prompts=PROMPTS,confidence_floor=.2,resolution=1008,batch=1,precision='bfloat16',
                               source='official SAM3, local user-supplied checkpoint',geometric_prompt=candidate_root is not None,
                               candidate_root=str(candidate_root) if candidate_root else None))
    images = manifest['images'][:limit] if limit else manifest['images']
    for index,im in enumerate(images):
        paths=[OUT/'predictions'/f"{im['id']:06d}_{pi}.json" for pi in range(len(PROMPTS))]
        if all(p.exists() for p in paths):
            continue
        start=time.perf_counter()
        with Image.open(ROOT/f"data/antiuav6k_common/images/{manifest['split']}"/im['file_name']) as source:
            image=source.convert('RGB')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=processor.set_image(image)
            for pi,prompt in enumerate(PROMPTS):
                if paths[pi].exists():
                    continue
                processor.reset_all_prompts(state)
                prediction=processor.set_text_prompt(prompt=prompt,state=state)
                candidate=None
                if candidate_root is not None:
                    cp=json.loads((candidate_root/f"{im['id']:06d}.json").read_text(encoding='utf-8'))
                    assert cp['image_id']==im['id'] and len(cp['boxes'])>0
                    x0,y0,x1,y1=cp['boxes'][0]
                    x0,x1=np.clip([x0,x1],0,im['width'])
                    y0,y1=np.clip([y0,y1],0,im['height'])
                    assert x1>x0 and y1>y0
                    candidate=[float((x0+x1)/2/im['width']),float((y0+y1)/2/im['height']),
                               float((x1-x0)/im['width']),float((y1-y0)/im['height'])]
                    prediction=processor.add_geometric_prompt(box=candidate,label=True,state=state)
                scores=prediction['scores'].float().cpu().numpy()
                order=np.argsort(-scores)
                boxes=prediction['boxes'].float().cpu().numpy()[order]
                masks=prediction['masks'].cpu().numpy()[order[:5],0]
                maskpath=OUT/'masks'/f"{im['id']:06d}_{pi}.npz"
                maskpath.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(maskpath,masks=masks)
                dump(paths[pi],dict(image_id=im['id'],prompt=prompt,boxes=boxes.tolist(),scores=scores[order].tolist(),
                                     candidate_cxcywh_normalized=candidate,
                                     saved_masks='top5, ordered by score; original image resolution'))
        del state
        print(f"IMAGE {index+1}/{len(images)} id={im['id']} group={im['group']} seconds={time.perf_counter()-start:.2f}",flush=True)
    summarize()


def verify():
    import numpy as np
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    expected=len(manifest['images'])*len(PROMPTS)
    assert len(list((OUT/'predictions').glob('*.json')))==expected
    assert len(list((OUT/'masks').glob('*.npz')))==expected
    for im in manifest['images']:
        for pi,prompt in enumerate(PROMPTS):
            stem=f"{im['id']:06d}_{pi}"
            pred=json.loads((OUT/'predictions'/f'{stem}.json').read_text(encoding='utf-8'))
            assert pred['image_id']==im['id'] and pred['prompt']==prompt
            assert len(pred['boxes'])==len(pred['scores'])
            assert pred['scores']==sorted(pred['scores'],reverse=True)
            assert all(np.isfinite(b).all() for b in pred['boxes'])
            with np.load(OUT/'masks'/f'{stem}.npz') as data:
                assert data['masks'].shape==(min(5,len(pred['scores'])),im['height'],im['width'])
                assert data['masks'].dtype==np.bool_
    assert json.loads((OUT/'summary.json').read_text(encoding='utf-8'))['complete']
    print(f'PASS: {expected} predictions and {expected} mask archives; IDs/prompts/shapes/sorting/finiteness/completeness')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','summary','test','verify'])
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--split',choices=['train','test'],default='train')
    parser.add_argument('--seed',type=int,default=20260906)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--drone-only',action='store_true')
    parser.add_argument('--candidate-root',type=Path)
    args=parser.parse_args()
    OUT=args.output
    if args.drone_only: PROMPTS=['drone']
    if args.action!='prepare' and (OUT/'manifest.json').exists():
        assert json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))['prompts']==PROMPTS
    if args.action=='prepare': prepare(args.split,args.seed)
    elif args.action=='run': run(args.limit,args.candidate_root)
    elif args.action=='summary': summarize()
    elif args.action=='verify': verify()
    else:
        assert iou([0,0,2,2],[0,0,2,2])==1
        assert iou([0,0,1,1],[2,2,3,3])==0
        assert abs(iou([0,0,2,2],[1,1,3,3])-1/7)<1e-12
        print('IoU tests PASS')
