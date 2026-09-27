"""Reuse four trained arms and M on/off caches; no inference/training mutation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path('E:/two_paper')
OUT = ROOT/'reports/156_sam_m_gain_retention'
RUNS = {
    'C': 'C_ONLY_GQ1_B8A4_20E_TESTDEV',
    'CS': 'S_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV',
    'CM': 'C_PLUS_M_SD22_HALF_SGC0_B8A4_20E_TESTDEV',
    'CMS': 'C_PLUS_M_SD22_HALF_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV',
}
CACHE_PATHS = {
    'C': 'reports/107_sgc1_per_image_diagnosis/c_predictions.npz',
    'CS': 'reports/109_sgc2_teacher_retirement/sgc2_predictions.npz',
    'CM': 'reports/113_sgc0_half_control/final/diagnostics/epoch13/predictions.pt',
    'CMS': 'reports/112_sgc2_rgbt_half/final/diagnostics/epoch8/predictions.pt',
}


def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def stats(boxes, scores, gt):
    order = np.argsort(-scores, kind='stable'); boxes, scores = boxes[order], scores[order]
    area = np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(1)
    inter = np.maximum(np.minimum(boxes[:,2:],gt[2:])-np.maximum(boxes[:,:2],gt[:2]),0).prod(1)
    ious = inter/np.maximum(area+np.prod(gt[2:]-gt[:2])-inter,1e-8)
    result = dict(top1_iou=float(ious[0]), top1_score=float(scores[0]), best_iou=float(ious.max()))
    for t in (.5,.75,.9):
        tag = str(t)
        good = np.flatnonzero(ious>=t)
        result[f'score/{tag}'] = float(scores[good[0]]) if len(good) else 0.
        result[f'rank/{tag}'] = int(good[0])+1 if len(good) else 301
        for s in (.2,.5): result[f'hit/{tag}/{s}'] = bool(((ious>=t)&(scores>=s)).any())
    return result


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'conclusion.json').exists(): raise RuntimeError('Completed audit exists')
    logs, sources = {}, {}
    for name, run in RUNS.items():
        path = ROOT/f'outputs/{run}/seed0/log.txt'
        log = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        assert [r['epoch'] for r in log] == list(range(20))
        logs[name] = np.array([r['test_coco_eval_bbox'] for r in log])
        sources[f'log/{name}'] = dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    output = dict(status='complete_cached_audit_no_detector_updates', phases={}, same_epoch=[],
        best={}, inference_intervention={}, cohorts={}, sources=sources)
    for name, values in logs.items():
        e = int(values[:,0].argmax()); output['best'][name] = dict(epoch=e,metrics=values[e].tolist())
    for start,end in ((0,10),(10,14),(14,20),(15,20)):
        means={k:v[start:end].mean(0) for k,v in logs.items()}
        single=means['CS']-means['C']; joint=means['CMS']-means['CM']
        output['phases'][f'{start}-{end-1}']=dict(means={k:v.tolist() for k,v in means.items()},
            sam_on_C_delta_points=(single*100).tolist(),sam_on_CM_delta_points=(joint*100).tolist(),
            descriptive_interaction_points=((joint-single)*100).tolist())
    for e in range(20):
        a=(logs['CS'][e]-logs['C'][e])*100; b=(logs['CMS'][e]-logs['CM'][e])*100
        output['same_epoch'].append(dict(epoch=e,sam_on_C=a.tolist(),sam_on_CM=b.tolist(),interaction=(b-a).tolist()))
    caches={}
    for name,rel in CACHE_PATHS.items():
        path=ROOT/rel
        sources[f'cache/{name}']=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if path.suffix=='.npz':
            with np.load(path) as d:
                image_ids,boxes,scores=d['image_ids'],d['boxes'],d['scores']
                caches[name]={int(i):{'boxes':boxes[j].copy(),'scores':scores[j].copy()}
                              for j,i in enumerate(image_ids)}
        else:
            d=torch.load(path,map_location='cpu',weights_only=False)
            for mode,suffix in (('1.0',''),('0.0','_off')):
                caches[name+suffix]={int(i):{k:v.numpy() for k,v in r[mode].items() if k in ('boxes','scores')} for i,r in d.items()}
    ids=set(caches['C']); assert len(ids)==1820 and all(set(c)==ids for c in caches.values())
    ann_path=ROOT/'data/antiuav6k_common/annotations/instances_visible_common_test.json'
    sources['annotations']=dict(path=str(ann_path),sha256=hashlib.sha256(ann_path.read_bytes()).hexdigest())
    ann=json.loads(ann_path.read_text()); annotations={}
    for a in ann['annotations']:
        if not a.get('iscrowd',0):
            assert a['image_id'] not in annotations
            annotations[a['image_id']]=a
    images={r['id']:r for r in ann['images']}; assert ids==set(images)
    rows,empty=[],[]
    for image_id in sorted(ids):
        if image_id not in annotations:
            empty.append(dict(image_id=image_id,scores={k:float(v[image_id]['scores'].max()) for k,v in caches.items()})); continue
        a=annotations[image_id];x,y,w,h=a['bbox'];gt=np.array([x,y,x+w,y+h])
        rows.append(dict(image_id=image_id,scale='small' if a['area']<1024 else 'medium',file_name=images[image_id]['file_name'],
            models={k:stats(v[image_id]['boxes'],v[image_id]['scores'],gt) for k,v in caches.items()}))
    assert len(rows)==1691 and len(empty)==129
    small=[r for r in rows if r['scale']=='small']; assert len(small)==545
    for name,rel in {'CM':'reports/113_sgc0_half_control/final/diagnostics/epoch13/summary.json',
                     'CMS':'reports/112_sgc2_rgbt_half/final/diagnostics/epoch8/summary.json'}.items():
        summary=json.loads((ROOT/rel).read_text())
        assert max(abs(np.array(summary['metrics']['1.0'])-output['best'][name]['metrics']))<1e-10
        on=np.array(summary['metrics']['1.0']);off=np.array(summary['metrics']['0.0'])
        output['inference_intervention'][name]=dict(on=on.tolist(),off=off.tolist(),M_delta_points=((on-off)*100).tolist())
    intervention=output['inference_intervention']
    off_delta=(np.array(intervention['CMS']['off'])-np.array(intervention['CM']['off']))*100
    on_delta=(np.array(intervention['CMS']['on'])-np.array(intervention['CM']['on']))*100
    intervention['SAM_delta_off_points']=off_delta.tolist();intervention['SAM_delta_on_points']=on_delta.tolist()
    intervention['SAM_delta_change_when_M_on_points']=(on_delta-off_delta).tolist()
    for t in (.5,.75,.9):
        for s in (.2,.5):
            key=f'hit/{t}/{s}'; recovered=[];lost=[];table={};transitions={}
            for r in small:
                c,cs,cm,cms=[r['models'][k][key] for k in RUNS]
                bit=''.join(str(int(v)) for v in (c,cs,cm,cms));table[bit]=table.get(bit,0)+1
                if not c and cs:
                    label=('CM_already_hits' if cm else 'CM_misses')+('/CMS_retains' if cms else '/CMS_misses')
                    transitions[label]=transitions.get(label,0)+1;recovered.append(r['image_id'])
                if c and not cs:lost.append(r['image_id'])
            output['cohorts'][key]=dict(n=545,table_order=list(RUNS),bit_pattern_counts=table,
                CS_recovered_vs_C=len(recovered),CS_lost_vs_C=len(lost),recovered_ids=recovered,lost_ids=lost,
                recovery_destination=transitions,
                hit_counts={k:sum(r['models'][k][key] for r in small) for k in caches})
    benefits=[r for r in small if r['models']['CS']['top1_iou']-r['models']['C']['top1_iou']>.01]
    output['geometry_beneficiaries']=dict(n=len(benefits),criterion='CS-C top1 IoU >0.01, no score threshold; not AP benefit',
        CS_minus_C_mean=float(np.mean([r['models']['CS']['top1_iou']-r['models']['C']['top1_iou'] for r in benefits])),
        CMS_minus_CM_mean=float(np.mean([r['models']['CMS']['top1_iou']-r['models']['CM']['top1_iou'] for r in benefits])),
        joint_improved=sum(r['models']['CMS']['top1_iou']-r['models']['CM']['top1_iou']>.01 for r in benefits),
        joint_worsened=sum(r['models']['CMS']['top1_iou']-r['models']['CM']['top1_iou']<-.01 for r in benefits))
    output['empty_background']=dict(n=129,thresholds={str(t):{k:sum(r['scores'][k]>=t for r in empty) for k in caches} for t in (.2,.5)})
    output['caveats']=['four independently trained best EMA at epochs17/17/13/8; cache cohorts exploratory',
        'same-epoch log comparisons available, but per-image same-epoch predictions not all cached',
        'selected beneficiaries are not additive AP decomposition; score thresholds affect hit cohorts',
        'same-weight M off is not independently trained RGB-only; cannot isolate training gradient conflict',
        'single seed, epochs correlated; original test is development validation, not untouched test']
    save('targets.json',rows);save('conclusion.json',output)
    print(json.dumps(dict(phase14_19=output['phases']['14-19'],intervention=intervention,
        cohorts={k:v for k,v in output['cohorts'].items() if k in ('hit/0.75/0.2','hit/0.75/0.5')},
        geometry_beneficiaries=output['geometry_beneficiaries']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
