"""Can box-prompted SAM3 distinguish true detector candidates from empty-image FPs?"""
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO=Path(__file__).resolve().parents[1]; ROOT=REPO.parent
SOURCE=ROOT/'reports/102_sam3_c_prompt'; CROOT=ROOT/'reports/101_sam3_c_comparison/c_predictions'
OUT=ROOT/'reports/108_sam3_hard_negative_audit'
sys.path.insert(0,str(REPO))
from tools.sam3_teacher_pilot import iou


def raster_box(box,height=512,width=640):
    x0,y0,x1,y1=box
    out=np.zeros((height,width),dtype=bool)
    out[max(0,int(np.floor(y0))):min(height,int(np.ceil(y1))),
        max(0,int(np.floor(x0))):min(width,int(np.ceil(x1)))]=True
    return out


def features(mask,candidate,predicted,teacher_score,count,c_score):
    candidate_mask=raster_box(candidate); predicted_mask=raster_box(predicted)
    area=max(int(mask.sum()),1); candidate_area=max(int(candidate_mask.sum()),1)
    intersection=int((mask&candidate_mask).sum()); union=int((mask|candidate_mask).sum())
    horizontal=np.count_nonzero(mask[:,1:]!=mask[:,:-1]); vertical=np.count_nonzero(mask[1:]!=mask[:-1])
    return dict(c_score=c_score,teacher_score=teacher_score,teacher_count=count,
        teacher_box_candidate_iou=iou(predicted,candidate),mask_candidate_iou=intersection/max(union,1),
        mask_inside_candidate=intersection/area,mask_area_ratio_candidate=area/candidate_area,
        mask_fill_predicted_box=int((mask&predicted_mask).sum())/max(int(predicted_mask.sum()),1),
        mask_perimeter_normalized=(horizontal+vertical)/np.sqrt(area))


def main():
    manifest=json.loads((SOURCE/'manifest.json').read_text()); rows=[]; excluded=[]
    for image in manifest['images']:
        image_id=image['id']; prediction=json.loads((SOURCE/'predictions'/f'{image_id:06d}_0.json').read_text())
        c=json.loads((CROOT/f'{image_id:06d}.json').read_text())
        assert prediction['boxes'] and prediction['scores'] and c['boxes'] and c['scores']
        candidate=prediction['candidate_cxcywh_normalized']; cx,cy,w,h=candidate
        candidate=[(cx-w/2)*640,(cy-h/2)*512,(cx+w/2)*640,(cy+h/2)*512]
        assert max(abs(a-b) for a,b in zip(candidate,c['boxes'][0]))<.05
        if image['annotations']:
            x,y,w,h=image['annotations'][0]['bbox']; gt=[x,y,x+w,y+h]
            candidate_iou=iou(candidate,gt)
            if candidate_iou<.5:
                excluded.append(dict(image_id=image_id,group=image['group'],candidate_iou=candidate_iou)); continue
            label=1
        else:
            candidate_iou=None; label=0
        with np.load(SOURCE/'masks'/f'{image_id:06d}_0.npz') as archive:
            mask=archive['masks'][0].astype(bool)
        values=features(mask,candidate,prediction['boxes'][0],prediction['scores'][0],len(prediction['scores']),c['scores'][0])
        sequence=image['file_name'].rsplit('_',1)[0]
        rows.append(dict(image_id=image_id,file_name=image['file_name'],group=image['group'],label=label,
                         candidate_iou=candidate_iou,sequence=sequence,features=values))
    names=list(rows[0]['features']); X=np.array([[r['features'][n] for n in names] for r in rows]); y=np.array([r['label'] for r in rows]); groups=np.array([r['sequence'] for r in rows])
    assert np.isfinite(X).all() and set(y)=={0,1}
    folds=GroupKFold(n_splits=5); model_sets={'C分数':['c_score'],'仅SAM属性':[n for n in names if n!='c_score'],'C分数+SAM属性':names}
    results={}
    for title,used in model_sets.items():
        columns=[names.index(n) for n in used]; oof=np.zeros(len(y)); fold_rows=[]
        for fold,(train,test) in enumerate(folds.split(X,y,groups)):
            assert not set(groups[train])&set(groups[test])
            model=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000,random_state=0))
            model.fit(X[train][:,columns],y[train]); oof[test]=model.predict_proba(X[test][:,columns])[:,1]
            fold_rows.append(dict(fold=fold,images=len(test),positive=int(y[test].sum()),negative=int((1-y[test]).sum()),
                                  roc_auc=roc_auc_score(y[test],oof[test]),average_precision=average_precision_score(y[test],oof[test])))
        results[title]=dict(features=used,roc_auc=roc_auc_score(y,oof),average_precision=average_precision_score(y,oof),folds=fold_rows)
    univariate={}
    for index,name in enumerate(names):
        auc=roc_auc_score(y,X[:,index]); univariate[name]=dict(roc_auc=auc,best_orientation_auc=max(auc,1-auc))
    result=dict(protocol='fixed200 test-development audit; C top1 box prompts SAM3; sequence-grouped 5-fold OOF',
        included=len(rows),positive=int(y.sum()),negative=int((1-y).sum()),excluded_positive_candidates=excluded,
        warning='Diagnostic only. Positives require C top1 IoU>=0.5; not full detection AP or independent held-out evidence.',
        models=results,univariate=univariate,rows=rows)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:{'roc_auc':v['roc_auc'],'average_precision':v['average_precision']} for k,v in results.items()},indent=2,ensure_ascii=False))


if __name__=='__main__': main()
