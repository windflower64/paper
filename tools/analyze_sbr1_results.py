"""Read-only 30-epoch SBR1 evidence audit and durable result export."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'outputs/SBR1_20261003'
NAMES=['AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','ARS','ARM','ARL']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    baseline=json.loads((RUN/'preflight.json').read_text(encoding='utf-8'))['baseline']
    result={'baseline':baseline,'arms':{}}
    series={}
    for arm in ['R1_GT_ONLY','R2_SAM']:
        rows=[json.loads(x) for x in (RUN/arm/'log.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        assert [r['epoch'] for r in rows]==list(range(30))
        assert all(r['validation']['images']==1820 for r in rows)
        assert all(r['validation']['paired_base_assignment']['targets']==1691 for r in rows)
        best=max(rows,key=lambda x:x['validation']['coco_eval_bbox'][0])
        saved=torch.load(RUN/arm/'best.pth',map_location='cpu')
        assert saved['epoch']==best['epoch']
        assert saved['best_ap']==best['validation']['coco_eval_bbox'][0]
        assert saved['last_row']==best
        last=torch.load(RUN/arm/'last.pth',map_location='cpu')
        assert last['epoch']==29 and last['last_row']==rows[-1]
        changed={k:float((last['ema'][k]-saved['ema'][k]).float().norm()) for k in saved['ema']}
        aps=np.array([r['validation']['coco_eval_bbox'][0] for r in rows])
        series[arm]=aps
        result['arms'][arm]={
            'epochs':len(rows),'best':best,'last':rows[-1],
            'best_metrics_percent':dict(zip(NAMES,[v*100 if v>=0 else None for v in best['validation']['coco_eval_bbox']])),
            'best_minus_baseline_pp':dict(zip(NAMES,[100*(a-b) if a>=0 and b>=0 else None for a,b in zip(best['validation']['coco_eval_bbox'],baseline['coco_eval_bbox'])])),
            'mean_ap_percent':float(aps.mean()*100),'last5_mean_ap_percent':float(aps[-5:].mean()*100),
            'epochs_above_baseline':int((aps>baseline['coco_eval_bbox'][0]).sum()),
            'topk_coverage':rows[-1]['train']['selected_matched']/rows[-1]['train']['matched'],
            'teacher_coverage_selected':rows[-1]['train']['teacher_valid']/rows[-1]['train']['selected_matched'],
            'required_correction_outside_radius_fraction':rows[-1]['train']['outside_radius']/rows[-1]['train']['selected_matched'],
            'max_saturation_fraction':max(r['train']['saturation_fraction'] for r in rows),
            'best_sha256':sha(RUN/arm/'best.pth'),'last_sha256':sha(RUN/arm/'last.pth'),
            'best_checkpoint_matches_log':True,'last_checkpoint_matches_log':True,
            'protocol':saved['protocol'],'last_ema_parameter_change_norm_from_best':changed,
            'curves':[{'epoch':r['epoch'],'ap':r['validation']['coco_eval_bbox'][0],
                       'ap75':r['validation']['coco_eval_bbox'][2],'aps':r['validation']['coco_eval_bbox'][3],
                       'shape_kl':r['train']['shape_kl'],'l1':r['train']['l1'],'giou':r['train']['giou'],
                       'mean_refined_iou':r['validation']['paired_base_assignment']['mean_refined_iou']} for r in rows]}
    a=result['arms']['R1_GT_ONLY']['best']['validation']['coco_eval_bbox']
    b=result['arms']['R2_SAM']['best']['validation']['coco_eval_bbox']
    result['sam_minus_gt_best_pp']=dict(zip(NAMES,[100*(y-x) if x>=0 else None for x,y in zip(a,b)]))
    result['sam_same_epoch_wins']=int((series['R2_SAM']>series['R1_GT_ONLY']).sum())
    result['sam_same_epoch_mean_delta_pp']=float((series['R2_SAM']-series['R1_GT_ONLY']).mean()*100)
    p1=result['arms']['R1_GT_ONLY']['protocol']; p2=result['arms']['R2_SAM']['protocol']
    assert p1['init_sha']==p2['init_sha'] and p1['base_sha']==p2['base_sha']
    assert {k:v for k,v in p1.items() if k not in ['arm','shape_weight']}=={k:v for k,v in p2.items() if k not in ['arm','shape_weight']}
    output=ROOT/'reports/SBR1_ANALYSIS_20261004'
    output.mkdir(exist_ok=False)
    (output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='arms'},ensure_ascii=False,indent=2))
    for arm,values in result['arms'].items():
        print(arm,json.dumps({k:v for k,v in values.items() if k not in ['best','last','curves','last_ema_parameter_change_norm_from_best','protocol']},ensure_ascii=False))


if __name__=='__main__': main()
