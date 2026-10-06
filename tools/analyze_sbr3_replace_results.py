"""Audit both complete SBR3 arms and independently re-evaluate best EMA."""
import json
from pathlib import Path
import torch
import train_sbr3_replace as task
from preflight_sqer1 import seed_all,sha256_file


def main():
    seed_all(0); torch.set_num_threads(4)
    directory=task.engine.ROOT/'reports/SBR3_REPLACE_ANALYSIS_20261004'
    directory.mkdir(exist_ok=False)
    cfg,source,model,_=task.setup(16)
    base=json.loads((task.RUN/'preflight.json').read_text(encoding='utf-8'))['baseline']
    report={'baseline':base,'arms':{}}
    protocols=[]; series=[]
    for arm in ['R1_GT_ONLY','R2_SAM']:
        root=task.RUN/arm
        rows=[json.loads(x) for x in (root/'log.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        assert [r['epoch'] for r in rows]==list(range(30))
        assert all(r['validation']['images']==1820 for r in rows)
        best=max(rows,key=lambda r:r['validation']['coco_eval_bbox'][0])
        saved=torch.load(root/'best.pth',map_location='cpu')
        assert saved['epoch']==best['epoch'] and saved['last_row']==best
        last=torch.load(root/'last.pth',map_location='cpu')
        assert last['epoch']==29 and last['last_row']==rows[-1]
        protocols.append(saved['protocol'])
        model.load_state_dict(saved['ema'],strict=True)
        reproduced=task.evaluate(cfg,source,model)
        error=max(abs(x-y) for x,y in zip(reproduced['coco_eval_bbox'],best['validation']['coco_eval_bbox']))
        assert error<.0002,error
        aps=[r['validation']['coco_eval_bbox'][0] for r in rows]; series.append(aps)
        report['arms'][arm]={'best':best,'last':rows[-1],'reproduced':reproduced,'max_abs_reproduction_error':error,
                            'best_checkpoint_sha256':sha256_file(root/'best.pth'),'protocol':saved['protocol'],
                            'epochs_above_baseline':sum(a>base['coco_eval_bbox'][0] for a in aps),
                            'mean_ap_percent':sum(aps)/30*100,
                            'curves':[{'epoch':r['epoch'],'ap':r['validation']['coco_eval_bbox'][0],
                                       'iou':r['validation']['paired_base_assignment']['mean_refined_iou'],
                                       'shape_loss':r['train']['shape_loss']} for r in rows]}
        task.engine.write_json(directory/f'{arm}_best_reevaluation.json',report['arms'][arm])
    assert {k:v for k,v in protocols[0].items() if k not in ['arm','shape_weight']}=={k:v for k,v in protocols[1].items() if k not in ['arm','shape_weight']}
    report['sam_wins_same_epoch']=sum(b>a for a,b in zip(*series))
    report['sam_mean_delta_pp']=sum(b-a for a,b in zip(*series))/30*100
    report['status']='PASS'
    task.engine.write_json(directory/'audit.json',report)
    print(json.dumps({'status':'PASS','sam_wins_same_epoch':report['sam_wins_same_epoch'],'sam_mean_delta_pp':report['sam_mean_delta_pp'],
                      'arms':{a:{k:v for k,v in x.items() if k in ['epochs_above_baseline','mean_ap_percent','max_abs_reproduction_error']} for a,x in report['arms'].items()}}),flush=True)
    source.close()


if __name__=='__main__': main()


