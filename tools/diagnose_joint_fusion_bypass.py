"""Evaluate report159 trained RGB paths with the fusion branch bypassed."""
import json
import sys
from pathlib import Path
import torch

ROOT=Path('E:/two_paper')
REPORT=ROOT/'reports/160_frozen_backbone_alignment'
SNAP=ROOT/'reports/159_local_alignment_joint_stable16/frozen_repo'
sys.path.insert(0,str(SNAP));sys.path.insert(0,str(SNAP/'tools'))
from run_local_alignment_joint import prepare,write
from run_mbudet_alignment import digest
from src.solver import TASKS
from src.solver.det_engine import evaluate


def main():
    run=ROOT/'outputs/M_LOCAL_ALIGN_JOINT_FUSION_B16A2_30E_TESTDEV/seed0'
    rows=[json.loads(s) for s in (run/'log.txt').read_text().splitlines() if s.strip()]
    result={}
    for label,file,row in [('best','best_stg1.pth',max(rows,key=lambda r:r['test_coco_eval_bbox'][0])),
                            ('last','last.pth',rows[-1])]:
        output=REPORT/f'bypass_{label}.json'
        assert not output.exists()
        cfg=prepare('fusion',REPORT/f'diagnostic_{label}',16)
        cfg.resume=str(run/file)
        solver=TASKS[cfg.yaml_cfg['task']](cfg);solver.eval()
        model=solver.ema.module
        before=digest(model)
        model.arm='rgb'
        metrics,_=evaluate(model,solver.criterion,solver.postprocessor,solver.val_dataloader,
                            solver.evaluator,solver.device,epoch=-1,use_wandb=False)
        assert digest(model)==before
        result[label]=dict(status='PASS',checkpoint=str(run/file),epoch=row['epoch'],images=1820,
            trained_weights_unchanged=True,uses_gt_in_forward=False,normal_metrics=row['test_coco_eval_bbox'],
            bypass_metrics=metrics['coco_eval_bbox'],
            ap_delta_points=(metrics['coco_eval_bbox'][0]-row['test_coco_eval_bbox'][0])*100)
        write(output,result[label]); print('BYPASS_RESULT',json.dumps(result[label]),flush=True)
        del model,solver,cfg;torch.cuda.empty_cache()
    write(REPORT/'bypass_summary.json',dict(status='PASS',results=result,
        limitation='同权重关闭残差改变输入分布，仅作机制诊断，不能严格分离RGB遗忘和与融合共同适配。'))


if __name__=='__main__':main()
