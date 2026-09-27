"""Independent full best-EMA evaluation using a frozen source snapshot."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.snapshot))
    from src.core import YAMLConfig
    from src.solver import TASKS
    from src.solver.det_engine import evaluate
    torch.set_num_threads(4)
    if args.output.exists():raise RuntimeError('Refusing to overwrite evaluation')
    checkpoint=args.run/'best_stg1.pth'
    cfg=YAMLConfig(str(args.config),resume=str(checkpoint),output_dir=str(args.output.parent/'runtime'))
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    solver=TASKS[cfg.yaml_cfg['task']](cfg);solver.eval()
    assert len(solver.val_dataloader.dataset)==1820
    assert solver.val_dataloader.dataset.sam_mask_root is None
    assert solver.ema is not None
    metrics,_=evaluate(solver.ema.module,solver.criterion,solver.postprocessor,
        solver.val_dataloader,solver.evaluator,solver.device,epoch=-1,use_wandb=False)
    rows=[json.loads(line) for line in (args.run/'log.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
    assert [r['epoch'] for r in rows]==list(range(20))
    best=max(rows,key=lambda r:r['test_coco_eval_bbox'][0])
    values=metrics['coco_eval_bbox']
    assert len(values)==12
    error=max(abs(a-b) for a,b in zip(values,best['test_coco_eval_bbox']))
    result={'status':'PASS' if error<.0002 else 'MISMATCH','images':1820,
        'best_epoch':best['epoch'],'coco_eval_bbox':values,'logged_metrics':best['test_coco_eval_bbox'],
        'max_abs_error':error,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'weight_source':'ema','evaluation_split':'original test, used as development set'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert result['status']=='PASS',result
    print('INDEPENDENT_EVALUATION_PASS',flush=True)


if __name__=='__main__':main()
