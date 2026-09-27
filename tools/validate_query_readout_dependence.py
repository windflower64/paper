"""Fixed best weights, RGB S4 readout interventions, full 1820-image development set."""
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
SNAP=ROOT/'reports/137_query_readout_trial/frozen_repo'
sys.path.insert(0,str(SNAP))
from src.core import YAMLConfig
from src.solver import TASKS
from src.solver.det_engine import evaluate
from tools.query_readout_runtime import install


def main():
    out=ROOT/'reports/138_query_readout_dependence';out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    results={}
    for arm in ('sam','box'):
        cfg=YAMLConfig(str(SNAP/'experiments/phase_s/sgc_scale_none_b16a2_20e.yml'),output_dir=str(out/'runtime'))
        cfg.yaml_cfg['HGNetv2']['pretrained']=False
        model=install(cfg.model)
        checkpoint=ROOT/f'outputs/QUERY_READ_{arm.upper()}_B16A2_10E_TESTDEV/seed0/best_stg1.pth'
        model.load_state_dict(torch.load(checkpoint,map_location='cpu')['ema']['module'],strict=True)
        solver=TASKS[cfg.yaml_cfg['task']](cfg);solver.eval()
        assert len(solver.val_dataloader.dataset)==1820 and solver.val_dataloader.dataset.sam_mask_root is None
        wrapper=solver.model.decoder.decoder.layers[0].cross_attn
        results[arm]={}
        for mode in ('disabled','shift'):
            wrapper.gain=0. if mode=='disabled' else .1
            wrapper.intervention='normal' if mode=='disabled' else 'shift'
            metrics,_=evaluate(solver.model,solver.criterion,solver.postprocessor,solver.val_dataloader,
                solver.evaluator,solver.device,epoch=-1,use_wandb=False)
            results[arm][mode]=metrics['coco_eval_bbox']
            (out/f'{arm}_{mode}.json').write_text(json.dumps({'arm':arm,'mode':mode,'metrics':metrics['coco_eval_bbox']},indent=2),encoding='utf-8')
            print('INTERVENTION_RESULT',arm,mode,metrics['coco_eval_bbox'],flush=True)
        del model,solver,cfg,wrapper
        torch.cuda.empty_cache()
    (out/'summary.json').write_text(json.dumps({'status':'complete','results':results,
        'scope':'same-checkpoint inference dependence; no retraining, no IR intervention; not training-causal attribution'},indent=2),encoding='utf-8')

if __name__=='__main__':main()
