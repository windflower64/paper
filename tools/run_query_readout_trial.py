"""Native solver entry with isolated readout adapter, tuning before key wrapping."""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
import torch
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from src.core import YAMLConfig
from src.solver import BaseSolver,TASKS
from src.solver.det_engine import evaluate
from tools.preflight_sgc_scale_interface import seed,digest
from tools.query_readout_runtime import install,GuidanceCriterion


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arm',choices=('none','box','sam'),required=True)
    p.add_argument('--mode',choices=('preflight','train','eval'),required=True)
    args=p.parse_args();root=args.root
    report=root/'reports/137_query_readout_trial';report.mkdir(parents=True,exist_ok=True)
    run=root/f'outputs/QUERY_READ_{args.arm.upper()}_B16A2_10E_TESTDEV/seed0'
    torch.set_num_threads(4);seed(0)
    cfg=YAMLConfig(str(REPO/'experiments/phase_s/sgc_scale_none_b16a2_20e.yml'),
        output_dir=str(run),epochs=10,print_freq=20,use_amp=True)
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    model=cfg.model
    shim=BaseSolver.__new__(BaseSolver);shim.model=model
    shim.load_tuning_state(str(root/'weights/m_sd2_joint_coco_thermal_identity_init.pth'))
    seed(0);install(model)
    initial=digest(model)
    base=cfg.criterion
    cfg._criterion=GuidanceCriterion(base,args.arm,None if args.mode!='train' else str(run/'guidance_batches.jsonl'))
    if args.mode=='preflight':
        from tools.preflight_q_rank1 import move_targets
        model.cuda();ema=cfg.ema.to('cuda');criterion=cfg.criterion.cuda();optimizer=cfg.optimizer;scaler=cfg.scaler
        samples,targets=next(iter(cfg.train_dataloader));samples=samples.cuda();targets=move_targets(targets,'cuda')
        assert samples.shape==(16,6,512,640)
        assert cfg.yaml_cfg['gradient_accumulation_steps']==2
        assert len(cfg.val_dataloader.dataset)==1820 and cfg.val_dataloader.dataset.sam_mask_root is None
        model.train();before=model.decoder.decoder.layers[0].cross_attn.reader.offsets.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        for micro in range(2):
            with torch.autocast('cuda'):pred=model(samples,targets)
            losses=criterion(pred,targets,epoch=0,step=micro,global_step=micro,epoch_step=200)
            total=sum(losses.values());assert torch.isfinite(total)
            scaler.scale(total/2).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_max_norm)
        scale=scaler.get_scale();scaler.step(optimizer);scaler.update();assert scaler.get_scale()>=scale
        assert not torch.equal(before,model.decoder.decoder.layers[0].cross_attn.reader.offsets.weight)
        ema.update(model);ema.module.eval()
        with torch.no_grad():expected=ema.module(samples[:2])
        buffer=io.BytesIO();torch.save(ema.state_dict(),buffer);buffer.seek(0)
        ema.load_state_dict(torch.load(buffer,map_location='cuda'))
        with torch.no_grad():actual=ema.module(samples[:2])
        assert all(torch.equal(expected[k],actual[k]) for k in ('pred_boxes','pred_logits'))
        assert model.decoder.decoder.layers[0].cross_attn.context=={}
        assert ema.module.decoder.decoder.layers[0].cross_attn.context=={}
        result={'status':'PASS','arm':args.arm,'initial_sha256':initial,'batch':16,'accumulation':2,
                'native_optimizer_update':True,'ema_and_strict_roundtrip':True}
        dest=report/f'preflight_{args.arm}.json'
        with dest.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
        print(json.dumps(result),flush=True);return
    run.mkdir(parents=True,exist_ok=True)
    if args.mode=='train':
        if (run/'log.txt').exists() or (run/'guidance_batches.jsonl').exists():raise RuntimeError('Existing training output')
        with (run/'protocol.json').open('x',encoding='utf-8') as f:
            json.dump({'arm':args.arm,'initial_sha256':initial,'epochs':10,'batch':16,'accumulation':2,
                       'resolved_config':cfg.yaml_cfg,'readout_gain':.1,'geometry_weight':1,
                       'sampling':'first layer S4, final matching, common reachable guidance'},f,indent=2,default=str)
        TASKS[cfg.yaml_cfg['task']](cfg).fit()
    else:
        checkpoint=run/'best_stg1.pth'
        state=torch.load(checkpoint,map_location='cpu')
        model.load_state_dict(state['ema']['module'],strict=True)
        solver=TASKS[cfg.yaml_cfg['task']](cfg);solver.eval()
        metrics,_=evaluate(solver.model,solver.criterion,solver.postprocessor,solver.val_dataloader,
                           solver.evaluator,solver.device,epoch=-1,use_wandb=False)
        rows=[json.loads(s) for s in (run/'log.txt').read_text().splitlines() if s.strip()]
        assert [r['epoch'] for r in rows]==list(range(10))
        best=max(rows,key=lambda r:r['test_coco_eval_bbox'][0]);values=metrics['coco_eval_bbox']
        error=max(abs(a-b) for a,b in zip(values,best['test_coco_eval_bbox']))
        result={'status':'PASS' if error<.0002 else 'MISMATCH','best_epoch':best['epoch'],
                'metrics':values,'max_abs_error':error,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        with (report/f'eval_{args.arm}.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
        assert error<.0002

if __name__=='__main__':main()
