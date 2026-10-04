"""SBR3 position-search pilot; new outputs, old checkpoints are never reused."""
import argparse
import json
from pathlib import Path
import time
import torch
import train_sbr1 as engine
from train_sbr2 import SourceV2
from preflight_sqer1 import seed_all,move_targets,sha256_file

RUN=engine.ROOT/'outputs/SBR3_20261004'


def setup(batch):
    cfg,old,unused,_=engine.setup(batch)
    base=old.model; old.close(); del unused
    from src.zoo.dfine.sam_side_search import SAMSideSearch
    model=SAMSideSearch(base.sqer).cuda()
    return cfg,SourceV2(base),model,model.loss


def readout(model,cache,out,hw):
    return model(cache['s4'],cache['s8'],cache['base_logits'],out['pred_boxes'],hw)


def evaluate(cfg,source,model):
    return engine.evaluate(cfg,source,model,readout_fn=readout)


def preflight(args):
    cfg,source,model,lossfn=setup(args.batch_size)
    RUN.mkdir(exist_ok=True,parents=True)
    init=RUN/'common_refiner_init.pth'
    if init.exists(): model.load_state_dict(torch.load(init,map_location='cuda'))
    else: engine.save_checkpoint(init,model.state_dict())
    original=engine.digest(source.model)
    samples,targets=next(iter(cfg.train_dataloader))
    samples=samples.cuda(); targets=move_targets(targets,torch.device('cuda'))
    out,cache=source(samples)
    matches=engine.matching(cfg.criterion.matcher.cuda(),out,targets)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        result=readout(model,cache,out,samples.shape[-2:])
        assert torch.equal(result['boxes'],out['pred_boxes'])
        assert result['selected']['scores'].shape==(len(samples),64,4,17)
    checks={}
    for name,weight in [('GT_ONLY',0.),('SAM',1.)]:
        model.load_state_dict(torch.load(init,map_location='cuda'))
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        rows=[]
        for step in range(4):
            seed_all(20000+step); opt.zero_grad(set_to_none=True)
            result=readout(model,cache,out,samples.shape[-2:])
            loss,stats=lossfn(result,out['pred_boxes'],targets,matches,weight)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            assert torch.isfinite(loss) and torch.isfinite(norm)
            grads={p:float(v.grad.norm()) if v.grad is not None else 0. for p,v in model.named_parameters()}
            opt.step(); rows.append({'loss':float(loss),'stats':stats,'gradient_norm':float(norm)})
        for prefix in ['shape_head','s4_projection','s8_projection','fusion','score']:
            assert sum(v for k,v in grads.items() if k.startswith(prefix))>0,prefix
        checks[name]={'steps':rows,'gradient_norms':grads}
    # Actual learned scoring must depend on predicted occupancy, not just
    # have an unused auxiliary shape head.
    with torch.no_grad():
        result=readout(model,cache,out,samples.shape[-2:])
        alternative=model.search(result['memory'],torch.zeros_like(result['shape_logits']),result['original'],samples.shape[-2:])
        effect=float((alternative['scores']-result['selected']['scores']).abs().max())
        assert effect>0,'Shape channel has no functional effect'
        chosen=torch.zeros_like(out['pred_boxes'][...,0],dtype=torch.bool).scatter(1,result['indices'],True)
        assert torch.equal(result['boxes'][~chosen],out['pred_boxes'][~chosen])
        valid=(out['pred_boxes'][...,2:]>0).all(-1)
        assert (result['boxes'][valid][...,2:]>0).all()
    assert engine.digest(source.model)==original
    assert not any(p.grad is not None for p in source.model.parameters())
    peak=torch.cuda.max_memory_allocated()/1024**3
    del loss,result,alternative,opt,samples,targets,out,cache
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(init,map_location='cuda'))
    baseline=evaluate(cfg,source,model)
    old=json.loads((engine.RUN/'preflight.json').read_text(encoding='utf-8'))['baseline']['coco_eval_bbox']
    error=max(abs(x-y) for x,y in zip(old,baseline['coco_eval_bbox']))
    assert error<.0002,error
    report={'status':'PASS','batch_size':args.batch_size,'precision':'float32','peak_memory_gib':peak,
            'parameters':sum(p.numel() for p in model.parameters()),'init_sha':sha256_file(init),
            'base_sha':engine.BASE_SHA,'baseline':baseline,'baseline_max_abs_error':error,
            'shape_intervention_max_score_change':effect,'checks':checks,'source_immutable':True}
    engine.write_json(RUN/'preflight.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}),flush=True)
    source.close()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--preflight',action='store_true')
    p.add_argument('--arm',choices=['R2_SAM','R1_GT_ONLY']); p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--epochs',type=int,default=30); args=p.parse_args()
    seed_all(0); torch.set_num_threads(4)
    if args.preflight: preflight(args)
    elif args.arm:
        files=[Path(__file__),Path(engine.__file__),engine.REPO/'tools/train_sbr2.py',engine.REPO/'src/zoo/dfine/sam_side_search.py']
        metadata={'module':'SBR3','precision':'float32','position_bins':17,'strip_points':16,
                  'epoch_random_seed_base':20000,'synthetic_candidates':'one exact GT plus two independently jittered boxes per target, side jitter +/-20%',
                  'synthetic_loss_weight':.5,'shape_loss':'local soft BCE + dice, weight 1 in SAM arm',
                  'local_encoder_trainable':True,'predicted_shape_in_forward':True,'teacher_in_forward':False,
                  'roi_size':None,'source_hashes':{str(f):sha256_file(f) for f in files}}
        engine.train(args,setup_fn=setup,readout_fn=readout,evaluate_fn=evaluate,run_dir=RUN,extra_protocol=metadata,sam_shape_weight=1.)
    else: p.error('Specify --preflight or --arm')


if __name__=='__main__':
    try: main()
    except Exception as exc:
        if RUN.exists(): engine.write_json(RUN/'status.json',{'state':'failed','error':repr(exc),'updated_at':time.time()})
        raise
