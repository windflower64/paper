"""Real AMP updates and inference invariance; benchmark batch16 versus batch32."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from tools.run_shape_feature_transfer import prepare,ROOT
from tools.shape_feature_interface import ARMS,common_weights,schedule


def digest(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        h.update(name.encode()); h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def move(targets):
    return [{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]


def check(config,arm,batch,updates):
    cfg=prepare(config,arm,ROOT/'reports/153_shape_feature_transfer/preflight_runtime',batch,log=False)
    model=cfg.model
    initial=digest(model)
    detector_initial=digest(model.detector)
    loader=cfg.train_dataloader; loader.set_epoch(0)
    assert loader.collate_fn.scales is None
    samples,targets=next(iter(loader))
    assert samples.shape==(batch,6,512,640)
    ids=[int(t['image_id'][0]) for t in targets]
    assert len(cfg.val_dataloader.dataset)==1820
    assert cfg.val_dataloader.dataset.sam_mask_root is None
    samples=samples.cuda();targets=move(targets)
    model.cuda();criterion=cfg.criterion.cuda()
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        a=model(samples[:2]); b=model.detector(samples[:2])
    errors={k:float((a[k]-b[k]).abs().max()) for k in ('pred_boxes','pred_logits')}
    assert max(errors.values())==0
    model.train();model.set_training_epoch(0)
    optimizer=cfg.optimizer
    scaler=cfg.scaler
    torch.cuda.reset_peak_memory_stats()
    times=[];losses=[];aux_values=[];actual=[]
    optimizer.register_step_post_hook(lambda *args:actual.append(True))
    probe=next(model.detector.backbone.stages[2].parameters())
    probe_before=probe.detach().flatten()[:128].clone()
    direct=None
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.float16):
            output=model(samples,targets)
        aux=output['feature_kd_loss']
        if step==0 and arm!='none':
            m_parameter=next(p for p in model.detector.sd2_conditioner.parameters() if p.requires_grad)
            rgb_grad,m_grad=torch.autograd.grad(aux,(probe,m_parameter),retain_graph=True,allow_unused=True)
            assert rgb_grad is not None and torch.isfinite(rgb_grad).all() and rgb_grad.abs().max()>0
            assert m_grad is None or float(m_grad.abs().max())==0
            direct={'rgb_gradient_max':float(rgb_grad.abs().max()),'m_gradient_max':0.}
        with torch.autocast('cuda',enabled=False):
            values=criterion(output,targets,epoch=0,step=step,global_step=step,epoch_step=len(loader))
            loss=sum(values.values())
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        assert all(p.grad is None for p in model.detector.thermal_backbone.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_max_norm)
        scaler.step(optimizer);scaler.update()
        torch.cuda.synchronize();times.append(time.perf_counter()-start)
        losses.append(float(loss.detach()));aux_values.append(float(aux.detach()))
        print('REAL_UPDATE',arm,batch,step,'seconds',times[-1],'loss',losses[-1],flush=True)
    assert len(actual)==updates
    changed=float((probe.detach().flatten()[:128]-probe_before).abs().max())
    assert changed>0
    choices=[common_weights(t) for t in targets]
    valid=[c for c in choices if c is not None]
    assert valid
    mask_difference=sum(float((c[0]['sam']-c[0]['filled']).abs().sum()) for c in valid)
    assert mask_difference>0
    model.set_training_epoch(10)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        retired=model(samples,targets)
    assert float(retired['feature_kd_loss'])==0
    clone=copy.deepcopy(model).eval()
    hook=list(clone.detector.backbone._forward_hooks.values())[-1]
    assert hook.__self__ is clone and clone.captured is None
    result=dict(arm=arm,batch=batch,initial_sha256=initial,detector_initial_sha256=detector_initial,
        input_ids=ids,inference_errors=errors,actual_updates=len(actual),probe_weight_change=changed,
        loss=losses,kd_loss=aux_values,direct_kd_gradient=direct,valid=len(valid),
        sam_filled_selection_l1_sum=mask_difference,seconds=times,
        # Last update avoids startup allocations; not full epoch I/O performance.
        last_update_samples_per_second=batch/times[-1],
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        phase10_zero=True,ema_hook_owned_by_clone=True)
    del clone,model,criterion,optimizer,cfg,samples,targets,output,retired,loss,aux,values
    torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise RuntimeError('Existing preflight; use a new attempt path')
    teacher=ROOT/'reports/153_shape_feature_transfer/teacher'
    assert json.loads((teacher/'status.json').read_text())['status']=='complete'
    historical=json.loads((ROOT/'reports/112_sgc2_rgbt_half/preflight.json').read_text())
    assert hashlib.sha256((ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth').read_bytes()).hexdigest()==historical['checkpoint_sha256']
    benchmarks=[]
    # Try both sizes; only a genuine OOM authorizes a size fallback.
    for batch in (16,32):
        try:benchmarks.append(check(args.config,'sam',batch,2))
        except torch.cuda.OutOfMemoryError as error:
            benchmarks.append(dict(batch=batch,status='OOM',error=str(error)))
            torch.cuda.empty_cache()
    passing=[b for b in benchmarks if b.get('actual_updates')==2]
    assert passing,'Neither batch16 nor batch32 completed real updates'
    best=max(passing,key=lambda b:b['last_update_samples_per_second'])
    batch=best['batch']
    runs=[check(args.config,arm,batch,1) for arm in ARMS]
    assert len({r['initial_sha256'] for r in runs})==1
    assert len({tuple(r['input_ids']) for r in runs})==1
    assert len({r['valid'] for r in runs})==1
    result=dict(status='PASS',selected_batch=batch,accumulation=32//batch,effective_batch=32,
                schedule=[schedule(i) for i in range(20)],benchmarks=benchmarks,runs=runs,
                note='Temporary real updates only; official training reloads same initialization')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PREFLIGHT_PASS','selected batch',batch,flush=True)


if __name__=='__main__':main()
