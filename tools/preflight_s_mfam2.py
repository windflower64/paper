"""Real-batch preflight for both support/shape experiment arms."""
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.core import YAMLConfig
from src.solver import BaseSolver
from tools.preflight_q_rank1 import move_targets


def main():
    arm=sys.argv[1].upper()
    assert arm in ('SAM','BOX')
    torch.set_num_threads(8);torch.manual_seed(0)
    cfg=YAMLConfig(str(ROOT/f'experiments/phase_s/s_mfam2_c_{arm.lower()}_b8a4_20e_testdev_local.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    model=cfg.model
    shim=BaseSolver.__new__(BaseSolver);shim.model=model
    shim.load_tuning_state(str(ROOT.parent/'weights/m_sd2_joint_coco_thermal_identity_init.pth'))
    digest=hashlib.sha256()
    for key,value in model.state_dict().items():
        digest.update(key.encode());digest.update(value.detach().cpu().numpy().tobytes())
    initial_sha=digest.hexdigest()
    model=model.cuda();criterion=cfg.criterion.cuda();optimizer=cfg.optimizer
    loader=cfg.train_dataloader
    assert cfg.val_dataloader.dataset.sam_mask_root is None
    covered={id(p) for group in optimizer.param_groups for p in group['params']}
    assert all(id(p) in covered for p in model.mfam.parameters())
    samples,targets=next(iter(loader))
    assert tuple(samples.shape)==(8,3,512,640)
    samples=samples.cuda()
    model.eval()
    with torch.no_grad():
        model.mfam.intervention='disabled';disabled=model(samples)
        module=model.mfam;model.mfam=None
        indices=model.backbone.return_idx;model.backbone.return_idx=[2,3]
        baseline=model(samples)
        model.mfam=module;model.backbone.return_idx=indices
        error=max((disabled[k]-baseline[k]).abs().max().item() for k in ('pred_logits','pred_boxes'))
        assert error==0
        del disabled,baseline
        model.mfam.intervention='learned';normal=model(samples)
        errors={}
        for mode in ('support_constant','shape_constant'):
            model.mfam.intervention=mode;changed=model(samples)
            errors[mode]=(normal['pred_logits']-changed['pred_logits']).abs().max().item()
            assert errors[mode]>0
        del normal,changed
    model.mfam.intervention='learned';model.train()
    before=model.mfam.output_proj.weight.detach().clone()
    scaler=torch.cuda.amp.GradScaler(init_scale=128)
    optimizer.zero_grad(set_to_none=True)
    rows=[]
    torch.cuda.reset_peak_memory_stats()
    for step,(samples,targets) in enumerate(loader):
        if step==4:break
        samples=samples.cuda();targets=move_targets(targets,'cuda')
        with torch.autocast('cuda',dtype=torch.float16):outputs=model(samples,targets)
        losses=criterion(outputs,targets,epoch=0,step=step,global_step=step,epoch_step=len(loader))
        assert {'loss_mfam_support','loss_mfam_shape'}.issubset(losses)
        assert all(torch.isfinite(v) for v in losses.values())
        total=sum(losses.values())
        if step==0:
            detector=sum(v for k,v in losses.items() if not k.startswith('loss_mfam_'))
            grad=torch.autograd.grad(detector,model.mfam.mask_head[-1].weight,retain_graph=True)[0]
            detection_grad=[grad[i].float().norm().item() for i in (0,1)]
            assert min(detection_grad)>0
        scaler.scale(total/4).backward()
        rows.append({k:v.item() for k,v in losses.items() if k.startswith('loss_mfam_')})
        print(arm,'STEP',step,rows[-1],flush=True)
        del outputs,losses,total
    scaler.unscale_(optimizer)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(model.parameters(),.1)
    scaler.step(optimizer);scaler.update()
    update=(model.mfam.output_proj.weight-before).abs().max().item();assert update>0
    report=dict(status='PASS',arm=arm,physical_batch=8,gradient_accumulation_steps=4,
                epochs=20,seed=0,initial_model_sha256=initial_sha,baseline_disabled_error=error,
                intervention_errors=errors,detector_head_gradients=detection_grad,
                module_parameters=sum(p.numel() for p in model.mfam.parameters()),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,update=update,steps=rows)
    out=ROOT.parent/'reports/99_s_mfam2';out.mkdir(parents=True,exist_ok=True)
    (out/f'preflight_{arm}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
