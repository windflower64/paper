"""Four-arm same-initialization real batch preflight, separate process per arm."""
import hashlib
import json
import sys
from pathlib import Path
import torch

REPO=Path(__file__).resolve().parents[1];ROOT=REPO.parent
sys.path.insert(0,str(REPO))
from src.core import YAMLConfig
from src.solver import BaseSolver
from tools.preflight_q_rank1 import move_targets


def main():
    arm=sys.argv[1].lower();assert arm in ['none','box','sam','edge']
    torch.set_num_threads(4);torch.manual_seed(0)
    cfg=YAMLConfig(str(REPO/f'experiments/phase_s/s_rc1_{arm}_b8a4_20e.yml'))
    model=cfg.model
    shim=BaseSolver.__new__(BaseSolver);shim.model=model
    shim.load_tuning_state(str(ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth'))
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    initial=digest.hexdigest()
    model=model.cuda();criterion=cfg.criterion.cuda();optimizer=cfg.optimizer
    covered={id(p) for g in optimizer.param_groups for p in g['params']}
    assert all(id(p) in covered for p in model.mfam.parameters())
    loader=cfg.train_dataloader
    assert cfg.val_dataloader.dataset.sam_mask_root is None
    model.train();optimizer.zero_grad(set_to_none=True)
    scaler=torch.cuda.amp.GradScaler(init_scale=128)
    before=model.mfam.output_proj.weight.detach().clone()
    rows=[];torch.cuda.reset_peak_memory_stats()
    for step,(samples,targets) in enumerate(loader):
        if step==4:break
        assert tuple(samples.shape)==(8,3,512,640)
        samples=samples.cuda();targets=move_targets(targets,'cuda')
        with torch.autocast('cuda',dtype=torch.float16): output=model(samples,targets)
        losses=criterion(output,targets,epoch=0,step=step,global_step=step,epoch_step=len(loader))
        assert all(torch.isfinite(v) for v in losses.values())
        aux={k:v for k,v in losses.items() if k.startswith('loss_mfam')}
        assert len(aux)==1
        if arm=='none': assert sum(aux.values()).item()==0
        if step==0:
            parameter=model.mfam.mask_head[-1].weight
            det=sum(v for k,v in losses.items() if not k.startswith('loss_mfam'))
            detection_gradient=torch.autograd.grad(det,parameter,retain_graph=True)[0].float().norm().item()
            auxiliary_gradient=torch.autograd.grad(sum(aux.values()),parameter,retain_graph=True)[0].float().norm().item()
            assert detection_gradient>0
        scaler.scale(sum(losses.values())/4).backward()
        rows.append({k:v.item() for k,v in aux.items()})
        print(arm,step,rows[-1],flush=True)
        del output,losses,aux
    scaler.unscale_(optimizer)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(model.parameters(),.1)
    scaler.step(optimizer);scaler.update()
    update=(model.mfam.output_proj.weight-before).abs().max().item();assert update>0
    result=dict(status='PASS',arm=arm,initial_model_sha256=initial,physical_batch=8,accumulation=4,
                detector_gradient=detection_gradient,auxiliary_gradient=auxiliary_gradient,update=update,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,steps=rows)
    path=ROOT/f'reports/104_sam3_role_control/preflight_{arm}.json'
    path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(result,flush=True)


if __name__=='__main__': main()
