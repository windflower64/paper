"""Actual batch8/accum4, inference equivalence, and auxiliary gradient checks."""
import hashlib
import json
import sys
from pathlib import Path
import torch

REPO=Path(__file__).resolve().parents[1]; ROOT=REPO.parent
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig
from src.solver import BaseSolver
from tools.preflight_q_rank1 import move_targets


def main(arm):
    assert arm in ('sam', 'box')
    torch.set_num_threads(4); torch.manual_seed(0)
    cfg=YAMLConfig(str(REPO/f'experiments/phase_s/s_sgc1_{arm}_b8a4_20e.yml'))
    model=cfg.model
    shim=BaseSolver.__new__(BaseSolver); shim.model=model
    shim.load_tuning_state(str(ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth'))
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.cpu().numpy().tobytes())
    model=model.cuda(); criterion=cfg.criterion.cuda(); optimizer=cfg.optimizer
    assert cfg.val_dataloader.dataset.sam_mask_root is None
    samples,targets=next(iter(cfg.train_dataloader))
    assert tuple(samples.shape)==(8,3,512,640)
    samples=samples.cuda(); targets=move_targets(targets,'cuda')
    model.eval()
    with torch.no_grad():
        ordinary=model(samples)
        assert 'sgc_group_loss' not in ordinary
        model.sgc_enabled=False; model.backbone.return_idx=[2,3]
        control=model(samples)
        for key in ('pred_logits','pred_boxes'):
            torch.testing.assert_close(ordinary[key],control[key],rtol=0,atol=0)
    model.sgc_enabled=True; model.backbone.return_idx=[1,2,3]
    del ordinary,control
    model.train(); optimizer.zero_grad(set_to_none=True)
    parameter=model.backbone.stages[1].blocks[0].layers[0].conv.weight
    before=parameter.detach().clone()
    scaler=torch.cuda.amp.GradScaler(init_scale=128)
    rows=[]; torch.cuda.reset_peak_memory_stats()
    for step,(samples,targets) in enumerate(cfg.train_dataloader):
        if step==4: break
        samples=samples.cuda(); targets=move_targets(targets,'cuda')
        with torch.autocast('cuda',dtype=torch.float16): output=model(samples,targets)
        losses=criterion(output,targets,epoch=0,step=step,global_step=step,epoch_step=len(cfg.train_dataloader))
        assert all(torch.isfinite(v) for v in losses.values())
        aux=losses['loss_sgc_group']; assert aux.item()>0
        if step==0:
            grad=torch.autograd.grad(aux,parameter,retain_graph=True)[0]
            assert torch.isfinite(grad).all() and grad.norm()>0
            aux_gradient=grad.float().norm().item()
        rows.append(float(aux.detach()))
        scaler.scale(sum(losses.values())/4).backward()
        del output,losses,aux
    scaler.unscale_(optimizer)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(model.parameters(),.1)
    scaler.step(optimizer); scaler.update()
    update=(parameter-before).abs().max().item(); assert update>0
    result=dict(status='PASS',arm=arm,initial_model_sha256=digest.hexdigest(),
                inference_exact_C_equivalence=True,physical_batch=8,accumulation=4,
                weighted_aux_losses=rows,aux_gradient=aux_gradient,parameter_update=update,
                peak_gib=torch.cuda.max_memory_allocated()/2**30)
    report=ROOT/'reports/105_sam_group_contrast_pilot'
    with (report/f'preflight_{arm}.json').open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
    print(result,flush=True)


if __name__=='__main__': main(sys.argv[1])
