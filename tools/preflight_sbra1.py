"""Full detector SBRA1 preflight; no persisted training weights."""
import json
import hashlib
import sys
from pathlib import Path
import torch
REPO=Path(__file__).resolve().parents[1]; ROOT=REPO.parent
sys.path.insert(0,str(REPO))
from src.core import YAMLConfig
from src.solver import BaseSolver
from tools.preflight_q_rank1 import move_targets

def main():
    torch.set_num_threads(4); torch.manual_seed(0)
    arm=sys.argv[1] if len(sys.argv)>1 else 'sam'
    assert arm in ('sam','box','none')
    cfg=YAMLConfig(str(REPO/f'experiments/phase_s/s_sbra1_{arm}_b8a4_20e.yml'))
    model=cfg.model
    shim=BaseSolver.__new__(BaseSolver); shim.model=model
    shim.load_tuning_state(str(ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth'))
    digest=hashlib.sha256()
    for k,v in model.state_dict().items():
        digest.update(k.encode()); digest.update(v.cpu().numpy().tobytes())
    model.cuda(); criterion=cfg.criterion.cuda(); opt=cfg.optimizer
    assert cfg.val_dataloader.dataset.sam_mask_root is None
    loader=cfg.train_dataloader
    samples,targets=next(iter(loader)); assert tuple(samples.shape)==(8,3,512,640)
    samples=samples.cuda(); targets=move_targets(targets,'cuda')
    stage=model.backbone.stages[2]; module=stage.sbra
    model.eval()
    with torch.no_grad():
        actual=model(samples)
        stage.sbra=None
        expected=model(samples)
        stage.sbra=module
        for k in ('pred_logits','pred_boxes'):torch.testing.assert_close(actual[k],expected[k],rtol=0,atol=0)
    del actual,expected
    model.train(); scaler=torch.cuda.amp.GradScaler(init_scale=128)
    history=[]; torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):output=model(samples,targets)
        losses=criterion(output,targets,epoch=0,step=step,global_step=step,epoch_step=len(loader))
        assert all(torch.isfinite(v) for v in losses.values())
        det=sum(v for k,v in losses.items() if k!='loss_sbra_relation')
        gd=torch.autograd.grad(det,module.relation[-1].weight,retain_graph=True)[0]
        ga=torch.autograd.grad(losses['loss_sbra_relation'],module.relation[-1].weight,retain_graph=True)[0]
        scaler.scale(sum(losses.values())).backward(); scaler.unscale_(opt)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        row=dict(step=step,aux=float(losses['loss_sbra_relation'].detach()),det_grad=float(gd.norm()),aux_grad=float(ga.norm()))
        history.append(row); print(row,flush=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(),.1); scaler.step(opt); scaler.update()
        del output,losses,det,gd,ga
    assert history[0]['det_grad']==0 and history[-1]['det_grad']>0
    assert all((r['aux_grad']==0 if arm=='none' else r['aux_grad']>0) for r in history)
    result=dict(status='PASS',arm=arm,initial_sha256=digest.hexdigest(),aux_weight=model.sbra_aux_weight,batch=8,exact_initial_forward=True,validation_no_masks=True,steps=history,peak_gib=torch.cuda.max_memory_allocated()/2**30,
                caveat='Three repeated-batch engineering updates, not detector validation or epoch training')
    out=ROOT/'reports/119_sbra1'; out.mkdir(parents=True,exist_ok=True)
    (out/f'preflight_{arm}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
