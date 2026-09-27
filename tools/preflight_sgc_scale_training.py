"""Real batch16/acc2 optimizer preflight, discarded models, three fixed arms."""
import gc
import json
import sys
from pathlib import Path
import torch

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO.parent
sys.path.insert(0,str(REPO))
from src.core import YAMLConfig
from src.solver import BaseSolver
from tools.preflight_sgc_scale_interface import seed,digest
from tools.preflight_q_rank1 import move_targets
from tools.sgc_scale_interface import scale_losses as prototype
from tools.preflight_s_sgc2_c_plus_m import grad_summary

REPORT=ROOT/'reports/133_sgc_scale_training'
INIT=ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth'


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    destination=REPORT/'preflight.json'
    if destination.exists(): raise RuntimeError('Refusing to overwrite preflight')
    torch.set_num_threads(4)
    protocol=json.loads((ROOT/'reports/132_sgc_scale_interface/attempt02/protocol.json').read_text(encoding='utf-8'))
    ids=protocol['sample_groups']['rescued_tiny'][:8]+protocol['sample_groups']['covered_small'][:8]
    seed(0)
    cfg=YAMLConfig(str(REPO/'experiments/phase_s/sgc_scale_none_b16a2_20e.yml'))
    dataset=cfg.train_dataloader.dataset
    indices={image_id:i for i,image_id in enumerate(dataset.ids)}
    items=[]
    for image_id in ids:
        seed(20260913+image_id);items.append(dataset[indices[image_id]])
    samples=torch.stack([x for x,t in items]).cuda()
    targets=move_targets([t for x,t in items],'cuda')
    assert samples.shape==(16,6,512,640)
    results=[];initial_hash=None;reference_output=None
    for arm in ('none','box','sam'):
        seed(0)
        cfg=YAMLConfig(str(REPO/f'experiments/phase_s/sgc_scale_{arm}_b16a2_20e.yml'),use_amp=True)
        cfg.yaml_cfg['HGNetv2']['pretrained']=False
        model=cfg.model
        shim=BaseSolver.__new__(BaseSolver);shim.model=model;shim.load_tuning_state(str(INIT))
        before=digest(model)
        if initial_hash is None: initial_hash=before
        assert before==initial_hash,'Different initialization'
        model.cuda();criterion=cfg.criterion.cuda();optimizer=cfg.optimizer
        scaler=cfg.scaler
        assert cfg.yaml_cfg['gradient_accumulation_steps']==2
        assert len(cfg.val_dataloader.dataset)==1820
        assert cfg.val_dataloader.dataset.sam_mask_root is None
        model.eval()
        with torch.no_grad(),torch.autocast('cuda'):
            output=model(samples[:2])
        pred={k:output[k].detach().cpu() for k in ('pred_logits','pred_boxes')}
        if reference_output is None:reference_output=pred
        assert all(torch.equal(pred[k],reference_output[k]) for k in pred)
        assert 'sgc_group_loss' not in output
        del output,pred
        model.train();model.set_training_epoch(0)
        captured={}
        handles=[model.backbone.stages[i].register_forward_hook(
            lambda m,x,y,key=key:captured.update({key:y})) for i,key in ((0,'s4'),(1,'s8'))]
        # Native model.train() norm policy, not the report132 forced-BN-eval probe.
        bn_training=sum(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and m.training for m in model.modules())
        torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True)
        aux_values=[];steps=[]
        param=next(p for p in model.backbone.stages[0].parameters() if p.requires_grad)
        param_before=param.detach().clone()
        for micro in range(2):
            seed(20260913+micro)
            with torch.autocast('cuda'):
                output=model(samples,targets)
            with torch.no_grad():
                manual,_=prototype(captured['s4'],captured['s8'],targets,'sam')
                expected=manual['s8']
                if arm!='none':
                    extra,_=prototype(captured['s4'],captured['s8'],targets,arm)
                    assert float(extra['rescue'])>0
                    expected=expected+extra['rescue']
                torch.testing.assert_close(output['sgc_group_loss'],expected*10,rtol=1e-6,atol=1e-6)
            losses=criterion(output,targets,epoch=0,step=micro,global_step=micro,epoch_step=200)
            total=sum(losses.values());assert torch.isfinite(total)
            aux_values.append(float(output['sgc_group_loss'].detach()))
            scaler.scale(total/2).backward()
            steps.append(float(total.detach()))
            captured.clear()
            del output,losses,total,manual,expected
        scaler.unscale_(optimizer)
        visible=grad_summary(model,'backbone.');fusion=grad_summary(model,'sd2_conditioner.')
        thermal=grad_summary(model,'thermal_backbone.')
        assert visible['all_finite'] and fusion['all_finite']
        assert visible['with_nonzero_gradient']>0 and fusion['with_nonzero_gradient']>0
        assert thermal['with_gradient']==0
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_max_norm)
        scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
        assert scaler.get_scale()>=scale_before,'AMP skipped optimizer step'
        assert not torch.equal(param_before,param),'No real parameter update'
        optimizer.zero_grad(set_to_none=True)
        for h in handles:h.remove()
        model.set_training_epoch(14)
        with torch.no_grad(),torch.autocast('cuda'):
            retired=model(samples,targets)
        assert 'sgc_group_loss' not in retired
        assert model.backbone.sgc_s4_source is None
        row={'arm':arm,'initial_sha256':before,'physical_batch':16,'accumulation':2,
             'optimizer_steps':1,'native_bn_training_count':bn_training,
             'peak_gib':torch.cuda.max_memory_allocated()/2**30,'full_losses':steps,
             'weighted_aux':aux_values,'visible_gradient':visible,'fusion_gradient':fusion,
             'thermal_gradient':thermal,'retirement_ok':True,'inference_exact':True}
        results.append(row);print(json.dumps(row),flush=True)
        del model,criterion,optimizer,scaler,cfg,shim,param,param_before,retired
        gc.collect();torch.cuda.empty_cache()
    destination.write_text(json.dumps({'status':'PASS','image_ids':ids,'runs':results,
        'note':'One optimizer update per discarded model; no production checkpoint update'},indent=2),encoding='utf-8')
    print('TRAINING_PREFLIGHT_PASS',flush=True)


if __name__=='__main__':main()
