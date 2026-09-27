"""Fixed training-image gradient pilot, two existing C+M states, no optimizer.

This establishes upstream connectivity only; no AP or edge-restoration claim.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
from torch.nn import functional as F

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO.parent
sys.path.insert(0,str(REPO))
from src.core import YAMLConfig
from src.solver import BaseSolver
from tools.preflight_q_rank1 import move_targets
from tools.sgc_scale_interface import scale_losses

CONFIG=REPO/'experiments/phase_s/s_sgc2_sam_c_plus_m_sd22_half_b8a4_20e.yml'
CHECKPOINTS={
    'initial':ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth',
    'best':ROOT/'outputs/C_PLUS_M_SD22_HALF_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0/best_stg1.pth',
}


def digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)


def select_samples(rows):
    rng=random.Random(20260913)
    groups={
      'rescued_tiny': [r for r in rows if r.get('area',1e9)<256 and r.get('S4',{}).get('joint_usable') and not r['S8']['joint_usable']],
      'rescued_other_small':[r for r in rows if 256<=r.get('area',1e9)<1024 and r.get('S4',{}).get('joint_usable') and not r['S8']['joint_usable']],
      'covered_small':[r for r in rows if r.get('area',1e9)<1024 and r.get('S8',{}).get('joint_usable')],
      'covered_medium':[r for r in rows if 1024<=r.get('area',1e9)<9216 and r.get('S8',{}).get('joint_usable')],
    }
    result={}
    for name,candidates in groups.items():
        rng.shuffle(candidates)
        selected=[];videos=set()
        for r in candidates:
            video=r['file_name'].rsplit('_',1)[0]
            if video in videos:continue
            videos.add(video);selected.append(r['image_id'])
            if len(selected)==16:break
        # The tiny-rescue subset has fewer than 16 source videos. Prefer one
        # frame/video, then fill with distinct frames; do not invent diversity.
        if len(selected)<16:
            for r in candidates:
                if r['image_id'] not in selected:
                    selected.append(r['image_id'])
                if len(selected)==16:break
        assert len(selected)==16,(name,len(selected))
        result[name]=selected
    return result


def stats(gradient, reference):
    if gradient is None:
        return {'norm':0.,'cosine':None,'ratio':0.}
    assert torch.isfinite(gradient).all()
    g=gradient.float().flatten();r=reference.float().flatten()
    return {'norm':float(g.norm()),'cosine':float(F.cosine_similarity(g,r,dim=0)),
            'ratio':float(g.norm()/r.norm().clamp_min(1e-12))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve();out.relative_to((ROOT/'reports').resolve())
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    coverage=json.loads((ROOT/'reports/131_sgc_supervision_coverage/per_image.json').read_text(encoding='utf-8'))
    sample_groups=select_samples(coverage)
    seed(0)
    dataset=YAMLConfig(str(CONFIG)).train_dataloader.dataset
    by_id={im:i for i,im in enumerate(dataset.ids)}
    cached={}
    for group,ids in sample_groups.items():
        items=[]
        for image_id in ids:
            seed(20260913+image_id)
            items.append(dataset[by_id[image_id]])
        cached[group]=(torch.stack([x for x,t in items]),[t for x,t in items])
    with (out/'protocol.json').open('x',encoding='utf-8') as f:
        json.dump({'sample_groups':sample_groups,'batch':16,'BN':'eval',
          'optimizer':False,'description':'training labels only, diagnostic stratified batches; not representative epoch averages; some frames share videos',
          'config':str(CONFIG),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},f,ensure_ascii=False,indent=2)
    results=[]
    for state,path in CHECKPOINTS.items():
        seed(0)
        cfg=YAMLConfig(str(CONFIG));cfg.yaml_cfg['HGNetv2']['pretrained']=False
        model=cfg.model
        if state=='initial':
            shim=BaseSolver.__new__(BaseSolver);shim.model=model;shim.load_tuning_state(str(path))
        else:
            model.load_state_dict(torch.load(path,map_location='cpu')['ema']['module'],strict=True)
        model=model.cuda()
        criterion=cfg.criterion.cuda()
        # Keep the same feature return path but suppress production aux loss.
        model.sgc_aux_weight=0.
        model.train()
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
        before=digest(model)
        captured={}
        handles=[model.backbone.stages[idx].register_forward_hook(
            lambda m,i,o,key=key:captured.update({key:o})) for idx,key in [(0,'s4'),(1,'s8'),(2,'s16')]]
        early_name,early=next((n,p) for n,p in model.named_parameters()
                              if n.startswith('backbone.stages.0.') and p.requires_grad and p.ndim==4)
        later=next(p for p in model.backbone.stages[2].parameters() if p.requires_grad)
        mparam=next(p for p in model.sd2_conditioner.parameters() if p.requires_grad)
        state_rows=[];torch.cuda.reset_peak_memory_stats()
        for batch_index,(group,(cpu_samples,cpu_targets)) in enumerate(cached.items()):
            seed(20260913+batch_index)
            samples=cpu_samples.cuda();targets=move_targets(cpu_targets,'cuda')
            with torch.autocast('cuda',dtype=torch.float16):
                prediction=model(samples,targets)
            losses=criterion(prediction,targets,epoch=0,step=batch_index,global_step=batch_index,epoch_step=200)
            detection=sum(v for k,v in losses.items() if k!='loss_sgc_group')
            assert torch.isfinite(detection)
            assert captured['s4'].shape[-2:]==(128,160)
            assert captured['s8'].shape[-2:]==(64,80)
            assert captured['s16'].shape[-2:]==(32,40)
            det_early,det_s4=torch.autograd.grad(detection,(early,captured['s4']),retain_graph=True)
            row={'state':state,'group':group,'image_ids':sample_groups[group],
                 'detection_loss':float(detection.detach()),'early_parameter':early_name,
                 'detection_early_norm':float(det_early.float().norm()),'arms':{}}
            for arm in ['sam','box']:
                aux,routes=scale_losses(captured['s4'],captured['s8'],targets,arm)
                expected='S4_rescue' if group.startswith('rescued') else 'S8'
                assert all(route==expected for route in routes),(group,routes)
                metrics={}
                for component in ['s8','rescue','total']:
                    g_early,g_s4,g_later,g_m=torch.autograd.grad(aux[component],
                        (early,captured['s4'],later,mparam),retain_graph=True,allow_unused=True)
                    assert g_m is None or g_m.abs().max()==0
                    assert g_later is None or g_later.abs().max()==0
                    metrics[component]={'loss':float(aux[component].detach()),
                      'early_parameter':stats(g_early,det_early),'S4_feature':stats(g_s4,det_s4),
                      'direct_S16_parameter_grad':g_later is not None and bool(g_later.abs().max()>0),
                      'direct_M_parameter_grad':g_m is not None and bool(g_m.abs().max()>0)}
                assert metrics['total']['early_parameter']['norm']>0
                row['arms'][arm]={'routes':routes,'components':metrics}
            # Return-feature hooks and objective computation must not edit any tensor.
            for value in losses.values():assert torch.isfinite(value)
            state_rows.append(row)
            print(json.dumps({'state':state,'group':group,'sam':row['arms']['sam']['components']['total'],
                              'box':row['arms']['box']['components']['total']}),flush=True)
            captured.clear()
            del prediction,losses,detection,aux,det_early,det_s4,g_early,g_s4,samples,targets
        for handle in handles:handle.remove()
        after=digest(model);assert before==after,'Model state changed during read-only probe'
        item={'state':state,'checkpoint':str(path),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'model_hash_before':before,'model_hash_after':after,'state_unchanged':True,
              'peak_gib':torch.cuda.max_memory_allocated()/2**30,'rows':state_rows}
        with (out/f'{state}.json').open('x',encoding='utf-8') as f:json.dump(item,f,ensure_ascii=False,indent=2)
        results.append(item)
        del model,criterion,cfg,early,later,mparam
        if state=='initial':del shim
        torch.cuda.empty_cache()
    with (out/'summary.json').open('x',encoding='utf-8') as f:
        json.dump({'status':'complete','timestamp_utc':datetime.now(timezone.utc).isoformat(),
          'physical_batch':16,'images_per_state':64,'runs':results,
          'conclusion':'coverage and upstream gradient reachability only; no detector improvement demonstrated',
          'caveats':['S4 loss has no direct gradient to downstream S16/M parameters',
                     'shared early parameter changes can affect later features, but this pilot has no parameter update',
                     'single first-layer gradient is not whole-model gradient','BN frozen; not a full training step',
                     'gradient cosine is descriptive and does not predict AP']},f,ensure_ascii=False,indent=2)
    print('SCALE_INTERFACE_PREFLIGHT_COMPLETE',flush=True)


if __name__=='__main__':main()
