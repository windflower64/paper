"""Three discarded models, two native AMP updates each; no AP or checkpoint writes."""
import gc
import argparse
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
from tools.sam_query_readout import QueryReadout,CrossAttentionReadout,geometry_loss
from tools.query_reachable_guidance import reachable_geometry_loss


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reachable',action='store_true')
    args=parser.parse_args()
    out=ROOT/('reports/136_query_guidance_matching/reachable_updates' if args.reachable else 'reports/135_query_readout_range/optimizer_smoke')
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    config=REPO/'experiments/phase_s/sgc_scale_none_b16a2_20e.yml'
    seed(0);dataset=YAMLConfig(str(config)).train_dataloader.dataset
    groups=json.loads((ROOT/'reports/132_sgc_scale_interface/attempt02/protocol.json').read_text(encoding='utf-8'))['sample_groups']
    ids=groups['rescued_tiny']+groups['covered_small']
    lookup={im:i for i,im in enumerate(dataset.ids)};items=[]
    for im in ids:
        seed(20260913+im);items.append(dataset[lookup[im]])
    # Two mixed batch16 inputs, repeated once for the second optimizer update.
    order=[*range(8),*range(16,24),*range(8,16),*range(24,32)]
    batches=[]
    for start in (0,16):
        subset=[items[i] for i in order[start:start+16]]
        batches.append((torch.stack([x for x,t in subset]).cuda(),move_targets([t for x,t in subset],'cuda')))
    rows=[];initial=None
    for arm in ('none','box','sam'):
        seed(0);cfg=YAMLConfig(str(config),use_amp=True);cfg.yaml_cfg['HGNetv2']['pretrained']=False
        model=cfg.model;shim=BaseSolver.__new__(BaseSolver);shim.model=model
        shim.load_tuning_state(str(ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth'))
        context={};layer=model.decoder.decoder.layers[0]
        seed(0);reader=QueryReadout(64,layer.cross_attn.embed_dim)
        layer.cross_attn=CrossAttentionReadout(layer.cross_attn,reader,context)
        hook=model.backbone.stages[0].register_forward_hook(lambda m,x,y:context.update(s4=y))
        model.cuda().train();model.set_training_epoch(0)
        h=digest(model)
        if initial is None:initial=h
        assert h==initial
        optimizer=cfg.optimizer;scaler=cfg.scaler;criterion=cfg.criterion.cuda()
        bn=sum(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and m.training for m in model.modules())
        before=reader.offsets.weight.detach().clone();optimizer.zero_grad(set_to_none=True)
        values=[];torch.cuda.reset_peak_memory_stats()
        for micro in range(4):
            seed(20260913+micro);samples,targets=batches[micro%2]
            with torch.autocast('cuda'):pred=model(samples,targets)
            losses=criterion(pred,targets,epoch=0,step=micro,global_step=micro,epoch_step=200)
            native=sum(losses.values())
            if arm!='none':
                dn=pred['dn_meta']['dn_num_split'][0] if pred.get('dn_meta') else 0
                locations=reader.locations[:,dn:]
                assert locations.shape[1]==pred['pred_boxes'].shape[1]
                indices=criterion.matcher(pred,targets)['indices']
                if args.reachable:
                    aux,count=reachable_geometry_loss(locations,reader.references[:,dn:],targets,indices,arm)
                else:aux,count=geometry_loss(locations,targets,indices,arm)
            else:aux=native.new_zeros(());count=0
            total=native+aux  # Fixed 1.0 diagnostic weight, not tuned with validation AP.
            assert torch.isfinite(total)
            scaler.scale(total/2).backward()
            values.append({'micro':micro,'native_loss':float(native.detach()),'geometry_loss':float(aux.detach()),'eligible':count})
            if micro%2==1:
                scaler.unscale_(optimizer)
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                assert reader.offsets.weight.grad.abs().max()>0
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.clip_max_norm)
                scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                assert scaler.get_scale()>=scale,'Skipped AMP update'
                optimizer.zero_grad(set_to_none=True)
            context.clear();reader.locations=None
            del pred,losses,native,total,aux
        delta=float((reader.offsets.weight-before).norm());assert delta>0
        row={'arm':arm,'initial_sha256':h,'updates':2,'batch':16,'accumulation':2,
             'native_bn_training':bn,'offset_parameter_change':delta,
             'peak_gib':torch.cuda.max_memory_allocated()/2**30,'microbatches':values}
        rows.append(row);print(json.dumps(row),flush=True)
        (out/f'{arm}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        hook.remove();del model,shim,reader,layer,optimizer,criterion,scaler,cfg,before
        gc.collect();torch.cuda.empty_cache()
    (out/'summary.json').write_text(json.dumps({'status':'PASS','runs':rows,'image_ids':ids,
        'scope':'32 stratified training images, repeated microbatches, 2 optimizer steps; no AP; no trained weights retained'},indent=2),encoding='utf-8')

if __name__=='__main__':main()
