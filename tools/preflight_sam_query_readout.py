"""Batch16 interface probe, no optimizer, no validation labels or AP."""
import json
import sys
from pathlib import Path
import torch
REPO=Path(__file__).resolve().parents[1]
ROOT=REPO.parent
sys.path.insert(0,str(REPO))
from src.core import YAMLConfig
from tools.preflight_sgc_scale_interface import seed,digest
from tools.preflight_q_rank1 import move_targets
from tools.sam_query_readout import QueryReadout,CrossAttentionReadout,geometry_loss,role_points


def main():
    out=ROOT/'reports/134_sam_query_readout/attempt03'
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);seed(0)
    config=REPO/'experiments/phase_s/sgc_scale_none_b16a2_20e.yml'
    cfg=YAMLConfig(str(config));cfg.yaml_cfg['HGNetv2']['pretrained']=False
    dataset=cfg.train_dataloader.dataset
    ids=json.loads((ROOT/'reports/133_sgc_scale_training/preflight.json').read_text(encoding='utf-8'))['image_ids']
    index={i:k for k,i in enumerate(dataset.ids)}
    items=[]
    for image_id in ids:
        seed(20260913+image_id);items.append(dataset[index[image_id]])
    samples=torch.stack([x for x,t in items]).cuda();targets=move_targets([t for x,t in items],'cuda')
    model=cfg.model
    checkpoint=ROOT/'outputs/SGC_SCALE_NONE_B16A2_20E_TESTDEV/seed0/best_stg1.pth'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['ema']['module'],strict=True)
    model.cuda().eval();criterion=cfg.criterion.cuda();before=digest(model)
    context={}
    hook=model.backbone.stages[0].register_forward_hook(lambda m,x,y:context.update(s4=y))
    with torch.no_grad(),torch.autocast('cuda'):
        base=model(samples)
    assert context['s4'].shape[-2:]==(128,160)
    layer=model.decoder.decoder.layers[0]
    original=layer.cross_attn
    seed(0)
    reader=QueryReadout(context['s4'].shape[1],original.embed_dim).cuda()
    wrapper=CrossAttentionReadout(original,reader,context)
    layer.cross_attn=wrapper
    results={'checkpoint':str(checkpoint),'image_ids':ids,'batch':16,
        'feature_shape':list(context['s4'].shape),'reader_parameters':sum(p.numel() for p in reader.parameters()),
        'optimizer':False,'scope':'engineering reachability only, not AP or edge preservation'}
    differences={}
    for name,gain,intervention in [('disabled',0.,'normal'),('zero',.1,'zero'),('normal',.1,'normal'),('shift',.1,'shift')]:
        wrapper.gain=gain;wrapper.intervention=intervention
        with torch.no_grad(),torch.autocast('cuda'):pred=model(samples)
        differences[name]={k:float((pred[k]-base[k]).abs().max()) for k in ('pred_logits','pred_boxes')}
        if name=='normal':
            fixed_indices=criterion.matcher(base,targets)['indices']
            matched=torch.cat([(pred['pred_boxes'][b,q]-base['pred_boxes'][b,q]).abs().flatten()
                for b,(q,t) in enumerate(fixed_indices) if len(q)])
            results['normal_matched_box_abs_delta']={'median':float(matched.median()),
                'mean':float(matched.mean()),'max':float(matched.max()),
                'unit':'normalized cxcywh coordinates, not IoU or AP'}
        if name in ('disabled','zero'):assert max(differences[name].values())==0
        if name=='normal':normal={k:pred[k].clone() for k in ('pred_logits','pred_boxes')}
        if name=='shift':
            results['normal_shift_difference']={k:float((pred[k]-normal[k]).abs().max()) for k in normal}
    assert differences['normal']['pred_boxes']>0 and differences['normal']['pred_logits']>0
    results['interventions_vs_original']=differences
    del base,pred,normal
    wrapper.intervention='normal';wrapper.gain=.1
    model.train()
    # Native training outputs (including denoising); fix BN for a read-only probe.
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda'):prediction=model(samples,targets)
    losses=criterion(prediction,targets,epoch=0,step=0,global_step=0,epoch_step=200)
    detection=sum(v for k,v in losses.items() if k!='loss_sgc_group');assert torch.isfinite(detection)
    results['detection_gradient_excludes_sgc']=True
    parameters=(reader.offsets.weight,reader.project.weight,context['s4'])
    gradients=torch.autograd.grad(detection,parameters,retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().max()>0 for g in gradients)
    results['detection_loss']=float(detection.detach())
    results['detection_gradient_norms']={k:float(g.float().norm()) for k,g in zip(('offsets','S4_projection','S4_feature'),gradients)}
    indices=criterion.matcher(prediction,targets)['indices']
    dn_count=prediction['dn_meta']['dn_num_split'][0] if prediction.get('dn_meta') else 0
    locations=reader.locations[:,dn_count:]
    assert locations.shape[1]==prediction['pred_boxes'].shape[1]
    results['denoising_queries_excluded']=dn_count
    results['training_output_mode']=True
    results['BN_fixed_for_read_only_probe']=True
    guidance={};teacher_gradients=[]
    for arm in ('sam','box'):
        loss,count=geometry_loss(locations,targets,indices,arm)
        grad=torch.autograd.grad(loss,reader.offsets.weight,retain_graph=True)[0]
        assert torch.isfinite(loss) and torch.isfinite(grad).all() and grad.abs().max()>0
        guidance[arm]={'loss':float(loss.detach()),'eligible_matched_images':count,'offset_gradient_norm':float(grad.norm())}
        teacher_gradients.append(grad.detach())
    results['guidance']=guidance
    results['sam_box_offset_gradient_difference']=float((teacher_gradients[0]-teacher_gradients[1]).norm())
    assert results['sam_box_offset_gradient_difference']>0
    results['peak_gib']=torch.cuda.max_memory_allocated()/2**30
    results['target_shape_difference_images']=sum(
        any(a.shape!=b.shape or not torch.equal(a,b) for a,b in zip(role_points(t,'sam'),role_points(t,'box')))
        for t in targets if role_points(t,'sam') is not None and role_points(t,'box') is not None)
    layer.cross_attn=original;hook.remove();context.clear();reader.locations=None
    after=digest(model);assert after==before
    results.update(status='PASS',original_state_unchanged=True,original_sha256=before)
    (out/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
