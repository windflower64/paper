"""Frozen feature-space finite-step audit. No parameter/optimizer update."""
import itertools
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'reports/148_sgc2_half_box/frozen_repo'))
from src.core import YAMLConfig
OUT = ROOT/'reports/152_s16_shape_finite_step/attempt02'


def project(g, normals):
    """Projection onto <=3 homogeneous halfspaces n.dot(g)>=0."""
    v = g.flatten().double()
    ns = [n.flatten().double() for n in normals if float(n.norm())>1e-12]
    if not ns: return g
    N = torch.stack([n/n.norm() for n in ns])
    dots = N@v; gram = N@N.T
    tolerance = max(float(v.norm())*1e-9,1e-12)
    if bool((dots>=-tolerance).all()): return g
    best, distance = torch.zeros_like(v), float(v.square().sum())
    for size in range(1,len(ns)+1):
        for active in itertools.combinations(range(len(ns)),size):
            idx = torch.tensor(active,device=g.device)
            coeff = torch.linalg.pinv(gram[idx][:,idx])@(-dots[idx])
            candidate = v+coeff@N[idx]
            if bool((coeff>=-tolerance).all()) and bool((N@candidate>=-tolerance).all()):
                delta = float((candidate-v).square().sum())
                if delta<distance: best,distance=candidate,delta
    return best.reshape_as(g).float()


def reader_loss(feature, rows, reader, label):
    values=[]
    mu,std,beta=reader
    for i,r in enumerate(rows):
        grid=torch.as_tensor(r['grid'],device='cuda')[None]
        sampled=F.grid_sample(feature[i:i+1].float(),grid,align_corners=False)[0].flatten(1).T
        geometry=torch.as_tensor(r['features']['S16'][:,:4],device='cuda')
        x=torch.cat([geometry,sampled],1)
        score=((x-mu)/std)@beta[:-1]+beta[-1]
        y=torch.as_tensor(r['labels'][label],device='cuda',dtype=torch.float32)
        p=np.flatnonzero(r['labels']['SAM']);n=np.flatnonzero(r['inside'] & ~r['labels']['SAM'])
        rng=np.random.default_rng(20260916+r['image_id']);count=min(32,len(p),len(n))
        idx=torch.as_tensor(np.r_[rng.choice(p,count,replace=False),rng.choice(n,count,replace=False)],device='cuda')
        values.append((score[idx]-y[idx]).square().mean())
    return torch.stack(values).mean()


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--parameter-space',action='store_true')
    parser.add_argument('--reader-path',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.parameter_space:
        OUT=ROOT/'reports/152_s16_shape_finite_step/attempt03_parameter'
    if args.output:OUT=args.output
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(0)
    export=ROOT/'reports/151_sam_internal_shape/none_backbone/scale_audit/gradient_export/fixed_reader.pt'
    if args.reader_path:export=args.reader_path
    data=torch.load(export,map_location='cpu',weights_only=False)
    selected=data['rows'];lookup={r['image_id']:r for r in selected}
    ann=json.loads((ROOT/'data/antiuav6k_common/annotations/instances_visible_common_train.json').read_text(encoding='utf-8'))
    annotations={a['image_id']:a for a in ann['annotations']}
    images={i['id']:i for i in ann['images']}
    reader=tuple(torch.as_tensor(data[k],device='cuda',dtype=torch.float32) for k in ('mu','std','beta'))
    cfg=YAMLConfig(str(ROOT/'reports/148_sgc2_half_box/frozen_repo/experiments/phase_s/s_sgc2_sam_c_plus_m_sd22_half_b8a4_20e.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    train=cfg.yaml_cfg['train_dataloader']['dataset']
    for key in ('img_folder','ann_file','infrared_folder','infrared_label_folder'):
        cfg.yaml_cfg['val_dataloader']['dataset'][key]=train[key]
    dataset=cfg.val_dataloader.dataset
    scale_batches=[]
    for scale in ('small','other'):
        indices=[i for i,r in enumerate(selected) if r['scale']==scale]
        scale_batches.extend(indices[start:start+16] for start in range(0,len(indices),16))
    loader=DataLoader(Subset(dataset,[dataset.ids.index(r['image_id']) for r in selected]),batch_sampler=scale_batches,
                      collate_fn=cfg.val_dataloader.collate_fn,num_workers=0)
    model=cfg.model.cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(data['checkpoint'],map_location='cpu',weights_only=False)['ema']['module'],strict=True)
    model.decoder.train();model.decoder.num_denoising=0
    parameters=list(model.backbone.stages[2].parameters()) if args.parameter_space else []
    for p in parameters: p.requires_grad_(True)
    original_parameters=torch.cat([p.detach().flatten() for p in parameters]) if parameters else None
    def assign(vector):
        start=0
        with torch.no_grad():
            for p in parameters:
                p.copy_(vector[start:start+p.numel()].reshape_as(p));start+=p.numel()
    criterion=cfg.criterion.cuda().train()
    cache={};replacement={'value':None}
    def hook(module,inputs,output):
        leaf=output if args.parameter_space else output.detach().requires_grad_(True) if replacement['value'] is None else replacement['value']
        cache['feature']=leaf
        return leaf
    handle=model.backbone.stages[2].register_forward_hook(hook)
    def forward(samples,targets):
        output=model(samples,targets)
        losses=criterion(output,targets,epoch=0,step=0,global_step=0,epoch_step=1)
        return output,{'classification':losses['loss_vfl'],'localization':losses['loss_bbox']+losses['loss_giou'],
                       'total':sum(losses.values())}
    batches=[]
    for batch,(samples,targets) in enumerate(loader):
        samples=samples.cuda();targets=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]
        rows=[lookup[int(t['image_id'])] for t in targets]
        # The val loader retains pixel XYXY boxes for evaluator use. Criterion
        # requires normalized CXCYWH. Rebuild from original annotation geometry
        # under the deterministic resize (normalized coordinates invariant).
        for target in targets:
            image_id=int(target['image_id']);info=images[image_id]
            x,y,w,h=annotations[image_id]['bbox']
            box=[(x+w/2)/info['width'],(y+h/2)/info['height'],w/info['width'],h/info['height']]
            target['boxes']=torch.tensor([box],device='cuda',dtype=torch.float32)
            assert bool(((target['boxes']>=0)&(target['boxes']<=1)).all()) and bool((target['boxes'][:,2:]>0).all())
            assert len(target['labels'])==1 and int(target['labels'][0])==0
        assert len({r['scale'] for r in rows})==1,'Do not mix scales in a diagnostic batch'
        output,losses=forward(samples,targets);feature=cache['feature'];base=feature.detach()
        assert feature.requires_grad and torch.isfinite(feature).all()
        sample_errors=[]
        for i,r in enumerate(rows):
            sampled=F.grid_sample(base[i:i+1],torch.as_tensor(r['grid'],device='cuda')[None],align_corners=False)[0].flatten(1).T.cpu().numpy()
            sample_errors.append(float(np.abs(sampled-r['features']['S16'][:,4:]).max()))
        assert max(sample_errors)<1e-5, sample_errors
        if args.parameter_space:
            def parameter_gradient(loss):
                grads=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
                return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for p,g in zip(parameters,grads)])
            detgrads={k:parameter_gradient(v) for k,v in losses.items()}
        else:
            detgrads={k:torch.autograd.grad(v,feature,retain_graph=True)[0].detach() for k,v in losses.items()}
        base_losses={k:float(v.detach()) for k,v in losses.items()}
        base_shape=float(reader_loss(feature,rows,reader,'SAM').detach())
        repeated,repeated_losses=forward(samples,targets)
        repeat_error=max(float((repeated[k]-output[k]).abs().max()) for k in ('pred_logits','pred_boxes'))
        assert repeat_error==0
        del repeated,repeated_losses
        trials={}
        if args.parameter_space:
            # Compute all directions before virtual writes to avoid autograd
            # version-counter invalidation of the original graph.
            directions={}
            for label in ('SAM','filled_extent','scrambled'):
                g=parameter_gradient(reader_loss(feature,rows,reader,label))
                directions[label]={'raw':g,'protected':project(g,list(detgrads.values()))}
            for label,dd in directions.items():
                for name,direction in dd.items():
                    delta=-direction*original_parameters.square().mean().sqrt()*1e-5/direction.square().mean().sqrt().clamp_min(1e-20)
                    try:
                        assign(original_parameters+delta)
                        with torch.no_grad():
                            new_output,new_losses=forward(samples,targets)
                            changed_feature=cache['feature'].detach()
                            new_shape=float(reader_loss(changed_feature,rows,reader,'SAM'))
                            actual={k:float(v) for k,v in new_losses.items()}
                        trials[f'{label}/{name}']={'delta_losses':{k:actual[k]-base_losses[k] for k in actual},
                            'delta_real_SAM_mse':new_shape-base_shape,
                            'relative_parameter_rms':float(delta.square().mean().sqrt()/original_parameters.square().mean().sqrt()),
                            'relative_feature_rms':[float(v) for v in ((changed_feature-base).square().mean((1,2,3)).sqrt()/base.square().mean((1,2,3)).sqrt()).cpu()],
                            'first_order_dots':{k:float((direction*v).sum()) for k,v in detgrads.items()},
                            'direction_nonzero':int(float(direction.norm())>1e-12)}
                        print(batch,rows[0]['scale'],label,name,'parameter',json.dumps(trials[f'{label}/{name}']['delta_losses']), 'SAM_mse',new_shape-base_shape,flush=True)
                    finally:
                        assign(original_parameters)
                    assert torch.equal(torch.cat([p.detach().flatten() for p in parameters]),original_parameters)
            batches.append({'scale':rows[0]['scale'],'image_ids':[r['image_id'] for r in rows],
                'base_losses':base_losses,'base_SAM_mse':base_shape,'eval_sample_max_error':max(sample_errors),
                'repeat_forward_max_error':repeat_error,'trials':trials})
            del output,losses,feature,detgrads,g,directions
            continue
        for label in ('SAM','filled_extent','scrambled'):
            value=reader_loss(feature,rows,reader,label)
            g=torch.autograd.grad(value,feature,retain_graph=True)[0].detach()
            protected=torch.stack([project(g[i],[v[i] for v in detgrads.values()]) for i in range(len(rows))])
            original_dots={k:[float((g[i]*v[i]).sum()) for i in range(len(rows))] for k,v in detgrads.items()}
            for name,direction in (('raw',g),('protected',protected)):
                rms=base.square().mean((1,2,3),keepdim=True).sqrt()
                norm=direction.square().mean((1,2,3),keepdim=True).sqrt()
                change=-direction*rms*.001/norm.clamp_min(1e-20)
                replacement['value']=(base+change).detach()
                with torch.no_grad():
                    new_output,new_losses=forward(samples,targets)
                    new_shape=float(reader_loss(replacement['value'],rows,reader,'SAM'))
                    actual={k:float(v) for k,v in new_losses.items()}
                trials[f'{label}/{name}']={'delta_losses':{k:actual[k]-base_losses[k] for k in actual},
                    'delta_real_SAM_mse':new_shape-base_shape,
                    'relative_feature_rms':[float(v) for v in (change.square().mean((1,2,3)).sqrt()/rms.flatten()).cpu()],
                    'first_order_dots':original_dots if name=='raw' else {k:[float((direction[i]*v[i]).sum()) for i in range(len(rows))] for k,v in detgrads.items()},
                    'direction_nonzero':sum(float(direction[i].norm())>1e-12 for i in range(len(rows)))}
                print(batch,rows[0]['scale'],label,name,json.dumps(trials[f'{label}/{name}']['delta_losses']), 'SAM_mse',new_shape-base_shape,flush=True)
                replacement['value']=None
        batches.append({'scale':rows[0]['scale'],'image_ids':[r['image_id'] for r in rows],
                        'base_losses':base_losses,'base_SAM_mse':base_shape,'eval_sample_max_error':max(sample_errors),
                        'repeat_forward_max_error':repeat_error,'trials':trials})
        del output,losses,feature,detgrads,g,protected
    handle.remove();assert all(p.grad is None for p in model.parameters())
    report={'status':'virtual_parameter_steps_restored_exact' if args.parameter_space else 'finite_step_complete_no_parameter_updates',
            'space':'stage2_parameters' if args.parameter_space else 'S16_features',
            'step_relative_rms':1e-5 if args.parameter_space else .001,'batches':batches,
            'selected_parameter_count':sum(p.numel() for p in parameters),
            'optimizer_steps':0,'parameter_gradients_absent':True,'reader_fit_probe_parent_groups_disjoint':True,
            'criterion_target_format':'normalized_cxcywh_from_original_annotations',
            'caveats':['virtual local check, not committed training or AP',
                       'projection first-order nonconflict is constructed, finite-step result is the check',
                       'common SAM-trained reader and SAM-selected internal points; controls isolate target values only',
                       'frozen detector has seen probe training images, masks and extent privileged',
                       'decoder training-format loss without DN; not exact full original training recipe']}
    (OUT/'finite_step.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
