"""Privileged, video-held-out box-action readout with frozen fine RGB evidence."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = Path('E:/two_paper')
REPO = ROOT/'reports/148_sgc2_half_box/frozen_repo'
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig

OUT = ROOT/'reports/155_shape_conditioned_detection_evidence'


def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def scramble(mask, image_id):
    points = np.flatnonzero(mask)
    if not len(points): return mask.copy()
    y, x = np.unravel_index(points, mask.shape)
    anchors = points[(x==x.min()) | (x==x.max()) | (y==y.min()) | (y==y.max())]
    yy, xx = np.indices(mask.shape)
    inside = (xx>=x.min()) & (xx<=x.max()) & (yy>=y.min()) & (yy<=y.max())
    remaining = np.setdiff1d(np.flatnonzero(inside), anchors)
    result = np.zeros(mask.size, dtype=bool); result[anchors] = True
    rng = np.random.default_rng(20260917+image_id)
    result[rng.choice(remaining, len(points)-len(anchors), replace=False)] = True
    result = result.reshape(mask.shape)
    assert result.sum() == mask.sum()
    ry, rx = np.where(result)
    assert (rx.min(),rx.max(),ry.min(),ry.max()) == (x.min(),x.max(),y.min(),y.max())
    return result


def descriptor(feature, support, candidate):
    # Same number of inputs for all controls; empty support returns zeros.
    f, m, c = feature.reshape(feature.shape[0], -1), support.ravel(), candidate.ravel()
    fg, bg = m*c, (1-m)*c
    a = f@fg/max(float(fg.sum()),1e-6)
    b = f@bg/max(float(bg.sum()),1e-6)
    return np.r_[a, b, fg.sum()/max(m.sum(),1e-6), fg.sum()/max(c.sum(),1e-6),
                 bg.sum()/max(c.sum(),1e-6), c.mean()].astype(np.float32)


def evaluate(rows, field, held_sets):
    details = []
    for fold, held in enumerate(held_sets):
        fit = [r for r in rows if r['group'] not in held]
        test = [r for r in rows if r['group'] in held]
        assert fit and test
        x = np.concatenate([r['features'][field] for r in fit]).astype(np.float64)
        y = np.concatenate([r['target'] for r in fit])
        mu, std = x.mean(0), np.maximum(x.std(0), 1e-6)
        z = np.c_[(x-mu)/std, np.ones(len(x))]
        penalty = np.eye(z.shape[1]); penalty[-1,-1] = 0
        beta = np.linalg.solve(z.T@z+10*penalty, z.T@y)
        for r in test:
            p = np.c_[(r['features'][field]-mu)/std, np.ones(9)]@beta
            p -= p[0]
            action = int(p.argmax()) if p.max()>.005 else 0
            details.append(dict(image_id=r['image_id'], group=r['group'], scale=r['scale'], fold=fold,
                action=action, gain=float(r['target'][action]), base_iou=r['base_iou'],
                oracle_gain=float(max(r['target'])), prediction=p.tolist()))
    assert len(details) == len(rows) == len({r['image_id'] for r in details})
    summary = {}
    for scale in ('all','small','other'):
        for fold in (None,0,1,2,3):
            chosen = [r for r in details if (scale=='all' or r['scale']==scale)
                      and (fold is None or r['fold']==fold)]
            summary[f'{scale}/fold{fold}'] = dict(n=len(chosen), mean_gain=float(np.mean([r['gain'] for r in chosen])),
                changed=sum(r['action']!=0 for r in chosen), improved=sum(r['gain']>.01 for r in chosen),
                worsened=sum(r['gain']<-.01 for r in chosen), oracle_gain=float(np.mean([r['oracle_gain'] for r in chosen])))
    return dict(summary=summary, rows=details)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'conclusion.json').exists(): raise RuntimeError('Completed output exists')
    torch.set_num_threads(4); torch.manual_seed(0)
    pilot_path = ROOT/'reports/150_sam_local_geometry/attempt02/teacher_geometry_pilot.json'
    pilot = json.loads(pilot_path.read_text())
    selected = pilot['selected_ids']; actions = {r['image_id']:r for r in pilot['samples']}
    geometry = {r['image_id']:r for r in json.loads((ROOT/'reports/153_shape_feature_transfer/teacher/geometry.json').read_text())['rows']}
    protocol154 = json.loads((ROOT/'reports/154_shape_feature_learning_audit/protocol.json').read_text())
    held_sets = [set(h) for h in protocol154['held_groups']]
    assert selected == protocol154['selected'] and len(selected)==600
    checkpoint = ROOT/'outputs/C_PLUS_M_SD22_HALF_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0/best_stg1.pth'
    config = REPO/'experiments/phase_s/s_sgc2_sam_c_plus_m_sd22_half_b8a4_20e.yml'
    cfg = YAMLConfig(str(config)); cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    train = cfg.yaml_cfg['train_dataloader']['dataset']
    for key in ('img_folder','ann_file','infrared_folder','infrared_label_folder'):
        cfg.yaml_cfg['val_dataloader']['dataset'][key] = train[key]
    dataset = cfg.val_dataloader.dataset
    loader = DataLoader(Subset(dataset, [dataset.ids.index(i) for i in selected]),batch_size=8,
        shuffle=False, num_workers=0, collate_fn=cfg.val_dataloader.collate_fn)
    model = cfg.model.cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['ema']['module'],strict=True)
    features = {}
    handles = [model.backbone.stages[k].register_forward_hook(
        lambda m,a,o,k=k: features.__setitem__(k,o.detach())) for k in (0,2)]
    projections = {}
    write('protocol.json',dict(status='running', selected=selected, small=300, held_groups=[sorted(h) for h in held_sets],
        checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        geometry_source=str(pilot_path), geometry_sha256=hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
        grid=32, frozen_random_channels=32, ridge_lambda=10, action_threshold=.005, updates=0,
        actions='same nine translations/resizes as report150; no action grid search',
        candidate_selection='frozen original detector top1, no GT selection',
        caveats=['training-side privileged ability probe, not test AP or deployed model',
                 'GT-prompt masks shared with filled and scrambled; no mask oracle as final model',
                 'detector saw training images, only reader video groups held out',
                 'one top1 box per positive image; does not measure background false detections or recall']))
    rows = []; max_box_error = 0.; changed_fraction = []
    with torch.no_grad():
        for batch,(samples,targets) in enumerate(loader):
            predictions = cfg.postprocessor(model(samples.cuda()), torch.stack([t['orig_size'] for t in targets]).cuda())
            for j,(p,t) in enumerate(zip(predictions,targets)):
                image_id=int(t['image_id']); g=geometry[image_id]; old=actions[image_id]
                h,w=g['height'],g['width']
                base=np.array(old['candidates'][0]['box'])
                actual=p['boxes'][p['scores'].argmax()].cpu().numpy()
                actual=np.clip(actual,0,[w,h,w,h])
                max_box_error=max(max_box_error,float(abs(actual-base).max()))
                assert abs(actual-base).max()<.01, (image_id,actual,base)
                size=np.maximum(base[2:]-base[:2],1.)
                center=(base[:2]+base[2:])/2
                lo=np.maximum(center-size*.75,0); hi=np.minimum(center+size*.75,[w,h])
                xs=lo[0]+(np.arange(32)+.5)/32*(hi[0]-lo[0])
                ys=lo[1]+(np.arange(32)+.5)/32*(hi[1]-lo[1])
                xx,yy=np.meshgrid(xs,ys)
                grid=torch.tensor(np.stack([xx/w*2-1,yy/h*2-1],-1),dtype=torch.float32,device='cuda')[None]
                mask=np.asarray(Image.open(ROOT/f'reports/104_sam3_role_control/masks_train/masks/{image_id:06d}.png'))>0
                sam=mask[np.clip(yy.astype(int),0,h-1),np.clip(xx.astype(int),0,w-1)]
                e=np.array(g['extent'])*[w,h,w,h]
                filled=(xx>=e[0]) & (xx<e[2]) & (yy>=e[1]) & (yy<e[3])
                scrambled=scramble(sam,image_id)
                changed_fraction.append(float((scrambled!=sam).mean()))
                # Uniform uses a fixed central rectangular support of the predicted
                # ROI, preventing a wholly empty complementary descriptor.
                uniform=(xx>=base[0]) & (xx<base[2]) & (yy>=base[1]) & (yy<base[3])
                support={'uniform':uniform,'filled':filled,'sam':sam,'scrambled':scrambled}
                descriptors={}
                geom=np.array([c['geometry']+[v*v for v in c['geometry'][:4]] for c in old['candidates']],dtype=np.float32)
                descriptors['geometry']=geom
                for level in (0,2):
                    source=features[level][j:j+1].float()
                    channels=source.shape[1]
                    if level not in projections:
                        generator=torch.Generator(device='cuda').manual_seed(20260917+level)
                        projections[level]=torch.randn(32,channels,generator=generator,device='cuda')/channels**.5
                    sampled=F.grid_sample(source,grid,padding_mode='border',align_corners=False)[0]
                    reduced=torch.einsum('oc,chw->ohw',projections[level],sampled).cpu().numpy()
                    for name,m in support.items():
                        vectors=[]
                        for c in old['candidates']:
                            b=c['box']; within=(xx>=b[0]) & (xx<b[2]) & (yy>=b[1]) & (yy<b[3])
                            vectors.append(descriptor(reduced,m,within))
                        v=np.array(vectors)
                        descriptors[f'S{4*2**level}/{name}']=np.c_[geom,v,v-v[:1]]
                rows.append(dict(image_id=image_id, group='_'.join(Path(g['file_name']).stem.split('_')[:2]),
                    scale=old['scale'], features=descriptors, base_iou=old['base_iou'],
                    target=np.array([c['delta_iou'] for c in old['candidates']])))
            if batch%8==0: print('cached',len(rows),flush=True)
    for handle in handles: handle.remove()
    assert len(rows)==600 and sum(r['scale']=='small' for r in rows)==300
    results={}
    for name in rows[0]['features']:
        results[name]=evaluate(rows,name,held_sets)
        print(name,json.dumps(results[name]['summary']['small/foldNone']),flush=True)
    output=dict(status='complete_no_detector_updates', n=600, small=300, max_base_box_error=max_box_error,
        scramble_changed_fraction=float(np.mean(changed_fraction)), results=results,
        limitations=['mean IoU changes, not AP; masks are privileged GT-prompt teacher inputs',
                     'fixed random compression and linear reader limit this specific probe',
                     'cannot infer all other architectures fail if this probe has no gain'])
    write('conclusion.json',output)
    print('COMPLETE',flush=True)


if __name__=='__main__': main()
