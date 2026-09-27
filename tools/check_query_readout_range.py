"""Training-image sampling-domain audit. No optimization or validation AP."""
import json
import argparse
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
from tools.sam_query_readout import role_points


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-train',action='store_true')
    args=parser.parse_args()
    out=ROOT/('reports/135_query_readout_range/native_training_range' if args.native_train else 'reports/135_query_readout_range')
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);seed(0)
    config=REPO/'experiments/phase_s/sgc_scale_none_b16a2_20e.yml'
    dataset=YAMLConfig(str(config)).train_dataloader.dataset
    groups=json.loads((ROOT/'reports/132_sgc_scale_interface/attempt02/protocol.json').read_text(encoding='utf-8'))['sample_groups']
    lookup={image_id:i for i,image_id in enumerate(dataset.ids)}
    cache={}
    for group,ids in groups.items():
        items=[]
        for image_id in ids:
            seed(20260913+image_id);items.append(dataset[lookup[image_id]])
        cache[group]=(torch.stack([x for x,t in items]),[t for x,t in items])
    results=[]
    paths={'initial':ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth',
           'best':ROOT/'outputs/SGC_SCALE_NONE_B16A2_20E_TESTDEV/seed0/best_stg1.pth'}
    for state,path in paths.items():
        seed(0);cfg=YAMLConfig(str(config));cfg.yaml_cfg['HGNetv2']['pretrained']=False
        model=cfg.model
        if state=='initial':
            shim=BaseSolver.__new__(BaseSolver);shim.model=model;shim.load_tuning_state(str(path))
        else:model.load_state_dict(torch.load(path,map_location='cpu')['ema']['module'],strict=True)
        model.cuda()
        model.train() if args.native_train else model.eval()
        matcher=cfg.criterion.matcher.cuda();before=digest(model)
        captured={}
        hook=model.decoder.decoder.layers[0].cross_attn.register_forward_pre_hook(
            lambda m,args:captured.update(reference=args[1].detach()))
        rows=[]
        for group,(cpu,targets_cpu) in cache.items():
            samples=cpu.cuda();targets=move_targets(targets_cpu,'cuda')
            with torch.no_grad(),torch.autocast('cuda'):pred=model(samples,targets if args.native_train else None)
            dn=pred['dn_meta']['dn_num_split'][0] if pred.get('dn_meta') else 0
            indices=matcher(pred,targets)['indices']
            for b,(queries,objects) in enumerate(indices):
                target=targets[b]
                both={a:role_points(target,a) for a in ('sam','box')}
                if any(v is None for v in both.values()):continue
                assert len(queries)==1
                box=captured['reference'][b,queries[0]+dn,0].float()
                low=box[:2]-box[2:];high=box[:2]+box[2:]
                row={'image_id':groups[group][b],'group':group,'reference':box.tolist(),'arms':{}}
                for arm,roles in both.items():
                    stats=[]
                    for points in roles:
                        inside=((points>low)&(points<high)).all(-1)
                        outside=torch.maximum(low-points,points-high).clamp_min(0)
                        norm=target['boxes'][0,2:].float().clamp_min(1e-4)
                        stats.append({'fraction_inside':float(inside.float().mean()),
                            'any_inside':bool(inside.any()),
                            'mean_squared_distance_lower_bound':float((outside/norm).square().sum(-1).mean())})
                    row['arms'][arm]=stats
                rows.append(row)
            print(state,group,'done',flush=True)
        hook.remove();unchanged=digest(model)==before
        if not args.native_train:assert unchanged
        summary={}
        for arm in ('sam','box'):
            summary[arm]={}
            for i,role in enumerate(('interior','boundary','context')):
                stats=[r['arms'][arm][i] for r in rows]
                summary[arm][role]={'images':len(stats),'no_reachable_target_images':sum(not s['any_inside'] for s in stats),
                    'mean_target_fraction_inside':sum(s['fraction_inside'] for s in stats)/len(stats)}
        results.append({'state':state,'state_unchanged':unchanged,'native_train':args.native_train,'rows':rows,'summary':summary})
        print(json.dumps({'state':state,'summary':summary}),flush=True)
        del model,cfg,matcher,pred,captured,samples,targets
        if state=='initial':del shim
        torch.cuda.empty_cache()
    (out/'summary.json').write_text(json.dumps({'status':'complete','runs':results,
        'scope':'64 stratified training images per state; original final-query matching; no readout training; not population estimate'},indent=2),encoding='utf-8')

if __name__=='__main__':main()
