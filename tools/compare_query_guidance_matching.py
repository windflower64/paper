"""Same-forward comparison of final/first/pre query matching; no training."""
import json
import sys
from pathlib import Path
import torch
REPO=Path(__file__).resolve().parents[1];ROOT=REPO.parent
sys.path.insert(0,str(REPO))
from src.core import YAMLConfig
from src.solver import BaseSolver
from tools.preflight_sgc_scale_interface import seed
from tools.preflight_q_rank1 import move_targets
from tools.sam_query_readout import role_points


def main():
    out=ROOT/'reports/136_query_guidance_matching';out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);seed(0)
    cfg=YAMLConfig(str(REPO/'experiments/phase_s/sgc_scale_none_b16a2_20e.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained']=False
    dataset=cfg.train_dataloader.dataset;lookup={im:i for i,im in enumerate(dataset.ids)}
    groups=json.loads((ROOT/'reports/132_sgc_scale_interface/attempt02/protocol.json').read_text(encoding='utf-8'))['sample_groups']
    cache={}
    for group,ids in groups.items():
        items=[]
        for im in ids:
            seed(20260913+im);items.append(dataset[lookup[im]])
        cache[group]=(torch.stack([x for x,t in items]),[t for x,t in items])
    seed(0);model=cfg.model;shim=BaseSolver.__new__(BaseSolver);shim.model=model
    shim.load_tuning_state(str(ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth'))
    model.cuda().train();matcher=cfg.criterion.matcher.cuda()
    captured={}
    hook=model.decoder.decoder.layers[0].cross_attn.register_forward_pre_hook(
        lambda m,args:captured.update(ref=args[1].detach()))
    rows=[]
    for group,(cpu,cputargets) in cache.items():
        targets=move_targets(cputargets,'cuda')
        with torch.no_grad(),torch.autocast('cuda'):pred=model(cpu.cuda(),targets)
        dn=pred['dn_meta']['dn_num_split'][0]
        refs=captured['ref'][:,dn:,0].float()
        choices={name:matcher(output,targets)['indices'] for name,output in (
            ('final',pred),('first',pred['aux_outputs'][0]),('pre',pred['pre_outputs']))}
        for b,t in enumerate(targets):
            both={a:role_points(t,a) for a in ('sam','box')}
            if any(v is None for v in both.values()):continue
            record={'image_id':groups[group][b],'group':group,'choices':{}}
            for name,indices in choices.items():
                q=int(indices[b][0][0]);box=refs[b,q];low=box[:2]-box[2:];high=box[:2]+box[2:]
                armrows={}
                for arm,roles in both.items():
                    armrows[arm]=[float(((p>low)&(p<high)).all(-1).float().mean()) for p in roles]
                record['choices'][name]={'query':q,'fractions':armrows}
            # Structural upper bound: does ANY ordinary query have reachable points?
            all_reachable={}
            for arm,roles in both.items():
                all_reachable[arm]=[]
                for p in roles:
                    inside=((p[None]>(refs[b,: ,None,:2]-refs[b,:,None,2:]))&
                            (p[None]<(refs[b,:,None,:2]+refs[b,:,None,2:]))).all(-1)
                    all_reachable[arm].append(float(inside.float().mean(-1).max()))
            record['any_query_upper_bound']=all_reachable;rows.append(record)
        print(group,'done',flush=True)
    hook.remove()
    summary={}
    for mode in ('final','first','pre'):
        summary[mode]={}
        for arm in ('sam','box'):
            values=[r['choices'][mode]['fractions'][arm] for r in rows]
            summary[mode][arm]={'no_reachable':[sum(v[i]==0 for v in values) for i in range(3)],
                'mean_fraction':[sum(v[i] for v in values)/len(values) for i in range(3)]}
        summary[mode]['query_changed_vs_final']=sum(r['choices'][mode]['query']!=r['choices']['final']['query'] for r in rows)
    result={'status':'complete','images':len(rows),'summary':summary,'rows':rows,
        'note':'same native-training forward, no optimizer, temporary BN updates; three roles interior/boundary/context; any-query is GT-assisted diagnostic upper bound, not a proposal'}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
