"""Fixed-weight, training-set mechanism diagnostic; not deployable evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from preflight_sqer1 import build_config, move_targets, seed_all, sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-images', type=int, default=256)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    seed_all(20261002)
    torch.set_num_threads(4)
    repo = Path(__file__).resolve().parents[1]
    cfg = build_config(repo, repo/'experiments/phase_m/m_layout_recovery_r2_main.yml')
    dl = cfg.yaml_cfg['train_dataloader']
    dl.update(shuffle=False, drop_last=False, total_batch_size=4, num_workers=0)
    transforms = dl['dataset']['transforms']
    transforms['ops'] = [x for x in transforms['ops'] if not x['type'].startswith('Random')]
    transforms.pop('policy', None)
    dl['collate_fn'] = {'type': 'BatchImageCollateFunction'}
    model = cfg.model.cuda().eval()
    checkpoint = repo.parent/'outputs/M_LAYOUT_RECOVERY_20E_TESTDEV/R2_OTE2_LAYOUT/seed0/best_stg1.pth'
    saved = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(saved['ema']['module'], strict=True)
    del saved
    reader = model.sqer
    assert reader.evidence_only
    loader0 = cfg.train_dataloader
    ids = np.random.default_rng(20261002).permutation(len(loader0.dataset))[:a.max_images].tolist()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(loader0.dataset, ids), batch_size=4,
                                         num_workers=0, collate_fn=loader0.collate_fn)
    from src.zoo.dfine.sam_query_evidence_reader import _region_distributions, _box_occupancy
    matcher = cfg.criterion.matcher.cuda()
    cache = {}
    original = reader.forward
    def wrapped(*args, **kwargs):
        kwargs['return_aux'] = True
        result = original(*args, **kwargs)
        cache['result'] = result
        return result
    reader.forward = wrapped
    def capture(module, args):
        cache['memory'] = args[1].detach().float()
    handle = reader.query_blocks[0].register_forward_pre_hook(capture)
    rows = []
    skips = dict(no_teacher=0, not_selected=0, invalid_region=0)
    def delta(attn, memory):
        pooled = attn @ memory
        e = torch.cat((pooled[0]-pooled[2], pooled[1]-pooled[2]))
        norm = e.norm()
        z = (reader.quality_head(e)-reader.quality_head(torch.zeros_like(e))).tanh()
        return float((z*norm/(norm+1)).item())
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for bi, (samples, targets) in enumerate(loader):
            targets = move_targets(targets, torch.device('cuda'))
            out = model(samples.cuda())
            base = {'pred_logits':out['base_pred_logits'], 'pred_boxes':out['pred_boxes']}
            matches = matcher(base, targets)['indices']
            r = cache['result']; k = r['query_indices'].shape[1]
            for b, (sources, dests) in enumerate(matches):
                t = targets[b]
                if float(torch.as_tensor(t.get('sam_quality', 0)).flatten()[0]) <= 0 or t.get('masks') is None:
                    skips['no_teacher'] += len(sources); continue
                for q, gt in zip(sources.tolist(), dests.tolist()):
                    if len(t['boxes']) != 1:
                        raise RuntimeError('This audit requires single-instance Anti-UAV masks')
                    selected = (r['query_indices'][b] == q).nonzero().flatten()
                    if not len(selected): skips['not_selected'] += 1; continue
                    j = int(selected[0]); grid = r['roi_grid'][b,j][None].float()
                    mask = t['masks'][gt].float()
                    bm = _box_occupancy(t['boxes'][gt].reshape(1,1,4), *mask.shape[-2:])[0,0]
                    distributions = []
                    for field in (mask, bm):
                        occ = F.grid_sample(field[None,None], grid, align_corners=False)[0,0]
                        dist, valid, mass = _region_distributions(occ)
                        distributions.append((dist[0], bool(valid[0])))
                    if not all(x[1] for x in distributions): skips['invalid_region'] += 1; continue
                    sam, box = [x[0] for x in distributions]
                    attn = r['attention'][b,j].float()
                    uniform = torch.full_like(attn, 1/attn.shape[-1])
                    memory = cache['memory'][b*k+j]
                    def kl(x,y): return (x*(x.clamp_min(1e-8).log()-y.clamp_min(1e-8).log())).sum(-1).cpu().tolist()
                    ds = {name:delta(weights,memory) for name,weights in [('student',attn),('sam',sam),('box',box),('uniform',uniform),('sam_roles_swapped',sam[[2,1,0]])]}
                    actual = float(r['delta_logits'][b,q,0])
                    if abs(ds['student']-actual)>0.003: raise RuntimeError(f'Readout reconstruction mismatch {ds["student"]} vs {actual}')
                    rows.append({'image_id':int(t['image_id'].item()),'query':q,
                        'sam_to_student_kl':kl(sam,attn),'sam_to_uniform_kl':kl(sam,uniform),
                        'box_to_student_kl':kl(box,attn),'sam_to_box_kl':kl(sam,box),
                        'student_role_tv':[(float((attn[x]-attn[y]).abs().sum()/2)) for x,y in [(0,1),(0,2),(1,2)]],
                        'delta':ds,'base_logit':float(base['pred_logits'][b,q,0]),'reconstruction_error':abs(ds['student']-actual)})
            print(f'images={min((bi+1)*4,len(ids))} eligible={len(rows)}', flush=True)
    handle.remove()
    reader.forward = original
    summary = {'protocol':'seen training images, no augmentation, fixed EMA, matched positives only; no deployable AP or generalization claim',
               'checkpoint':str(checkpoint),'sha256':sha256_file(checkpoint),'sample_indices':ids,'eligible':len(rows),'skips':skips}
    for name in ['sam_to_student_kl','sam_to_uniform_kl','box_to_student_kl','sam_to_box_kl','student_role_tv']:
        summary[name] = np.mean([r[name] for r in rows],axis=0).tolist() if rows else None
    summary['delta_mean'] = {n:float(np.mean([r['delta'][n] for r in rows])) for n in ['student','sam','box','uniform','sam_roles_swapped']} if rows else {}
    summary['sam_delta_gt_student_fraction'] = float(np.mean([r['delta']['sam']>r['delta']['student'] for r in rows])) if rows else None
    for name, value in [('rows.json',rows),('summary.json',summary)]:
        with (a.output/name).open('x',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
