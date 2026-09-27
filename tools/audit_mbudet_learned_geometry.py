"""Read-only geometric audit of the trained dense offset, GT only for scoring."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

ROOT=Path('E:/two_paper'); REPORT=ROOT/'reports/158_mbudet_rgb_alignment'
SNAPSHOT=REPORT/'frozen_repo'
sys.path.insert(0,str(SNAPSHOT)); sys.path.insert(0,str(SNAPSHOT/'tools'))
from run_mbudet_alignment import prepare, fixed_hashes, write


def describe(values):
    a=np.asarray(values,dtype=float)
    return dict(count=len(a),mean=float(a.mean()),p50=float(np.median(a)),
                p90=float(np.quantile(a,.9)),p95=float(np.quantile(a,.95)))


def score_split(model, split):
    cfg=prepare('aligned',REPORT/'geometry_runtime',32)
    spec=cfg.yaml_cfg['val_dataloader']['dataset']
    spec.update(img_folder=str(ROOT/f'data/antiuav6k_common/images/{split}'),
        ann_file=str(ROOT/f'data/antiuav6k_common/annotations/instances_visible_common_{split}.json'),
        infrared_folder=f'F:/data/Anti-UAV/Anti_UAV_6K/{split}/infrared/images',
        infrared_label_folder=str(ROOT/f'data/antiuav6k_ir_raw_verified/{split}/labels'))
    dataset=cfg.val_dataloader.dataset
    # Full development-test; train random 800 frames with EVAL preprocessing.
    indices=list(range(len(dataset))) if split=='test' else sorted(np.random.default_rng(158).choice(len(dataset),800,replace=False).tolist())
    loader=DataLoader(Subset(dataset,indices),batch_size=32,num_workers=0,
        collate_fn=lambda items:(torch.stack([i[0] for i in items]),[i[1] for i in items]))
    rows=[]
    for batch,(samples,targets) in enumerate(loader):
        samples=samples.cuda()
        with torch.inference_mode():
            d=model.detector
            rgb=d.encoder(d.backbone(samples[:,:3]))
            ir=d.thermal_encoder(d.thermal_backbone(samples[:,3:]))
            flows=[level.offset(torch.cat((r,t),1)) for level,r,t in zip(model.alignment,rgb,ir)]
        for i,target in enumerate(targets):
            if len(target['boxes'])!=1 or len(target['infrared_boxes'])!=1: continue
            r=target['boxes'][0].float(); t=target['infrared_boxes'][0].float()
            r_center=(r[:2]+r[2:])/2; t_center=(t[:2]+t[2:])/2
            truth=t_center-r_center
            record=dict(image_id=int(target['image_id'][0]),raw_error_pixels=float(torch.linalg.vector_norm(truth)),
                rgb_box_wh_pixels=(r[2:]-r[:2]).tolist(), ir_box_wh_pixels=(t[2:]-t[:2]).tolist(),levels=[])
            for flow in flows:
                _,_,h,w=flow.shape
                with torch.inference_mode():
                    grid=(r_center/r_center.new_tensor([640,512])*2-1).cuda().reshape(1,1,1,2)
                    f=F.grid_sample(flow[i:i+1].float(),grid,align_corners=False).reshape(2).cpu()
                predicted=f*f.new_tensor([640/w,512/h])
                error=predicted-truth
                inside=bool((error.abs() <= (t[2:]-t[:2])/2).all())
                error_pixels=float(torch.linalg.vector_norm(error))
                record['levels'].append(dict(error_pixels=error_pixels,read_inside_ir_box=inside,
                    normalized_box_error=float(torch.linalg.vector_norm(error/(t[2:]-t[:2]).clamp_min(1))),
                    predicted_shift_pixels=predicted.tolist(),true_shift_pixels=truth.tolist()))
            rows.append(record)
        if batch%10==0: print('GEOMETRY',split,batch,len(rows),flush=True)
    raw=[r['raw_error_pixels'] for r in rows]
    summary=dict(images=len(indices),both_positive=len(rows),raw_center_error_pixels=describe(raw),levels=[])
    for level in range(2):
        values=[r['levels'][level] for r in rows]
        errors=[v['error_pixels'] for v in values]
        summary['levels'].append(dict(stride=16*2**level,learned_center_error_pixels=describe(errors),
            fraction_better_than_no_alignment=float(np.mean(np.asarray(errors)<np.asarray(raw))),
            read_center_inside_ir_gt_fraction=float(np.mean([v['read_inside_ir_box'] for v in values])),
            raw_center_inside_ir_gt_fraction=float(np.mean([r['raw_error_pixels']==0 or
                np.all(np.abs(v['true_shift_pixels'])<=np.array(r['ir_box_wh_pixels'])/2) for r,v in zip(rows,values)])),
            normalized_box_error=describe([v['normalized_box_error'] for v in values])))
    return summary,rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--last',action='store_true')
    args=parser.parse_args()
    output=REPORT/('learned_geometry_last.json' if args.last else 'learned_geometry_audit.json')
    assert not output.exists()
    cfg=prepare('aligned',REPORT/'geometry_runtime',32)
    model=cfg.model.cuda().eval()
    run=ROOT/'outputs/M_MBUDET_RGB_ALIGNED_B32A1_20E_TESTDEV/seed0'
    path=run/('last.pth' if args.last else 'best_stg1.pth')
    checkpoint=torch.load(path,map_location='cpu')
    model.load_state_dict(checkpoint['ema']['module'],strict=True)
    before=fixed_hashes(model)
    summaries={}; rows={}
    for split in ('train','test'):
        summaries[split],rows[split]=score_split(model,split)
    assert before==fixed_hashes(model)
    write(output,dict(status='PASS',checkpoint=str(path),
        epoch=checkpoint['last_epoch'],coordinates='RGB reference -> IR backward sampling',
        inference_uses_gt=False,gt_used_only_to_score_offsets=True,
        train_sampling_seed=158,summary=summaries,rows=rows,
        limitation_zh='GT中心处位移质量是有利条件的诊断，不等同于所有预测位置的对齐或检测因果增益。'))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__': main()
