"""Read-only checkpoint diagnostics: teacher occupancy, student regions, box geometry."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.ops import box_convert

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig
from tools.diagnose_m_target_tradeoffs import target_metrics, false_predictions

OUT = ROOT/'reports/98_s_mfam1/region_diagnosis'


def describe(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), p10=float(np.quantile(a,.1)),
                median=float(np.median(a)), p90=float(np.quantile(a,.9))) if len(a) else None


def box_coverage(box, height=64, width=80):
    x0,y0,x1,y1 = box
    x = np.arange(width)
    y = np.arange(height)
    wx = np.clip(np.minimum(x+1,x1*width)-np.maximum(x,x0*width),0,1)
    wy = np.clip(np.minimum(y+1,y1*height)-np.maximum(y,y0*height),0,1)
    return torch.tensor(wy[:,None]*wx[None,:],dtype=torch.float32)


def teacher_audit():
    maskroot = ROOT/'reports/20_spatial_importance/S_TNDP2_RETENTION_DISTILLATION/masks_train'
    records = json.loads((maskroot/'records.json').read_text(encoding='utf-8'))
    rows = []
    for record in records:
        if not record['accepted']:
            continue
        image_id = record['image_id']
        mask = torch.from_numpy(np.asarray(Image.open(maskroot/'masks'/f'{image_id:06d}.png').convert('L')).copy()).float().gt(0).float()
        h,w = mask.shape
        x0,y0,x1,y1 = record['bbox_xyxy_annotation']
        area = (x1-x0)*(y1-y0)
        resized = F.interpolate(mask[None,None],size=(64,80),mode='area')[0,0]
        boxmap = box_coverage([x0/w,y0/h,x1/w,y1/h])
        rows.append(dict(image_id=image_id, scale='small' if area<1024 else 'medium',
                         mask_box_area_ratio=float(mask.sum()/area),
                         mask_bbox_iou=record['mask_bbox_iou'],
                         box_s8_area=float(boxmap.sum()), mask_s8_area=float(resized.sum()),
                         nonzero_cells=int((resized>0).sum()), cells_above_half=int((resized>=.5).sum()),
                         maximum_occupancy=float(resized.max())))
    result = {}
    for scale in ('small','medium'):
        selected = [r for r in rows if r['scale']==scale]
        result[scale] = {key:describe([r[key] for r in selected]) for key in
                        ('mask_box_area_ratio','mask_bbox_iou','box_s8_area','mask_s8_area','nonzero_cells','cells_above_half','maximum_occupancy')}
        result[scale]['zero_half_cells'] = sum(r['cells_above_half']==0 for r in selected)
    (OUT/'teacher_rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (OUT/'teacher_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('TEACHER',json.dumps(result),flush=True)


def evaluate_arm(arm):
    last = len(sys.argv)>2 and sys.argv[2]=='last'
    label = arm+'_LAST' if last else arm
    config = REPO/f'experiments/phase_s/s_mfam1_c_{arm.lower()}_b8a4_20e_testdev_local.yml'
    run = ROOT/f'outputs/S_MFAM1_C_{arm}_B8A4_20E_TESTDEV/seed0'
    cfg = YAMLConfig(str(config))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    model = cfg.model.cuda().eval()
    checkpoint = run/('last.pth' if last else 'best_stg1.pth')
    state = torch.load(checkpoint,map_location='cpu')
    model.load_state_dict(state['ema']['module'],strict=True)
    loader = cfg.val_dataloader
    assert loader.dataset.sam_mask_root is None
    coco = loader.dataset.coco
    capture = {}
    def hook(module, inputs, output):
        high = inputs[1].detach().float()
        capture['prob'] = output[1].detach().float().sigmoid().cpu()
        capture['ratio'] = ((output[0].detach().float()-high).square().mean((1,2,3)).sqrt()
                            / high.square().mean((1,2,3)).sqrt().clamp_min(1e-8)).cpu()
    rows, images = [], []
    with torch.inference_mode():
        for batch, (samples, targets) in enumerate(loader):
            samples = samples.cuda()
            sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
            predictions = {}
            model.mfam.intervention = 'learned'
            handle = model.mfam.register_forward_hook(hook)
            outputs = model(samples)
            handle.remove()
            maps, ratios = capture['prob'], capture['ratio']
            predictions['learned'] = [{k:v.cpu() for k,v in p.items()} for p in cfg.postprocessor(outputs,sizes)]
            for mode in (() if last else ('shifted','disabled')):
                model.mfam.intervention = 'learned' if mode=='shifted' else 'disabled'
                handle = None
                if mode=='shifted':
                    handle = model.mfam.mask_head.register_forward_hook(
                        lambda m,i,o:o.roll((o.shape[-2]//2,o.shape[-1]//2),(-2,-1)))
                outputs = model(samples)
                if handle is not None:
                    handle.remove()
                predictions[mode] = [{k:v.cpu() for k,v in p.items()} for p in cfg.postprocessor(outputs,sizes)]
            for index,target in enumerate(targets):
                image_id = int(target['image_id'])
                info = coco.imgs[image_id]
                anns = [a for a in coco.imgToAnns.get(image_id,[]) if not a.get('iscrowd',0)]
                gt = box_convert(torch.tensor([a['bbox'] for a in anns],dtype=torch.float32).reshape(-1,4),'xywh','xyxy')
                prob = maps[index,0]
                images.append(dict(image_id=image_id,empty=not anns,prob_mean=float(prob.mean()),
                                   prob_max=float(prob.max()),delta_rms_ratio=float(ratios[index]),
                                   fp={mode:false_predictions(ps[index],gt,.5) for mode,ps in predictions.items()}))
                for j,ann in enumerate(anns):
                    box = gt[j]/torch.tensor([info['width'],info['height']]*2)
                    coverage = box_coverage(box.tolist())
                    mask_in = float((prob*coverage).sum()/coverage.sum().clamp_min(1e-8))
                    mask_out = float((prob*(1-coverage)).sum()/(1-coverage).sum())
                    row = dict(image_id=image_id,scale='small' if ann['area']<1024 else 'medium',
                               inside_probability=mask_in,outside_probability=mask_out,
                               inside_half_coverage=float(((prob>=.5)*coverage).sum()/coverage.sum().clamp_min(1e-8)),
                               delta_rms_ratio=float(ratios[index]))
                    for mode in predictions:
                        row[mode] = target_metrics(predictions[mode][index],gt[j])
                    rows.append(row)
            if batch%50==0:
                print(arm,'processed',len(images),flush=True)
    summary = {}
    for scale in ('small','medium'):
        selected = [r for r in rows if r['scale']==scale]
        summary[scale] = {key:describe([r[key] for r in selected]) for key in
                          ('inside_probability','outside_probability','inside_half_coverage','delta_rms_ratio')}
        summary[scale]['localization'] = {mode:{key:describe([r[mode][key] for r in selected]) for key in
                                                ('top1_iou','best_iou_300','best_iou_10')}
                                           for mode in predictions}
        if not last:
            summary[scale]['shifted_best_iou_drop_over_005'] = sum(r['learned']['best_iou_300']-r['shifted']['best_iou_300']>.05 for r in selected)
            summary[scale]['shifted_best_iou_gain_over_005'] = sum(r['shifted']['best_iou_300']-r['learned']['best_iou_300']>.05 for r in selected)
    summary['empty'] = {key:describe([r[key] for r in images if r['empty']]) for key in ('prob_mean','prob_max','delta_rms_ratio')}
    summary['background_fp_05'] = {mode:sum(r['fp'][mode]['background'] for r in images) for mode in predictions}
    summary['empty_fp_05'] = {mode:sum(r['fp'][mode]['background'] for r in images if r['empty']) for mode in predictions}
    summary['checkpoint'] = str(checkpoint)
    summary['epoch'] = state.get('last_epoch',state.get('epoch'))
    summary['weight_source'] = 'ema.module'
    for name,data in [('rows',rows),('images',images),('summary',summary)]:
        (OUT/f'{label}_{name}.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('DONE',arm,flush=True)


if __name__=='__main__':
    torch.set_num_threads(8)
    OUT.mkdir(parents=True,exist_ok=True)
    if sys.argv[1]=='teacher':
        teacher_audit()
    else:
        evaluate_arm(sys.argv[1].upper())
