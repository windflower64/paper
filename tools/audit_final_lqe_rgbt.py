"""Final-output LQE score intervention; boxes/query features remain identical."""
import hashlib
import json
import sys
from pathlib import Path
import torch
from torchvision.ops import box_convert, box_iou

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
FROZEN = ROOT / 'reports/148_sgc2_half_box/frozen_repo'
sys.path.insert(0, str(FROZEN))
from src.core import YAMLConfig
from src.data import CocoEvaluator
sys.path.insert(0, str(REPO))
from tools.diagnose_m_target_tradeoffs import target_metrics, false_predictions

ARMS = {
    'SAM': ('s_sgc2_sam_c_plus_m_sd22_half_b8a4_20e.yml', 'C_PLUS_M_SD22_HALF_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV'),
    'BOX': ('s_sgc2_box_c_plus_m_sd22_half_b8a4_20e.yml', 'C_PLUS_M_SD22_HALF_SGC2_BOX_DECAY9_14_B8A4_20E_TESTDEV'),
    'NONE': ('c_plus_m_sd22_half_sgc0_b8a4_20e.yml', 'C_PLUS_M_SD22_HALF_SGC0_B8A4_20E_TESTDEV'),
}


def inspect(name, config, run_name):
    dest = ROOT / 'reports/149_final_lqe_audit' / name
    dest.mkdir(parents=True, exist_ok=True)
    config_path = FROZEN / 'experiments/phase_s' / config
    if not config_path.exists():
        # NONE config was created after the original SAM source snapshot.
        config_path = ROOT / 'outputs' / run_name / 'seed0/artifacts/experiments/phase_s' / config
    cfg = YAMLConfig(str(config_path))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    model = cfg.model.cuda().eval().requires_grad_(False)
    checkpoint = ROOT / 'outputs' / run_name / 'seed0/best_stg1.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['ema']['module'], strict=True)
    epoch = state['last_epoch']
    del state
    captured = []
    def hook(module, args, output):
        captured.append((args[0].detach(), output.detach()))
    handles = [module.register_forward_hook(hook) for module in model.modules() if module.__class__.__name__ == 'LQE']
    assert handles
    loader = cfg.val_dataloader
    assert len(loader.dataset) == 1820 and loader.dataset.sam_mask_root is None
    coco = loader.dataset.coco
    evaluators = {mode: CocoEvaluator(coco, ['bbox']) for mode in ('normal', 'without_final_lqe')}
    rows, images, cache = [], [], {}
    with torch.no_grad():
        for batch, (samples, targets) in enumerate(loader):
            captured.clear()
            output = model(samples.cuda())
            assert captured, 'LQE was not called'
            raw, adjusted = captured[-1]
            torch.testing.assert_close(adjusted, output['pred_logits'], atol=0, rtol=0)
            alternative = dict(output, pred_logits=raw)
            sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
            predictions = {mode: [{k: v.cpu() for k,v in p.items()} for p in cfg.postprocessor(o, sizes)]
                           for mode,o in (('normal',output),('without_final_lqe',alternative))}
            for mode in predictions:
                evaluators[mode].update({int(t['image_id']): p for t,p in zip(targets,predictions[mode])})
            for image_index,target in enumerate(targets):
                image_id = int(target['image_id'])
                anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])) if not a.get('iscrowd',0)]
                gt = box_convert(torch.tensor([a['bbox'] for a in anns],dtype=torch.float32).reshape(-1,4),'xywh','xyxy')
                cache[image_id] = {mode: predictions[mode][image_index] for mode in predictions}
                images.append({'image_id': image_id, 'targets':len(anns),
                               **{mode: false_predictions(predictions[mode][image_index],gt,.5) for mode in predictions}})
                for target_index,ann in enumerate(anns):
                    row = {'image_id':image_id, 'scale':'small' if ann.get('area',ann['bbox'][2]*ann['bbox'][3])<1024 else 'medium'}
                    for mode in predictions:
                        pred = predictions[mode][image_index]
                        row[mode] = target_metrics(pred,gt[target_index])
                        scores, order = pred['scores'].sort(descending=True)
                        ious = box_iou(pred['boxes'][order],gt[target_index:target_index+1]).flatten()
                        found = torch.where(ious>=.9)[0]
                        row[mode]['correct_rank_90'] = int(found[0])+1 if len(found) else None
                    rows.append(row)
            if batch % 40 == 0:
                print(f'{name}: {batch*8+len(targets)}/1820',flush=True)
    metrics = {}
    for mode,evaluator in evaluators.items():
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        metrics[mode] = evaluator.coco_eval['bbox'].stats.tolist()
    logs = [json.loads(s) for s in (checkpoint.parent/'log.txt').read_text().splitlines() if s.strip()]
    logged = next(r for r in logs if r['epoch']==epoch)['test_coco_eval_bbox']
    error = max(abs(a-b) for a,b in zip(metrics['normal'],logged))
    summary = {'arm':name,'epoch':epoch,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               'metrics':metrics,'normal_vs_log_max_error':error,'images':len(images),'targets':len(rows),
               'intervention':'remove only final LQE logit correction after the original forward; same boxes and latent features',
               'small_target':{},'false_predictions':{},
               'caveats':['same-weight inference intervention, not LQE retraining ablation',
                          'other decoder layer LQE unchanged; single seed and development set']}
    for mode in predictions:
        selected = [r for r in rows if r['scale']=='small']
        summary['small_target'][mode] = {'n':len(selected),'mean_top1_iou':sum(r[mode]['top1_iou'] for r in selected)/len(selected),
            **{f'rank1_{tag}':sum(r[mode][f'correct_rank_{tag}']==1 for r in selected) for tag in ('50','75','90')},
            **{f'top10_{tag}':sum(r[mode][f'correct_rank_{tag}'] is not None and r[mode][f'correct_rank_{tag}']<=10 for r in selected) for tag in ('75','90')}}
        summary['false_predictions'][mode] = {key:sum(r[mode][key] for r in images) for key in ('background','localization','duplicate')}
    for handle in handles:
        handle.remove()
    for filename,value in (('summary',summary),('targets',rows),('images',images)):
        (dest/(filename+'.json')).write_text(json.dumps(value,indent=2),encoding='utf-8')
    torch.save(cache,dest/'predictions.pt')
    assert error < .0002, 'Normal inference did not reproduce training'
    print('RESULT',name,json.dumps(summary),flush=True)
    del model
    torch.cuda.empty_cache()
    return summary


def main():
    torch.set_num_threads(4)
    requested = sys.argv[1:] or list(ARMS)
    assert all(name in ARMS for name in requested)
    results = {}
    for name,args in ARMS.items():
        if name in requested:
            results[name] = inspect(name,*args)
        else:
            results[name] = json.loads((ROOT/'reports/149_final_lqe_audit'/name/'summary.json').read_text())
    (ROOT / 'reports/149_final_lqe_audit/comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
