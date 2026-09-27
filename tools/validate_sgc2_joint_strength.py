"""Fixed EMA checkpoints, final M residual scaling, and full-test diagnostics."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from torchvision.ops import box_convert

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig
from src.data import CocoEvaluator
from tools.diagnose_m_target_tradeoffs import target_metrics, false_predictions, mean

RUN = ROOT / 'outputs/C_PLUS_M_SD22_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0'
OUT = ROOT / 'reports/111_sgc2_joint_strength/verified'


def scale_features(original, conditioned, alpha):
    if alpha == 1:
        return conditioned
    if alpha == 0:
        return original
    return [v + alpha * (c - v) for v, c in zip(original, conditioned)]


def main(config=None, run=RUN, out=OUT, epochs=(7, 13, 19), strengths=(0., .5, 1.), checkpoint_paths=None):
    run, out = Path(run), Path(out)
    assert 0. in strengths and 1. in strengths
    torch.set_num_threads(4)
    torch.manual_seed(0)
    out.mkdir(parents=True, exist_ok=True)
    cfg = YAMLConfig(str(config or REPO / 'experiments/phase_s/s_sgc2_sam_c_plus_m_sd22_b8a4_20e.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    model = cfg.model.cuda().eval()
    # ModelEMA freezes every parameter; reproduce that runtime state as well
    # as its state_dict. Loading EMA tensors alone does not copy these flags.
    model.requires_grad_(False)
    loader = cfg.val_dataloader
    assert len(loader.dataset) == 1820 and loader.dataset.sam_mask_root is None
    coco = loader.dataset.coco
    alpha = [1.0]
    def hook(module, args, output):
        return scale_features(args[0], output, alpha[0])
    handle = model.sd2_conditioner.register_forward_hook(hook)
    logs = [json.loads(s) for s in (run / 'log.txt').read_text().splitlines() if s.strip()]
    results = {}
    for epoch in epochs:
        dest = out / f'epoch{epoch}'
        dest.mkdir(exist_ok=True)
        checkpoint = Path(checkpoint_paths[epoch]) if checkpoint_paths else run / f'checkpoint{epoch:04d}.pth'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['last_epoch'] == epoch
        model.load_state_dict(state['ema']['module'], strict=True)
        del state
        evaluators = {str(a): CocoEvaluator(coco, ['bbox']) for a in strengths}
        rows, images, cache = [], [], {}
        with torch.no_grad():
            for batch, (samples, targets) in enumerate(loader):
                samples = samples.cuda()
                sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
                predictions = {}
                for a in strengths:
                    alpha[0] = a
                    output = model(samples)
                    processed = cfg.postprocessor(output, sizes)
                    predictions[str(a)] = [{k: v.cpu() for k, v in p.items()} for p in processed]
                    evaluators[str(a)].update({int(t['image_id']): p for t, p in zip(targets, predictions[str(a)])})
                if batch == 0:
                    alpha[0] = 0.
                    hooked = model(samples)
                    conditioner = model.sd2_conditioner
                    model.sd2_conditioner = None
                    removed = model(samples)
                    model.sd2_conditioner = conditioner
                    for key in ('pred_logits', 'pred_boxes'):
                        torch.testing.assert_close(hooked[key], removed[key], atol=0, rtol=0)
                for i, target in enumerate(targets):
                    image_id = int(target['image_id'])
                    anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])) if not a.get('iscrowd', 0)]
                    gt = box_convert(torch.tensor([a['bbox'] for a in anns], dtype=torch.float32).reshape(-1, 4), 'xywh', 'xyxy')
                    cache[image_id] = {m: predictions[m][i] for m in predictions}
                    record = {'image_id': image_id, 'targets': len(anns)}
                    for mode in predictions:
                        record[mode] = {str(t): false_predictions(predictions[mode][i], gt, t) for t in (.25, .5)}
                    images.append(record)
                    for j, ann in enumerate(anns):
                        area = ann.get('area', ann['bbox'][2] * ann['bbox'][3])
                        row = {'image_id': image_id, 'scale': 'small' if area < 1024 else 'medium' if area < 9216 else 'large'}
                        for mode in predictions:
                            row[mode] = target_metrics(predictions[mode][i], gt[j])
                        rows.append(row)
                if batch % 40 == 0:
                    print(f'epoch {epoch}: {len(images)}/1820 images, {len(strengths)} modes', flush=True)
        metrics = {}
        for mode, evaluator in evaluators.items():
            evaluator.synchronize_between_processes()
            evaluator.accumulate()
            evaluator.summarize()
            metrics[mode] = evaluator.coco_eval['bbox'].stats.tolist()
        error = max(abs(a-b) for a,b in zip(metrics['1.0'], logs[epoch]['test_coco_eval_bbox']))
        by_scale = {}
        for scale in ('small', 'medium'):
            selected = [r for r in rows if r['scale'] == scale]
            by_scale[scale] = {'count': len(selected)}
            for mode in evaluators:
                by_scale[scale][mode] = {k: mean([r[mode][k] for r in selected]) for k in ('best_iou_300', 'best_iou_10', 'top1_iou')}
                for tag in ('50', '75'):
                    by_scale[scale][mode]['covered_'+tag] = sum(r[mode]['correct_rank_'+tag] is not None for r in selected)
            for mode in (str(a) for a in strengths if a != 0.):
                both = [r for r in selected if r[mode]['correct_rank_75'] is not None and r['0.0']['correct_rank_75'] is not None]
                by_scale[scale][mode]['both_correct_count'] = len(both)
                by_scale[scale][mode]['correct_score_delta_vs_off'] = mean([r[mode]['correct_score_75']-r['0.0']['correct_score_75'] for r in both])
                by_scale[scale][mode]['correct_rank_delta_vs_off'] = mean([r[mode]['correct_rank_75']-r['0.0']['correct_rank_75'] for r in both])
        false = {}
        for group in ('all', 'empty'):
            selected = [r for r in images if group == 'all' or r['targets'] == 0]
            false[group] = {'images': len(selected)}
            for mode in evaluators:
                false[group][mode] = {str(t): {k: sum(r[mode][str(t)][k] for r in selected) for k in ('background', 'localization', 'duplicate')} for t in (.25, .5)}
        result = {'epoch': epoch, 'checkpoint': str(checkpoint), 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'images': len(images), 'targets': len(rows), 'weight_source': 'EMA', 'metrics': metrics, 'normal_vs_log_max_error': error, 'zero_equals_removal': True, 'by_scale': by_scale, 'false_predictions': false}
        result['configured_final_residual_scale'] = model.sd2_conditioner.final_residual_scale
        result['mode_semantics'] = 'Multipliers relative to configured output, not absolute M scales'
        for name, data in [('summary', result), ('targets', rows), ('images', images)]:
            (dest / (name+'.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        torch.save(cache, dest / 'predictions.pt')
        results[str(epoch)] = result
        (out / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print('EPOCH_DONE', epoch, 'AP', {m: v[0] for m,v in metrics.items()}, 'reproduction_error', error, flush=True)
        assert error < 0.0002, 'Normal inference failed reproduction tolerance; inspect before interpreting.'
    handle.remove()
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
