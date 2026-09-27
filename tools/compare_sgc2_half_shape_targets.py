"""SAM/BOX target geometry, ranking and background scores from saved outputs."""
import json
from pathlib import Path
import statistics
import torch
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parents[2]


def mean(values):
    return sum(values)/len(values) if values else None


def main():
    torch.set_num_threads(4)
    sources = {'SAM': ROOT / 'reports/112_sgc2_rgbt_half/final/diagnostics/epoch8/predictions.pt',
               'BOX': ROOT / 'reports/148_sgc2_half_box/diagnostics/epoch7/predictions.pt'}
    data = {k: torch.load(p, map_location='cpu', weights_only=False) for k,p in sources.items()}
    assert data['SAM'].keys() == data['BOX'].keys()
    annotations = json.loads((ROOT / 'data/antiuav6k_common/annotations/instances_visible_common_test.json').read_text())
    gt = {}
    for ann in annotations['annotations']:
        if not ann.get('iscrowd',0):
            assert ann['image_id'] not in gt
            gt[ann['image_id']] = ann
    rows = []
    for image_id, ann in gt.items():
        x,y,w,h = ann['bbox']
        box = torch.tensor([[x,y,x+w,y+h]], dtype=torch.float32)
        row = {'image_id': image_id, 'scale': 'small' if ann.get('area',w*h)<1024 else 'medium'}
        for name in data:
            prediction = data[name][image_id]['1.0']
            scores, order = prediction['scores'].sort(descending=True)
            ious = box_iou(prediction['boxes'][order],box).flatten()
            background = scores[ious < .1]
            item = {'best_iou': float(ious.max()), 'top1_iou': float(ious[0]),
                    'highest_background_score': float(background.max()) if len(background) else None}
            for t in (.5,.75,.9):
                found = torch.where(ious>=t)[0]
                score = float(scores[found[0]]) if len(found) else None
                rank = int(found[0])+1 if len(found) else None
                margin = score-item['highest_background_score'] if score is not None and item['highest_background_score'] is not None else None
                item[str(t)] = {'rank':rank, 'score':score, 'margin_to_background':margin}
            row[name] = item
        rows.append(row)
    summary = {}
    for scale in ('small','medium'):
        selected = [r for r in rows if r['scale']==scale]
        summary[scale] = {'n':len(selected)}
        for name in data:
            values = [r[name] for r in selected]
            summary[scale][name] = {'mean_best_iou':mean([r['best_iou'] for r in values]),
                'mean_top1_iou':mean([r['top1_iou'] for r in values]),
                'mean_highest_background_score':mean([r['highest_background_score'] for r in values if r['highest_background_score'] is not None])}
            for t in (.5,.75,.9):
                items = [v[str(t)] for v in values]
                ranks = [v['rank'] for v in items if v['rank'] is not None]
                summary[scale][name][str(t)] = {'covered':len(ranks), 'rank1':sum(v==1 for v in ranks),
                    'top10':sum(v<=10 for v in ranks), 'median_rank_if_covered':statistics.median(ranks) if ranks else None,
                    'mean_score_if_covered':mean([v['score'] for v in items if v['score'] is not None]),
                    'mean_margin_if_covered':mean([v['margin_to_background'] for v in items if v['margin_to_background'] is not None])}
        summary[scale]['common_coverage'] = {}
        for t in (.5,.75,.9):
            common = [r for r in selected if all(r[a][str(t)]['rank'] is not None for a in data)]
            summary[scale]['common_coverage'][str(t)] = {'n':len(common),
                'SAM_minus_BOX_mean_rank':mean([r['SAM'][str(t)]['rank']-r['BOX'][str(t)]['rank'] for r in common]),
                'SAM_minus_BOX_mean_correct_score':mean([r['SAM'][str(t)]['score']-r['BOX'][str(t)]['score'] for r in common]),
                'SAM_minus_BOX_mean_background_score':mean([r['SAM']['highest_background_score']-r['BOX']['highest_background_score'] for r in common if all(r[a]['highest_background_score'] is not None for a in data)])}
    result = {'sources':{k:str(v) for k,v in sources.items()}, 'summary':summary, 'rows':rows,
              'caveats':['different best epochs; exploratory single seed', 'GT used only for output diagnosis',
                         'background means IoU<0.1 under visible labels; not direct feature measurement',
                         'conditional scores and ranks do not equal AP; common coverage excludes missing candidates']}
    (ROOT / 'reports/148_sgc2_half_box/shape_target_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
