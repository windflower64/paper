"""Report final LQE gains and exact top-choice changes for each frozen model."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'reports/149_final_lqe_audit'


def main():
    results = {}
    for name in ('SAM','BOX','NONE'):
        summary = json.loads((ROOT/name/'summary.json').read_text())
        rows = json.loads((ROOT/name/'targets.json').read_text())
        values = {}
        for scale in ('small','medium'):
            selected = [r for r in rows if r['scale']==scale]
            values[scale] = {'n':len(selected),
                'top1_iou_changed':sum(r['normal']['top1_iou']!=r['without_final_lqe']['top1_iou'] for r in selected),
                'rank75_changed':sum(r['normal']['correct_rank_75']!=r['without_final_lqe']['correct_rank_75'] for r in selected),
                'rank90_changed':sum(r['normal']['correct_rank_90']!=r['without_final_lqe']['correct_rank_90'] for r in selected)}
        results[name] = {'final_lqe_gain_ap_points':{label:100*(summary['metrics']['normal'][i]-summary['metrics']['without_final_lqe'][i])
                            for label,i in (('AP',0),('AP50',1),('AP75',2),('APS',3),('AR100',8))},
                         'top_choice':values,'small_target':summary['small_target'],
                         'false_predictions':summary['false_predictions'],
                         'normal_vs_log_max_error':summary['normal_vs_log_max_error']}
    (ROOT/'conclusion.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
