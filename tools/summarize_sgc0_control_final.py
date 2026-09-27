"""Fixed half-strength, SAM versus zero auxiliary gradient comparison."""
import hashlib
import json
from pathlib import Path
from summarize_sgc2_half_final import summarize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports/113_sgc0_half_control/final'


def main():
    runs = []
    diagnostics = []
    for label, name, report, epoch in (
        ('without_SAM', 'C_PLUS_M_SD22_HALF_SGC0_B8A4_20E_TESTDEV', '113_sgc0_half_control', 13),
        ('with_SAM', 'C_PLUS_M_SD22_HALF_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV', '112_sgc2_rgbt_half', 8),
    ):
        run = ROOT/'outputs'/name/'seed0'
        result = summarize(label, run/'log.txt')
        result['checkpoint_sha256'] = hashlib.sha256((run/'best_stg1.pth').read_bytes()).hexdigest()
        p = ROOT/'reports'/report/'final'
        evaluation = json.loads((p/'enabled.json').read_text(encoding='utf-8'))
        diag = json.loads((p/f'diagnostics/epoch{epoch}/summary.json').read_text(encoding='utf-8'))
        assert result['metrics'] == evaluation['coco_eval_bbox'] == diag['metrics']['1.0']
        assert diag['normal_vs_log_max_error'] == 0 and diag['configured_final_residual_scale'] == 0.5
        targets = json.loads((p/f'diagnostics/epoch{epoch}/targets.json').read_text())
        tp = sum(r['1.0']['correct_score_50'] is not None and r['1.0']['correct_score_50'] >= .5 for r in targets)
        result['threshold_0_5'] = dict(TP=tp,FN=len(targets)-tp,
            FP=sum(diag['false_predictions']['all']['1.0']['0.5'].values()))
        runs.append(result); diagnostics.append(diag)
    a,b = runs
    labels = ['AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','ARS','ARM','ARL']
    delta = {k:100*(y-x) for k,x,y in zip(labels,a['metrics'],b['metrics'])}
    delta['tail5_AP'] = 100*(b['tail5_AP']-a['tail5_AP'])
    rows = [json.loads(s) for s in Path(a['log']).read_text().splitlines() if s.strip()]
    assert all(r.get('train_loss_sgc_group',0.) == 0. for r in rows)
    result = dict(status='complete',runs=runs,SAM_minus_control_ap_points=delta,
        diagnostics=diagnostics,zero_aux_all_epochs=True,
        same_epoch_SAM_AP_wins=sum(y['AP']>x['AP'] for x,y in zip(a['curve'],b['curve'])),
        decision='small_positive_detection_signal_not_evidence_of_edge_restoration',
        caveats=['single seed and test-as-development', 'epochs are correlated, not independent replicates',
                 'best checkpoints occur at different epochs; compare according to the same selection rule'])
    (OUT/'sam_attribution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'delta':delta,'threshold':[r['threshold_0_5'] for r in runs],
                      'geometry':[d['by_scale'] for d in diagnostics]},indent=2))


if __name__ == '__main__':
    main()
