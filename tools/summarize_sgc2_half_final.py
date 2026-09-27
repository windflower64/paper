"""Summarize fixed-protocol half-strength training, with exact EMA checks."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/112_sgc2_rgbt_half/final'
RUN = ROOT / 'outputs/C_PLUS_M_SD22_HALF_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0'


def summarize(name, path):
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    assert [r['epoch'] for r in rows] == list(range(20))
    assert all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v, (float, int)))
    assert all(math.isfinite(v) for r in rows for v in r['test_coco_eval_bbox'])
    best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
    return dict(name=name, log=str(path), best_epoch=best['epoch'],
                metrics=best['test_coco_eval_bbox'],
                tail5_AP=sum(r['test_coco_eval_bbox'][0] for r in rows[-5:])/5,
                final_metrics=rows[-1]['test_coco_eval_bbox'],
                best_telemetry={k:v for k,v in best.items() if 'msd2' in k},
                phase_AP={tag:sum(r['test_coco_eval_bbox'][0] for r in rows[a:b])/(b-a)
                          for tag,a,b in [('7-9',7,10),('10-13',10,14),('14-19',14,20)]},
                curve=[dict(epoch=r['epoch'],AP=r['test_coco_eval_bbox'][0]) for r in rows])


def main():
    previous = json.loads((ROOT/'reports/110_sgc2_rgbt_joint/final_comparison.json').read_text(encoding='utf-8'))
    runs = [summarize('C+M-half+SGC2-SAM', RUN/'log.txt')]
    runs += [summarize(r['name'], ROOT/r['log']) for r in previous['runs']]
    enabled = json.loads((OUT/'enabled.json').read_text(encoding='utf-8'))
    diagnosis = json.loads((OUT/'diagnostics/epoch8/summary.json').read_text(encoding='utf-8'))
    assert runs[0]['best_epoch'] == 8
    assert enabled['coco_eval_bbox'] == runs[0]['metrics'] == diagnosis['metrics']['1.0']
    assert diagnosis['normal_vs_log_max_error'] == 0
    assert diagnosis['configured_final_residual_scale'] == 0.5
    assert diagnosis['zero_equals_removal'] and diagnosis['images'] == 1820 and diagnosis['targets'] == 1691
    manifest = json.loads((RUN/'artifacts/manifest.json').read_text(encoding='utf-8'))
    sources = ['src/zoo/dfine/dfine.py', 'src/solver/det_engine.py',
               'experiments/phase_s/s_sgc2_sam_c_plus_m_sd22_half_b8a4_20e.yml']
    assert all(hashlib.sha256((ROOT/'D-FINE'/f).read_bytes()).hexdigest() == manifest[str(ROOT/'D-FINE'/f)] for f in sources)
    result = dict(status='complete', date='2026-09-09', runs=runs,
                  exact_standard_and_diagnostic_reproduction=True,
                  evaluated_final_residual_scale=0.5, images=1820,
                  best_checkpoint_sha256=hashlib.sha256((RUN/'best_stg1.pth').read_bytes()).hexdigest(),
                  diagnostics=diagnosis,
                  differences_ap_points={r['name']:dict(
                      AP=100*(runs[0]['metrics'][0]-r['metrics'][0]),
                      tail5=100*(runs[0]['tail5_AP']-r['tail5_AP'])) for r in runs[1:]},
                  decision='retain_half_joint_as_current_candidate_not_a_proven_SAM_increment',
                  caveats=['single-seed test-as-development; no statistical significance claim',
                           'different training runs best checkpoints are not fixed-weight interventions',
                           'C+M-half without SAM is the missing matched control',
                           'do not claim fewer false positives merely from higher AP'])
    (OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','differences_ap_points')},indent=2))
    print('ON_OFF',diagnosis['metrics'])
    print('FALSE',diagnosis['false_predictions'])


if __name__ == '__main__':
    main()
