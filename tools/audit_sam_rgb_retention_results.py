"""Audit completed report157; cached metrics only, no detector mutation."""
import json
from pathlib import Path

ROOT = Path('E:/two_paper')
REPORT = ROOT/'reports/157_sam_rgb_retention_training'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def delta(a, b):
    assert len(a) == len(b) == 12
    return [100*(x-y) for x, y in zip(a, b)]


def main():
    state = load(REPORT/'queue_status.json')
    assert state['status'] == 'complete' and len(state['completed']) == 4
    comparison = load(REPORT/'comparison.json')
    initial = {s: load(REPORT/'initial_validation'/s/'verified.json')['metrics'] for s in ('sam', 'c')}
    results = {}
    for arm, value in comparison.items():
        run = ROOT/f'outputs/S_RGB_RETENTION_{arm.upper()}_B32A1_10E_TESTDEV/seed0'
        verified = load(run/'independent_best_ema.json')
        fixed = load(run/'final_fixed_state_check.json')
        off = load(run/'same_weight_m_off.json')
        assert verified['status'] == fixed['status'] == 'PASS'
        assert verified['images'] == off['images'] == 1820
        assert verified['max_abs_error'] == 0 and fixed['actual_updates'] == 1000
        assert [r['epoch'] for r in value['all_epochs']] == list(range(10))
        assert verified['metrics'] == value['best_metrics']
        source = initial[arm.split('_')[0]]
        if arm.endswith('_frozen'):
            assert off['metrics'] == source
        results[arm] = dict(best_epoch=value['best_epoch'], best_metrics=value['best_metrics'],
            initial_metrics=source, m_off_metrics=off['metrics'],
            best_minus_initial_pp=delta(value['best_metrics'], source),
            m_on_minus_off_pp=delta(value['best_metrics'], off['metrics']),
            m_off_minus_initial_pp=delta(off['metrics'], source),
            last3_metrics=value['last3_mean'], last3_minus_initial_pp=delta(value['last3_mean'], source))
    paired = {}
    for mode in ('frozen', 'joint'):
        sam, c = comparison['sam_'+mode], comparison['c_'+mode]
        curves = [dict(epoch=a['epoch'], delta_pp=delta(a['metrics'], b['metrics']))
                  for a, b in zip(sam['all_epochs'], c['all_epochs'])]
        paired[mode] = dict(same_epoch_sam_minus_c=curves,
            positive_ap_epochs=sum(r['delta_pp'][0] > 0 for r in curves),
            positive_aps_epochs=sum(r['delta_pp'][3] > 0 for r in curves),
            last3_sam_minus_c_pp=delta(sam['last3_mean'], c['last3_mean']))
    result = dict(status='complete', completed_at=state['updated_at'], images=1820,
        metric_order=['AP', 'AP50', 'AP75', 'APS', 'APM', 'APL', 'AR1', 'AR10', 'AR100', 'ARS', 'ARM', 'ARL'],
        arms=results, paired=paired,
        last3_sam_aps_advantage_frozen_minus_joint_pp=(
            paired['frozen']['last3_sam_minus_c_pp'][3]-paired['joint']['last3_sam_minus_c_pp'][3]),
        conclusions=[
            '固定组SAM相对C的APS优势10/10轮为正，末3轮优势0.7752点；联合组末3轮0.2999点。',
            '这是已有SAM相关优势保留，不是本轮M带来的新增提升；四组最佳AP均未超过各自RGB起点。',
            '固定组M旁路完整复现RGB源，联合组最佳权重旁路M仍比起点差；参数状态变化与推理影响均需区分。',
            '缺少相同协议的RGB单独继续训练对照，不能把联合退化归因于M反向或特定梯度冲突。',
            '不续训本轮四组，不替换历史最佳完整RGB-T模型，不宣布边缘恢复或SAM精细形状独特性。',
        ])
    (REPORT/'conclusion.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], paired=paired,
        retention_difference_pp=result['last3_sam_aps_advantage_frozen_minus_joint_pp']), ensure_ascii=False))


if __name__ == '__main__':
    main()
