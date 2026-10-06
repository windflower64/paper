"""Inventory saved scale-study candidates; reuse audited metrics, no new evaluation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/SCALE_CANDIDATE_SELECTION_20261006'
SPECS = [
    ('Replacement GT-only', 'SBR3_REPLACE_20261004', 'R1_GT_ONLY'),
    ('Replacement SAM', 'SBR3_REPLACE_20261004', 'R2_SAM'),
    ('SGC S8', 'SGC_MULTISCALE_JOINT_20261005', 'S8'),
    ('SGC multiscale', 'SGC_MULTISCALE_JOINT_20261005', 'MULTISCALE'),
    ('SBR3 detail control', 'SBR3_CONTEXT_20261005', 'DETAIL'),
    ('SBR3 S16 on', 'SBR3_CONTEXT_20261005', 'CONTEXT'),
    ('M original', 'M_P32_20261005', 'ORIGINAL'),
    ('M P32', 'M_P32_20261005', 'P32'),
    ('Combination S8', 'SCALE_COMBINATION_20261005', 'S8'),
    ('Combination multiscale', 'SCALE_COMBINATION_20261005', 'MULTISCALE'),
]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    results = []
    for name, run, arm in SPECS:
        directory = ROOT / 'outputs' / run / arm
        rows = [json.loads(line) for line in (directory / 'log.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        best = max(rows, key=lambda row: row['validation']['coco_eval_bbox'][0])
        checkpoint = directory / 'best.pth'
        assert checkpoint.is_file(), checkpoint
        results.append(dict(name=name, run=run, arm=arm, epoch=best['epoch'],
                            checkpoint=str(checkpoint), sha256=sha(checkpoint),
                            mode='normal', metrics=best['validation']['coco_eval_bbox'],
                            source=str(directory / 'log.jsonl')))
    dependence = ROOT / 'reports/SBR3_CONTEXT_20261005/context_dependence.json'
    audit = json.loads(dependence.read_text(encoding='utf-8'))['best']
    parent = next(row for row in results if row['name'] == 'SBR3 S16 on')
    assert parent['sha256'] == audit['checkpoint_sha'], 'Intervention checkpoint changed'
    results.append(dict(parent, name='SBR3 S16 training / score off at inference',
                        mode='zero entire context_score output; not zero S16 input',
                        metrics=audit['context_disabled']['coco_eval_bbox'], source=str(dependence)))
    results.sort(key=lambda row: row['metrics'][0], reverse=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'manifest.json').write_text(json.dumps(dict(
        scope='SBR3 replacement and scale studies only; not all project history',
        evidence='Existing logs and prior evaluation; no fresh GPU evaluation in this inventory',
        split='1820-image repeatedly used development set, not untouched test',
        candidates=results), ensure_ascii=False, indent=2), encoding='utf-8')
    indexes = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    lines = ['# 已保存候选的完整指标', '',
             '单位：%；APl、ARl 均为 -1（不可用）。行内指标均来自同一个最佳 AP 权重，不拼接各指标峰值。', '',
             '|候选|轮次|AP|AP50|AP75|APs|APm|AR1|AR10|AR100|ARs|ARm|',
             '|---|' + '---:|' * 11]
    for row in results:
        lines.append('|' + row['name'] + '|' + str(row['epoch'] + 1) + '|' + '|'.join(f"{100 * row['metrics'][i]:.4f}" for i in indexes) + '|')
    lines += ['', '权重路径、SHA256、推理模式和指标来源见同目录 manifest.json。跨训练协议的排序用于选择候选，不能直接归因到某一个模块。']
    (OUT / '完整指标.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
