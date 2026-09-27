"""Immutable four-arm queue; strict best-EMA re-evaluation after every arm."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('E:/two_paper')
REPO = ROOT/'D-FINE'
REPORT = ROOT/'reports/157_sam_rgb_retention_training'
SNAPSHOT = REPORT/'frozen_repo'
ARMS = ('sam_frozen', 'c_frozen', 'sam_joint', 'c_joint')
STATE = {'completed': []}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(**values):
    STATE.update(values, updated_at=datetime.datetime.now().isoformat())
    (REPORT/'queue_status.json').write_text(json.dumps(STATE, ensure_ascii=False, indent=2), encoding='utf-8')


def env():
    values = {k.upper(): v for k, v in os.environ.items()}
    values.update(PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', CUDA_MODULE_LOADING='LAZY', TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='100')
    return values


def execute(command, out, err, **values):
    with out.open('x', encoding='utf-8') as stdout, err.open('x', encoding='utf-8') as stderr:
        process = subprocess.Popen(command, cwd=REPO, env=env(), stdout=stdout, stderr=stderr,
            stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        save(pid=process.pid, command=[str(c) for c in command], console_log=str(out), **values)
        code = process.wait()
    if code:
        raise RuntimeError(f'Exit {code}, inspect {err}')


def main():
    preflight = json.loads((REPORT/'preflight.json').read_text(encoding='utf-8'))
    assert preflight['status'] == 'PASS' and len(preflight['runs']) == 4
    batch = preflight['selected_batch']
    assert shutil.disk_usage(ROOT).free > 25*2**30
    runs = {arm: ROOT/f'outputs/S_RGB_RETENTION_{arm.upper()}_B{batch}A{32//batch}_10E_TESTDEV/seed0'
            for arm in ARMS}
    assert not any(path.exists() for path in runs.values()), 'Existing output, no overwrite/resume'
    SNAPSHOT.mkdir(exist_ok=False)
    manifest = {}
    for folder, pattern in (('src', '*.py'), ('configs', '*.yml'), ('experiments', '*.yml'), ('tools', '*.py')):
        for source in (REPO/folder).rglob(pattern):
            destination = SNAPSHOT/source.relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest[str(destination)] = sha(destination)
            assert sha(source) == manifest[str(destination)]
    for source in (
        ROOT/'outputs/C_ONLY_GQ1_B8A4_20E_TESTDEV/seed0/best_stg1.pth',
        ROOT/'outputs/S_SGC2_SAM_DECAY9_14_B8A4_20E_TESTDEV/seed0/best_stg1.pth',
        ROOT/'weights/m_sd2_joint_coco_thermal_identity_init.pth',
        ROOT/'data/antiuav6k_common/annotations/instances_visible_common_train.json',
        ROOT/'data/antiuav6k_common/annotations/instances_visible_common_test.json',
        REPORT/'preflight.json',
    ):
        manifest[str(source)] = sha(source)
    (REPORT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    save(status='snapshot_complete', batch=batch, effective_batch=32, accumulation=32//batch,
         snapshot=str(SNAPSHOT))
    results = {}
    for arm in ARMS:
        for name, expected in manifest.items():
            assert sha(Path(name)) == expected, f'Protected asset changed: {name}'
        run = runs[arm]; run.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, '-u', str(SNAPSHOT/'tools/run_sam_rgb_retention.py'),
            '--config', str(SNAPSHOT/'experiments/phase_s/sam_rgb_retention_10e.yml'),
            '--arm', arm, '--run', str(run), '--batch', str(batch)]
        execute(command, run/'train_console.log', run/'train_error.log', status='training', arm=arm,
            output_dir=str(run), metric_log=str(run/'log.txt'), progress=str(run/'training_progress.json'))
        execute(command+['--eval'], run/'evaluation_console.log', run/'evaluation_error.log',
            status='evaluating', arm=arm)
        rows = [json.loads(s) for s in (run/'log.txt').read_text().splitlines() if s.strip()]
        assert [r['epoch'] for r in rows] == list(range(10))
        best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
        results[arm] = dict(best_epoch=best['epoch'], best_metrics=best['test_coco_eval_bbox'],
            last3_mean=[sum(r['test_coco_eval_bbox'][i] for r in rows[-3:])/3 for i in range(12)],
            all_epochs=[dict(epoch=r['epoch'], metrics=r['test_coco_eval_bbox']) for r in rows])
        (REPORT/'comparison.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        STATE['completed'].append(arm)
        save(status='arm_complete', pid=None)
    lines = ['# RGB＋SAM能力保留受控训练结果', '',
        '已有RGB模型20轮预训练，新增M训练10轮；原test1820作为开发验证集。',
        '下表为单种子描述性结果，不自动宣称梯度冲突、边缘恢复或SAM独特性。', '',
        '| 组别 | 最佳轮 | AP | AP75 | APS | 末3轮AP | 末3轮APS |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm, value in results.items():
        b, t = value['best_metrics'], value['last3_mean']
        lines.append(f"| {arm} | {value['best_epoch']} | {100*b[0]:.4f} | {100*b[2]:.4f} | {100*b[3]:.4f} | {100*t[0]:.4f} | {100*t[3]:.4f} |")
    lines += ['', '判断SAM是否保留，必须同时比较同训练方式下SAM−C及相同轮次曲线；',
              '冻结优于联合本身不能归因于SAM，也不能证明M结构错误。']
    (REPORT/'四组结果自动汇总.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save(status='complete', pid=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    if args.detach:
        assert not (REPORT/'queue_status.json').exists() and not (REPORT/'queue.lock').exists()
        with (REPORT/'queue_stdout.log').open('x', encoding='utf-8') as out, (REPORT/'queue_stderr.log').open('x', encoding='utf-8') as err:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve())],
                cwd=REPO, env=env(), stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        print(json.dumps(dict(queue_pid=process.pid)), flush=True)
    else:
        with (REPORT/'queue.lock').open('x', encoding='utf-8') as lock:
            lock.write(str(os.getpid()))
        try:
            main()
        except Exception as error:
            save(status='failed', pid=None, error=str(error)); raise
