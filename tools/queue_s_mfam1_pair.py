"""Sequential SAM/BoxMask experiment queue; never overwrite previous runs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
REPORT = WORKSPACE/'reports/98_s_mfam1'
ARMS = [
    ('SAM', 's_mfam1_c_sam_b8a4_20e_testdev_local.yml'),
    ('BOX', 's_mfam1_c_box_b8a4_20e_testdev_local.yml'),
]


def write_status(**data):
    data['updated_at'] = datetime.datetime.now().isoformat()
    (REPORT/'queue_status.json').write_text(json.dumps(data, indent=2), encoding='utf-8')


def main():
    preflight = json.loads((REPORT/'preflight.json').read_text(encoding='utf-8'))
    assert preflight['status'] == 'PASS'
    assert preflight['baseline_disabled_error'] == 0
    assert preflight['detector_only_mask_gradient'] > 0
    assert preflight['physical_batch'] == 8 and preflight['gradient_accumulation_steps'] == 4
    checkpoint = WORKSPACE/'weights/m_sd2_joint_coco_thermal_identity_init.pth'
    assert checkpoint.is_file()
    for arm, config_name in ARMS:
        run = WORKSPACE/f'outputs/S_MFAM1_C_{arm}_B8A4_20E_TESTDEV/seed0'
        if run.exists() and any(run.iterdir()):
            raise RuntimeError(f'Refusing to overwrite nonempty run: {run}')
        assert (ROOT/'experiments/phase_s'/config_name).is_file()
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
               CUDA_MODULE_LOADING='LAZY', TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='100')
    for arm, config_name in ARMS:
        config = ROOT/'experiments/phase_s'/config_name
        run = WORKSPACE/f'outputs/S_MFAM1_C_{arm}_B8A4_20E_TESTDEV/seed0'
        artifacts = run/'artifacts'
        artifacts.mkdir(parents=True, exist_ok=False)
        files = [
            config, ROOT/'src/zoo/dfine/sam_mask_aggregation.py',
            ROOT/'src/zoo/dfine/dfine.py', ROOT/'src/zoo/dfine/dfine_criterion.py',
            ROOT/'src/data/dataset/coco_dataset.py', ROOT/'tests/test_s_mfam.py',
            ROOT/'tools/preflight_s_mfam1.py', Path(__file__), REPORT/'preflight.json',
            ROOT/'experiments/phase_s/s_mfam1_c_sam_b8a4_20e_testdev_local.yml',
            ROOT/'experiments/phase_m/c_only_gq1_b8a4_20e_testdev_local.yml',
            ROOT/'experiments/phase_m/m_control_gq1_sameaug_b16_60e_local.yml',
        ]
        manifest = {}
        for path in dict.fromkeys(files):
            shutil.copy2(path, artifacts/path.name)
            manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[str(checkpoint)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        (artifacts/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        command = [sys.executable,'-u','train.py','-c',str(config),'-t',str(checkpoint),
                   '--seed','0','--use-amp']
        with (run/'train_console.log').open('x',encoding='utf-8') as out, \
             (run/'train_error.log').open('x',encoding='utf-8') as err:
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err)
            (run/'train.pid').write_text(str(process.pid),encoding='ascii')
            write_status(status='running',arm=arm,pid=process.pid,run=str(run),command=command)
            result = process.wait()
        if result:
            write_status(status='failed',arm=arm,exit_code=result,run=str(run))
            raise RuntimeError(f'{arm} training failed; queue stopped ({result})')
        logs = [json.loads(line) for line in (run/'log.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
        if not logs or logs[-1].get('epoch') != 19:
            write_status(status='incomplete',arm=arm,run=str(run))
            raise RuntimeError('Training returned without completing epoch 19')
    write_status(status='completed',arms=[arm for arm, _ in ARMS])


if __name__ == '__main__':
    main()
