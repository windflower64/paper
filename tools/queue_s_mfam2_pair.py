"""Frozen SAM/BoxMask pair for S-MFAM2. Separate processes, no run overwrite."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT.parent/'reports/99_s_mfam2'


def status(**data):
    data['updated_at']=datetime.datetime.now().isoformat()
    (REPORT/'queue_status.json').write_text(json.dumps(data,indent=2),encoding='utf-8')


def main():
    checks=[json.loads((REPORT/f'preflight_{arm}.json').read_text()) for arm in ('SAM','BOX')]
    assert checks[0]['initial_model_sha256']==checks[1]['initial_model_sha256']
    for check in checks:
        assert check['status']=='PASS' and check['baseline_disabled_error']==0
        assert min(check['detector_head_gradients'])>0
        assert check['physical_batch']==8 and check['gradient_accumulation_steps']==4
    checkpoint=ROOT.parent/'weights/m_sd2_joint_coco_thermal_identity_init.pth'
    assert checkpoint.is_file()
    sources=[ROOT/'src/zoo/dfine'/name for name in
             ('sam_mask_aggregation.py','sam_support_shape.py','dfine.py','dfine_criterion.py')]
    sources += [ROOT/'tools/preflight_s_mfam2.py',Path(__file__),ROOT/'tests/test_s_mfam2.py',
                ROOT/'src/data/dataset/coco_dataset.py',ROOT/'src/data/dataloader.py',
                ROOT/'src/data/transforms/_transforms.py',ROOT/'src/solver/det_engine.py',
                ROOT/'src/solver/_solver.py']
    sources += [ROOT/'experiments/phase_s'/name for name in
                ('s_mfam2_c_sam_b8a4_20e_testdev_local.yml','s_mfam2_c_box_b8a4_20e_testdev_local.yml',
                 's_mfam1_c_sam_b8a4_20e_testdev_local.yml')]
    sources += [ROOT/'experiments/phase_m/c_only_gq1_b8a4_20e_testdev_local.yml',
                REPORT/'preflight_SAM.json',REPORT/'preflight_BOX.json']
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    for arm in ('SAM','BOX'):
        run=ROOT.parent/f'outputs/S_MFAM2_C_{arm}_B8A4_20E_TESTDEV/seed0'
        if run.exists() and any(run.iterdir()):raise RuntimeError(f'Nonempty output: {run}')
    env=os.environ.copy()
    env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',CUDA_MODULE_LOADING='LAZY',
               TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT='100')
    for arm in ('SAM','BOX'):
        for path,digest in hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                status(status='blocked_source_changed',arm=arm,path=path)
                raise RuntimeError('Source changed during paired experiment')
        run=ROOT.parent/f'outputs/S_MFAM2_C_{arm}_B8A4_20E_TESTDEV/seed0'
        artifacts=run/'artifacts';artifacts.mkdir(parents=True,exist_ok=False)
        for path in sources:shutil.copy2(path,artifacts/path.name)
        manifest=dict(hashes)
        manifest[str(checkpoint)]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        (artifacts/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        config=ROOT/f'experiments/phase_s/s_mfam2_c_{arm.lower()}_b8a4_20e_testdev_local.yml'
        command=[sys.executable,'-u','train.py','-c',str(config),'-t',str(checkpoint),'--seed','0','--use-amp']
        with (run/'train_console.log').open('x',encoding='utf-8') as out, (run/'train_error.log').open('x',encoding='utf-8') as err:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err)
            (run/'train.pid').write_text(str(process.pid),encoding='ascii')
            status(status='running',arm=arm,pid=process.pid,run=str(run))
            code=process.wait()
        if code:
            status(status='failed',arm=arm,exit_code=code)
            raise RuntimeError(f'{arm} failed: {code}')
        logs=[json.loads(line) for line in (run/'log.txt').read_text().splitlines() if line.strip()]
        if [row['epoch'] for row in logs]!=list(range(20)):
            status(status='incomplete',arm=arm);raise RuntimeError('Incomplete epochs')
    status(status='completed',arms=['SAM','BOX'])


if __name__=='__main__':main()
