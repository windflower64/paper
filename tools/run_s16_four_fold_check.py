"""Bounded sequential virtual parameter checks; reuse exact fold0 pilot."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
EXPORT=ROOT/'reports/151_sam_internal_shape/none_backbone/scale_audit/gradient_export'
OUT=ROOT/'reports/152_s16_shape_finite_step/four_fold_parameter'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old=torch.load(EXPORT/'fixed_reader.pt',weights_only=False,map_location='cpu')
    new=torch.load(EXPORT/'four_folds/fold0/fixed_reader.pt',weights_only=False,map_location='cpu')
    assert [r['image_id'] for r in old['rows']]==[r['image_id'] for r in new['rows']]
    assert all(np.array_equal(old[k],new[k]) for k in ('mu','std','beta'))
    paths=[ROOT/'reports/152_s16_shape_finite_step/attempt03_parameter/finite_step.json']
    for fold in (1,2,3):
        dest=OUT/f'fold{fold}';dest.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(ROOT/'D-FINE/tools/probe_s16_shape_finite_step.py'),'--parameter-space',
                 '--reader-path',str(EXPORT/f'four_folds/fold{fold}/fixed_reader.pt'),'--output',str(dest)]
        (OUT/'status.json').write_text(json.dumps({'status':'running','fold':fold,'detector_training':False}),encoding='utf-8')
        with (dest/'stdout.log').open('w',encoding='utf-8') as stdout,(dest/'stderr.log').open('w',encoding='utf-8') as stderr:
            result=subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr)
        if result.returncode:
            (OUT/'status.json').write_text(json.dumps({'status':'failed','fold':fold,'returncode':result.returncode}),encoding='utf-8')
            raise RuntimeError(f'fold{fold} failed; inspect logs, no automatic retry')
        paths.append(dest/'finite_step.json')
        print('completed_fold',fold,flush=True)
    details=[]
    for fold,path in enumerate(paths):
        d=json.loads(path.read_text(encoding='utf-8'))
        assert d['status']=='virtual_parameter_steps_restored_exact' and d['optimizer_steps']==0
        for batch in d['batches']:
            details.append({'fold':fold,**batch})
    summary={}
    for scale in ('small','other'):
        summary[scale]={}
        batches=[b for b in details if b['scale']==scale]
        for trial in ('SAM/raw','SAM/protected','filled_extent/raw','filled_extent/protected','scrambled/raw','scrambled/protected'):
            summary[scale][trial]={'fold_batches':len(batches),'images':sum(len(b['image_ids']) for b in batches),
                'deltas_by_fold':[{'fold':b['fold'],'n':len(b['image_ids']),**b['trials'][trial]['delta_losses'],
                                  'SAM_mse':b['trials'][trial]['delta_real_SAM_mse']} for b in batches],
                'all_three_GT_losses_decreased':sum(all(v<0 for v in b['trials'][trial]['delta_losses'].values()) for b in batches),
                'real_SAM_mse_decreased':sum(b['trials'][trial]['delta_real_SAM_mse']<0 for b in batches)}
    (OUT/'summary.json').write_text(json.dumps({'status':'complete_no_committed_training','fold0_exact_reader_reused':True,
        'paths':[str(p) for p in paths],'summary':summary,'batches':details,
        'caveats':['exploratory expansion after fold0 pilot; fixed amplitude, no threshold search',
                   'folds are grouped readout controls, not independent detector seeds',
                   'each batch has aggregated GT loss; no per-image outcome or AP claim']},indent=2),encoding='utf-8')
    (OUT/'status.json').write_text(json.dumps({'status':'complete','detector_training':False,'committed_optimizer_steps':0}),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
