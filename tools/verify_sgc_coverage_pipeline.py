"""Check standalone coverage against actual RGB/RGBT training transforms."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig
from src.zoo.dfine.sam_group_contrast import selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = args.report.resolve()
    report.relative_to((REPO.parent/'reports').resolve())
    destination = report/'pipeline_check.json'
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(4)
    rows = json.loads((report/'per_image.json').read_text(encoding='utf-8'))
    by_id = {r['image_id']: r for r in rows}
    rng = random.Random(20260913)
    selected_ids = []
    for group in ['lt8','8to16','16to24','24to32','32to96','ge96']:
        candidates = [r['image_id'] for r in rows if r.get('size_bin') == group]
        selected_ids.extend(rng.sample(candidates, min(8,len(candidates))))
    selected_ids.extend(rng.sample([r['image_id'] for r in rows if not r['objects']], 4))
    configs = [YAMLConfig(str(REPO/'experiments/phase_s'/name)) for name in
               ['s_sgc2_sam_decay9_14_b8a4_20e.yml',
                's_sgc2_sam_c_plus_m_sd22_half_b8a4_20e.yml']]
    data_configs = [c.yaml_cfg['train_dataloader'] for c in configs]
    assert data_configs[0]['dataset']['transforms'] == data_configs[1]['dataset']['transforms']
    assert data_configs[0]['collate_fn'] == data_configs[1]['collate_fn']
    assert data_configs[0]['collate_fn']['base_size_repeat'] is None
    records = []
    for label, cfg in zip(['C','C_plus_M'], configs):
        dataset = cfg.train_dataloader.dataset
        index = {image_id: i for i, image_id in enumerate(dataset.ids)}
        for image_id in selected_ids:
            # Same inputs into each actual pipeline; no detector is instantiated.
            random.seed(20260913+image_id)
            np.random.seed(20260913+image_id)
            torch.manual_seed(20260913+image_id)
            samples, target = dataset[index[image_id]]
            assert tuple(samples.shape[-2:]) == (512,640)
            r = by_id[image_id]
            result = {'pipeline': label, 'image_id': image_id, 'sample_shape': list(samples.shape)}
            choices = {s: selection(target,(64,80),s) for s in ['sam','box']}
            if not r.get('teacher_accepted'):
                assert all(v is None for v in choices.values())
                result['matched'] = True
            else:
                counts = {'sam_positive':len(choices['sam'][0]), 'sam_negative':len(choices['sam'][1]),
                          'box_positive':len(choices['box'][0]), 'box_negative':len(choices['box'][1])}
                matches = [orientation for orientation in ['S8','S8_flipped']
                           if all(r[orientation][k] == v for k,v in counts.items())]
                assert matches, (image_id, label, counts, r)
                result.update(counts=counts, matched=True, matched_orientations=matches)
            records.append(result)
    result = {'status':'pass', 'seed':20260913, 'sample_ids':selected_ids,
              'cases':len(records), 'same_transform_and_collate_config':True,
              'transforms':data_configs[0]['dataset']['transforms'],
              'collate':data_configs[0]['collate_fn'], 'records':records,
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'caveat':'sampled pipeline equivalence check, not historical random sequence replay'}
    with destination.open('x',encoding='utf-8') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'pass','cases':len(records),'images':len(selected_ids)}),flush=True)


if __name__ == '__main__':
    main()
