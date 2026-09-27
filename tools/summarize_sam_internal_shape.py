"""Summarize fixed-support readout controls and diagnostic limits."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports/151_sam_internal_shape'


def main():
    paths = {'SAM': OUT/'internal_shape_readout.json',
             'NONE': OUT/'none_backbone/internal_shape_readout.json',
             'NONE_scale': OUT/'none_backbone/scale_audit/internal_shape_readout.json'}
    reports = {k: json.loads(p.read_text(encoding='utf-8')) for k,p in paths.items()}
    first = reports['SAM']
    assert all(d['selected']==first['selected'] and d['held_parent_groups']==first['held_parent_groups'] for d in reports.values())
    assert all(d['eligible']==600 and not d['excluded'] for d in reports.values())
    def value(d, feature, label='SAM', scale='small', fold=None):
        return d['results'][f'{feature}/{label}']['summary'][f'{scale}/fold{fold}']['inside_bacc']
    table = {name: {k: v['summary']['small/foldNone'] for k,v in d['results'].items()} for name,d in reports.items()}
    d = reports['NONE_scale']
    gains = {feature: {label: value(d, feature)-value(d, feature, label) for label in ('filled_extent','scrambled')} for feature in ('S4','S8','S16')}
    scale_drop = [value(d,'S8',fold=f)-value(d,'S16',fold=f) for f in range(4)]
    output = {'status': 'complete_no_detector_or_new_segmentation_network_training',
              'same_selection_groups': True, 'selected':600, 'small':300,
              'small_inside_bacc':table, 'NONE_real_label_minus_control':gains,
              'NONE_S8_minus_S16_by_fold':scale_drop,
              'NONE_S8_minus_S16_mean':value(d,'S8')-value(d,'S16'),
              'interpretation': 'Pseudo-shape labels are learnable beyond spatial prior; coarse-stage linear readout is worse. Does not prove edge restoration or detector improvement.',
              'next_stage': 'Design and deduplicate S16 shape-separation intervention; verify detection-compatible gradients before any matched detector training.',
              'not_proven': ['manual boundary correctness', 'new SAM benefit on frozen detector',
                             'unique causal benefit of fine contour for detection', 'information-theoretic feature loss',
                             'AP improvement', 'deployable mask-extent-free interface']}
    (OUT/'conclusion.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__': main()
