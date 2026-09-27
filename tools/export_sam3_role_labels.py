"""Export all completed SAM3 teacher masks to the existing detector data interface."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from sam3_teacher_pilot import dump,iou
from audit_sam3_positive_edges import bounds,mask_iou

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/104_sam3_role_control'


def main():
    raw=REPORT/'teacher_raw'
    manifest=json.loads((raw/'manifest.json').read_text(encoding='utf-8'))
    assert len(manifest['images'])==3043
    summary=json.loads((raw/'summary.json').read_text(encoding='utf-8'))
    assert summary['complete']
    data=json.loads((ROOT/'data/antiuav6k_common/annotations/instances_visible_common_train.json').read_text(encoding='utf-8'))
    ids={im['id'] for im in manifest['images']}
    target=REPORT/'masks_train'
    (target/'masks').mkdir(parents=True,exist_ok=True)
    records=[]
    for im in manifest['images']:
        x,y,w,h=im['annotations'][0]['bbox'];box=[x,y,x+w,y+h]
        with np.load(raw/'masks'/f"{im['id']:06d}.npz") as archive:
            masks=archive['masks'];available=archive['available']
        assert masks.shape==(3,im['height'],im['width']) and masks.dtype==np.bool_
        stability=min(mask_iou(masks[0],m) for m in masks[1:])
        overlap=iou(bounds(masks[0]),box)
        accepted=bool(all(available) and overlap>=.5 and stability>=.7)
        record=dict(image_id=im['id'],file_name=im['file_name'],accepted=accepted,
                    supervision_weight=float(accepted),mask_bbox_iou=overlap,prompt_mask_stability=stability,
                    bbox_xyxy_annotation=box,quality_note='heuristic label usability, NOT pixel accuracy or calibrated SAM confidence')
        destination=target/'masks'/f"{im['id']:06d}.png"
        if destination.exists():
            assert np.array_equal(np.asarray(Image.open(destination))>0,masks[0])
        else: Image.fromarray(masks[0].astype(np.uint8)*255).save(destination)
        records.append(record)
    records.extend(dict(image_id=im['id'],file_name=im['file_name'],accepted=False,supervision_weight=0.,empty_gt=True)
                   for im in data['images'] if im['id'] not in ids)
    records.sort(key=lambda r:r['image_id'])
    assert len(records)==3200 and len(set(r['image_id'] for r in records))==3200
    dump(target/'records.json',records)
    stats=dict(images=len(records),positives=len(ids),accepted=sum(r['accepted'] for r in records),empty=sum(r.get('empty_gt',False) for r in records))
    dump(target/'export_summary.json',stats)
    print(stats,flush=True)


if __name__=='__main__': main()
