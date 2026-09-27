"""Audit original Anti-UAV rectangles; build an isolated corrected IR label set."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path('E:/two_paper')
DATA = Path('F:/data/Anti-UAV/Anti_UAV_6K')
RAW = Path('F:/data/Anti-UAV/dataset')
OUT = ROOT/'data/antiuav6k_ir_raw_verified'
REPORT = ROOT/'reports/158_mbudet_rgb_alignment'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), 'No overwrite of corrected labels'
    cache = {}; manifests = {}; plans = {}; statistics = {}
    examples = []
    for split in ('train', 'val', 'test'):
        visible = json.loads((ROOT/f'data/antiuav6k_common/annotations/instances_visible_common_{split}.json').read_text(encoding='utf-8'))
        rgb_anns = {a['image_id']: a for a in visible['annotations']}
        old_ir = json.loads((ROOT/f'data/antiuav6k_ir/annotations/instances_infrared_{split}.json').read_text(encoding='utf-8'))
        assert {i['file_name'] for i in old_ir['images']} == {i['file_name'] for i in visible['images']}
        counts = Counter(); images = []; anns = []; labels = {}; deltas = []; rows = []
        for im in visible['images']:
            name = im['file_name']; seq, frame = Path(name).stem.rsplit('_', 1); frame = int(frame)
            record = {}
            for modality in ('infrared', 'visible'):
                path = RAW/split/seq/(modality+'.json')
                if path not in cache:
                    cache[path] = json.loads(path.read_text(encoding='utf-8'))
                    manifests[str(path)] = sha(path)
                value = cache[path]
                rect = np.asarray(value['gt_rect'][frame], dtype=float)
                positive = bool(value['exist'][frame]) and len(rect) == 4 and bool((rect[2:] > 0).all())
                with Image.open(DATA/split/modality/'images'/name) as image:
                    width, height = image.size
                record[modality] = (rect, positive, width, height)
            rect, positive, width, height = record['infrared']
            assert (width, height) == (640, 512)
            images.append(dict(id=im['id'], file_name=name, width=width, height=height))
            old_label = DATA/split/'infrared/labels'/(Path(name).stem+'.txt')
            manifests[str(old_label)] = sha(old_label)
            lines = [l for l in old_label.read_text(encoding='utf-8').splitlines() if l.strip()]
            assert len(lines) == int(positive), (split, name, 'exist mismatch')
            if positive:
                observed = np.array(list(map(float, lines[0].split()[1:5])))
                x, y, w, h = rect
                wrong = rect/np.array([width, height, width, height])
                correct = np.array([x+w/2, y+h/2, w, h])/np.array([width, height, width, height])
                # Establish the actual source convention across every positive.
                assert np.max(np.abs(observed-wrong)) < 1e-7, (split, name, observed, wrong)
                assert np.max(np.abs(observed-correct)) > 1e-7
                counts['ir_positive_with_top_left_used_as_center'] += 1
                lo = np.maximum(rect[:2], 0); hi = np.minimum(rect[:2]+rect[2:], [width, height])
                assert (hi > lo).all(), name
                box = np.concatenate((lo, hi-lo)).tolist()
                anns.append(dict(id=len(anns)+1, image_id=im['id'], category_id=0, bbox=box,
                                 area=box[2]*box[3], iscrowd=0))
                clipped_center = (lo+hi)/2
                clipped_extent = hi-lo
                yolo = np.concatenate((clipped_center, clipped_extent))/np.array([width,height,width,height])
                labels[Path(name).stem] = '0 '+' '.join(f'{v:.10f}' for v in yolo)+'\n'
                deltas.append(float(np.linalg.norm(np.array([w/2, h/2]))))
                if split == 'train':
                    rows.append(dict(name=name, raw_rect=rect.tolist(), corrected_box=box,
                                     old_yolo=observed.tolist(), corrected_yolo=yolo.tolist()))
            else:
                labels[Path(name).stem] = ''
                counts['ir_negative'] += 1
            rect_r, positive_r, width_r, height_r = record['visible']
            ann_r = rgb_anns.get(im['id'])
            assert bool(ann_r) == positive_r, (split, name, 'RGB existence')
            if positive_r:
                lo = np.maximum(rect_r[:2], 0); hi = np.minimum(rect_r[:2]+rect_r[2:], [width_r, height_r])
                expected = np.concatenate((lo, hi-lo))*np.array([im['width']/width_r, im['height']/height_r]*2)
                assert np.max(np.abs(np.array(ann_r['bbox'])-expected)) < .001, (split, name, 'RGB source mismatch')
                counts['rgb_boxes_match_raw_rect'] += 1
            counts['images'] += 1
        plans[split] = dict(coco=dict(info=dict(description='Anti-UAV IR verified directly against original gt_rect XYWH',
            reference_modality='infrared', source=str(RAW), old_labels_preserved=True), images=images,
            annotations=anns, categories=[dict(id=0, name='drone')]), labels=labels)
        statistics[split] = dict(counts= dict(counts), old_center_error_pixels=dict(
            mean=float(np.mean(deltas)), p50=float(np.median(deltas)), p95=float(np.quantile(deltas,.95)), max=float(max(deltas))))
        if split == 'train':
            examples = [rows[i] for i in np.random.default_rng(158).choice(len(rows), 5, replace=False)]
    # Writes only after every split has passed the raw-source convention audit.
    for split, plan in plans.items():
        folder = OUT/split/'labels'; folder.mkdir(parents=True, exist_ok=False)
        ann_path = OUT/'annotations'/f'instances_infrared_{split}.json'; ann_path.parent.mkdir(parents=True, exist_ok=True)
        ann_path.write_text(json.dumps(plan['coco'], ensure_ascii=False, indent=2), encoding='utf-8')
        for stem, value in plan['labels'].items():
            (folder/(stem+'.txt')).write_text(value, encoding='utf-8')
    # Original labels and JSON are read-only throughout.
    assert all(sha(Path(path)) == expected for path, expected in manifests.items())
    REPORT.mkdir(parents=True, exist_ok=True)
    report = dict(status='PASS', bug_confirmed=True,
        bug_zh='源IR YOLO的前两项实际是归一化左上角，却被所有YOLO读取器当作中心。',
        statistics=statistics, examples=examples, corrected_root=str(OUT), source_hashes=manifests,
        rgb_protocol_unchanged=True, original_assets_unchanged=True,
        limitations_zh='源标注坐标正确性按原始gt_rect核验；不保证原始人工标注逐像素完美。旧IR指标与新正确协议不直接比较，不能将全部M失败归因于此。')
    (REPORT/'raw_ir_label_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    fig, axes = plt.subplots(5, 2, figsize=(12, 19))
    for row, ex in enumerate(examples):
        with Image.open(DATA/'train/infrared/images'/ex['name']) as image:
            pixels=np.array(image.convert('RGB')); width,height=image.size
        v=np.array(ex['old_yolo'])*np.array([width,height,width,height])
        old=[v[0]-v[2]/2,v[1]-v[3]/2,v[2],v[3]]
        for col, (box, title) in enumerate([(old,'Old interpreted YOLO box'),(ex['corrected_box'],'Original gt_rect verified box')]):
            axes[row,col].imshow(pixels)
            axes[row,col].add_patch(Rectangle(box[:2],box[2],box[3],fill=False,edgecolor='lime',linewidth=1.5))
            axes[row,col].set_title(title+' | '+ex['name'],fontsize=8); axes[row,col].axis('off')
    fig.tight_layout(); fig.savefig(REPORT/'ir_label_before_after_5.png',dpi=140); plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_hashes','examples')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
