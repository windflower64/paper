"""Train-only two-side label/augmentation audit and random pair overlays."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('E:/two_paper')
OUT = ROOT/'reports/158_mbudet_rgb_alignment'
sys.path.insert(0, str(REPO))
from src.core import YAMLConfig


def quantiles(values):
    a = np.asarray(values)
    return dict(count=len(a), mean=float(a.mean()), p50=float(np.quantile(a, .5)),
                p90=float(np.quantile(a, .9)), p95=float(np.quantile(a, .95)), max=float(a.max()))


def normalized(ann, image):
    x, y, w, h = ann['bbox']
    return np.array([(x+w/2)/image['width'], (y+h/2)/image['height'],
                     w/image['width'], h/image['height']])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'corrected_training_pair_audit.json').exists()
    raw = {}
    for modality, path in [('rgb', ROOT/'data/antiuav6k_common/annotations/instances_visible_common_train.json'),
                           ('ir', ROOT/'data/antiuav6k_ir_raw_verified/annotations/instances_infrared_train.json')]:
        data = json.loads(path.read_text(encoding='utf-8'))
        images = {i['id']: i for i in data['images']}
        annotations = defaultdict(list)
        for ann in data['annotations']:
            im = images[ann['image_id']]
            value = normalized(ann, im)
            assert np.isfinite(value).all() and (value[2:] > 0).all()
            assert (value[:2] >= 0).all() and (value[:2] <= 1).all()
            annotations[im['file_name']].append(value)
        assert all(len(a) <= 1 for a in annotations.values())
        raw[modality] = (data, images, annotations)
    rgb, ir = raw['rgb'][2], raw['ir'][2]
    names = sorted(i['file_name'] for i in raw['rgb'][0]['images'])
    assert set(names) == set(i['file_name'] for i in raw['ir'][0]['images']) and len(names) == 3200
    counts = Counter(); shifts = []; mismatches = []; paired = []; clipped_labels = 0
    for name in names:
        r, t = rgb.get(name, []), ir.get(name, [])
        counts['both_positive' if r and t else 'rgb_only' if r else 'ir_only' if t else 'both_negative'] += 1
        label = ROOT/'data/antiuav6k_ir_raw_verified/train/labels'/(Path(name).stem+'.txt')
        assert label.is_file(), label
        parsed = [np.array(list(map(float, line.split()[1:5]))) for line in label.read_text(encoding='utf-8').splitlines() if line.strip()]
        assert len(parsed) == len(t), name
        if t:
            value = parsed[0]
            lo = np.clip(value[:2]-value[2:]/2, 0, 1)
            hi = np.clip(value[:2]+value[2:]/2, 0, 1)
            clipped = np.concatenate(((lo+hi)/2, hi-lo))
            clipped_labels += float(np.abs(clipped-value).max()) > 1e-6
            mismatch = float(np.abs(clipped-t[0]).max())
            mismatches.append(mismatch)
        if r and t:
            shifts.append((t[0][:2]-r[0][:2])*np.array([640, 512]))
            paired.append(name)
    assert max(mismatches, default=0) < 1e-4
    cfg = YAMLConfig(str(REPO/'experiments/phase_m/mbudet_rgb_reference_20e.yml'))
    dataset = cfg.train_dataloader.dataset
    torch.manual_seed(0)
    checked = []; flip_count = 0
    for index in np.random.default_rng(20260918).choice(len(dataset), 32, replace=False):
        samples, target = dataset[int(index)]
        name = Path(target['image_path']).name
        assert samples.shape == (6, 512, 640)
        observed_r = target['boxes'].numpy()
        observed_t = target['infrared_boxes'].numpy()
        expected_r = np.asarray(rgb.get(name, [])).reshape(-1, 4)
        expected_t = np.asarray(ir.get(name, [])).reshape(-1, 4)
        # The loader clips IR boxes to the canvas before augmentation.
        if len(expected_t):
            lo = np.clip(expected_t[:, :2]-expected_t[:, 2:]/2, 0, 1)
            hi = np.clip(expected_t[:, :2]+expected_t[:, 2:]/2, 0, 1)
            expected_t = np.concatenate(((lo+hi)/2, hi-lo), 1)
        flipped_r, flipped_t = expected_r.copy(), expected_t.copy()
        if len(flipped_r): flipped_r[:, 0] = 1-flipped_r[:, 0]
        if len(flipped_t): flipped_t[:, 0] = 1-flipped_t[:, 0]
        def err(a, b):
            assert a.shape == b.shape
            return float(np.abs(a-b).max()) if a.size else 0.
        normal = max(err(observed_r, expected_r), err(observed_t, expected_t))
        flipped = max(err(observed_r, flipped_r), err(observed_t, flipped_t))
        assert min(normal, flipped) < 2e-5, (name, normal, flipped)
        flip_count += flipped < normal
        checked.append(dict(name=name, normalized_box_error=min(normal, flipped)))
    selected = np.random.default_rng(158).choice(paired, 5, replace=False).tolist()
    fig, axes = plt.subplots(5, 2, figsize=(12, 19))
    for row, name in enumerate(selected):
        for col, (folder, annotations, title) in enumerate([
            (ROOT/'data/antiuav6k_common/images/train', rgb, 'RGB reference'),
            (Path('F:/data/Anti-UAV/Anti_UAV_6K/train/infrared/images'), ir, 'IR source')]):
            with Image.open(folder/name) as image:
                image = image.convert('RGB'); width, height = image.size
                axes[row, col].imshow(image)
            cx, cy, bw, bh = annotations[name][0]*np.array([width, height, width, height])
            axes[row, col].add_patch(Rectangle((cx-bw/2, cy-bh/2), bw, bh, fill=False, edgecolor='lime', linewidth=1.3))
            axes[row, col].set_title(title+' | '+name, fontsize=8); axes[row, col].axis('off')
    fig.tight_layout(); fig.savefig(OUT/'corrected_random_training_pairs_5.png', dpi=140); plt.close(fig)
    delta = np.asarray(shifts)
    report = dict(status='PASS', split='train only', images=3200, presence=dict(counts),
        ir_coco_vs_original_yolo_max_normalized_error=max(mismatches),
        ir_boundary_clipped_labels=int(clipped_labels),
        augmentation_checked=checked, synchronized_flips_observed=int(flip_count),
        displacement_pixels=dict(distance=quantiles(np.linalg.norm(delta, axis=1)),
            abs_x=quantiles(np.abs(delta[:, 0])), abs_y=quantiles(np.abs(delta[:, 1]))),
        random_overlay_samples=selected,
        limits_zh='文件名、双侧框来源一致性和同步几何变换已核对；不等于全量人工核验或真实像素对应。')
    (OUT/'corrected_training_pair_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='augmentation_checked'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
