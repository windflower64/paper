"""Frozen SAM3 image features, sampled locally; no detector or teacher updates."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def roi_grid(box, size, device):
    a = (torch.arange(size, device=device, dtype=torch.float32) + .5) / size
    x = box[0] + a * (box[2] - box[0])
    y = box[1] + a * (box[3] - box[1])
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((xx, yy), -1).unsqueeze(0) * 2 - 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    args.output = args.output.resolve()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    sys.path.insert(0, str(ROOT / '_third_party/sam3'))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    masks = ROOT / 'reports/104_sam3_role_control/masks_train'
    records = json.loads((masks / 'records.json').read_text(encoding='utf-8'))
    selected = [r for r in records if r['accepted']]
    assert len(selected) == 2987
    metadata = dict(format_version=1, expected=2987, feature_index=1, roi_size=32,
                    extent_expansion=1.5, teacher_resolution=1008,
                    precision='bfloat16 forward; float16 local cache',
                    source='frozen SAM3 image backbone, no text/GT fed to encoder',
                    selection='existing accepted GT-prompt masks; training privilege',
                    records_sha256=hashlib.sha256((masks / 'records.json').read_bytes()).hexdigest(),
                    rows=[])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'features').mkdir(exist_ok=True)
    # Build geometry deterministically from immutable existing masks.
    for r in selected:
        with Image.open(masks / 'masks' / f"{r['image_id']:06d}.png") as im:
            mask = np.asarray(im) > 0
        yy, xx = np.nonzero(mask)
        assert len(xx)
        h, w = mask.shape
        bounds = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
        cx, cy = (bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2
        rw, rh = (bounds[2]-bounds[0])*1.5, (bounds[3]-bounds[1])*1.5
        roi = [max(0., cx-rw/2)/w, max(0., cy-rh/2)/h,
               min(float(w), cx+rw/2)/w, min(float(h), cy+rh/2)/h]
        metadata['rows'].append(dict(image_id=r['image_id'], file_name=r['file_name'],
            width=w, height=h, extent=[bounds[0]/w,bounds[1]/h,bounds[2]/w,bounds[3]/h], roi=roi))
    meta_path = args.output / 'geometry.json'
    if meta_path.exists():
        assert json.loads(meta_path.read_text(encoding='utf-8')) == metadata
    else:
        write_json(meta_path, metadata)
    model = build_sam3_image_model(checkpoint_path=str(ROOT/'weights/sam3/sam3.pt'),
                                  load_from_HF=False, compile=False)
    model.eval().requires_grad_(False)
    processor = Sam3Processor(model)
    shapes = None
    count = 0
    started = time.time()
    for row in metadata['rows']:
        path = args.output / 'features' / f"{row['image_id']:06d}.npy"
        if path.exists():
            existing = np.load(path)
            assert existing.shape == (256,32,32) and existing.dtype == np.float16
            assert np.isfinite(existing).all()
            continue
        if args.limit and count >= args.limit:
            break
        with Image.open(ROOT/'data/antiuav6k_common/images/train'/row['file_name']) as im:
            rgb = im.convert('RGB')
        assert rgb.size == (row['width'], row['height'])
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            state = processor.set_image(rgb)
            maps = state['backbone_out']['backbone_fpn']
            current = [list(t.shape) for t in maps]
            assert len(maps) >= 2 and maps[1].shape[1] == 256, current
            if shapes is None:
                shapes = current
                write_json(args.output/'interface.json', dict(native_shapes=shapes,
                    selected_index=1, selected_shape=current[1], cache_shape=[256,32,32],
                    torch=torch.__version__, device=torch.cuda.get_device_name(),
                    interpolation='bilinear align_corners=False; normalized original-image ROI',
                    note='32x32 is sampled grid, not 32x32 independent native observations'))
                print('NATIVE_FEATURES', current, flush=True)
            assert shapes == current
            with torch.autocast('cuda', enabled=False):
                feature = F.grid_sample(maps[1].float(), roi_grid(row['roi'],32,'cuda'),
                                        mode='bilinear',padding_mode='border',align_corners=False)
            array = feature[0].cpu().numpy().astype(np.float16)
            assert array.shape == (256,32,32) and np.isfinite(array).all()
            # Exclusive writes; an interrupted partial cache is detected on restart.
            with path.open('xb') as f:
                np.save(f, array, allow_pickle=False)
            del state, maps, feature
        count += 1
        if count <= 3 or count % 25 == 0:
            completed = len(list((args.output/'features').glob('*.npy')))
            write_json(args.output/'status.json', dict(status='exporting',completed=completed,
                expected=2987,seconds=time.time()-started,detector_training=False))
            print(f'EXPORTED {completed}/2987 seconds={time.time()-started:.1f}',flush=True)
    files = list((args.output/'features').glob('*.npy'))
    complete = len(files) == 2987
    if complete:
        sums = torch.zeros(256,dtype=torch.float64)
        squares = torch.zeros(256,dtype=torch.float64)
        pixels = 0
        hashes = {}
        for row in metadata['rows']:
            path = args.output/'features'/f"{row['image_id']:06d}.npy"
            array = np.load(path)
            assert array.shape == (256,32,32) and np.isfinite(array).all()
            values = torch.from_numpy(array.astype(np.float64)).flatten(1)
            sums += values.sum(1); squares += values.square().sum(1); pixels += values.shape[1]
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        mean = sums/pixels
        std = (squares/pixels-mean.square()).clamp_min(0).sqrt().clamp_min(.01)
        write_json(args.output/'normalization.json', dict(mean=mean.tolist(),std=std.tolist(),
            source='training accepted local feature grids only',pixels=pixels,std_floor=.01))
        write_json(args.output/'feature_hashes.json',hashes)
    write_json(args.output/'status.json', dict(status='complete' if complete else 'limited_probe',
        completed=len(files),expected=2987,seconds=time.time()-started,detector_training=False))
    print('EXPORT_END',len(files),complete,flush=True)


if __name__ == '__main__':
    main()
