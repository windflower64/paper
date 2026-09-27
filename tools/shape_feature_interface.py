"""Experiment-only masked selection of fixed external SAM3 feature targets."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import tv_tensors

from src.core import register
from src.data.dataset.coco_dataset import RGBTCocoDetection

ROOT = Path('E:/two_paper')
ARMS = ('none', 'uniform', 'filled', 'sam')


def schedule(epoch):
    return 1. if epoch < 6 else max(0., (10-epoch)/5.)


def roi_grid(box, size):
    a = (torch.arange(size, device=box.device, dtype=torch.float32)+.5)/size
    x = box[0] + a*(box[2]-box[0])
    y = box[1] + a*(box[3]-box[1])
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack((xx, yy), -1)[None]*2-1


def rectangle_occupancy(roi, extent, size=32):
    """Exact fractional coverage of normalized rectangular ROI cells."""
    a = torch.arange(size+1,device=roi.device,dtype=torch.float32)/size
    x = roi[0]+a*(roi[2]-roi[0]); y = roi[1]+a*(roi[3]-roi[1])
    dx = (torch.minimum(x[1:],extent[2])-torch.maximum(x[:-1],extent[0])).clamp_min(0)
    dy = (torch.minimum(y[1:],extent[3])-torch.maximum(y[:-1],extent[1])).clamp_min(0)
    return (dy[:,None]*dx[None,:])/((x[1]-x[0])*(y[1]-y[0]))


def common_weights(target):
    if len(target['boxes']) != 1 or float(target['sam_quality'][0]) <= 0:
        return None
    roi, extent = target['kd_roi'].float(), target['kd_extent'].float()
    shape = F.grid_sample(target['masks'][:1,None].float(),roi_grid(roi,32),
                          mode='bilinear',padding_mode='zeros',align_corners=False)[0,0].clamp(0,1)
    filled = rectangle_occupancy(roi,extent).clamp(0,1)
    if min(float(shape.sum()),float((1-shape).sum()),
           float(filled.sum()),float((1-filled).sum())) < 1e-4:
        return None
    # All variants have total weight one; selection alters where, not count scaling.
    weights = {'uniform':torch.full_like(shape,1/shape.numel())}
    for name, m in (('filled',filled),('sam',shape)):
        weights[name] = .5*m/m.sum()+.5*(1-m)/(1-m).sum()
    return weights, shape, filled


@register()
class ShapeFeatureRGBT(RGBTCocoDetection):
    __inject__ = ['transforms']

    def __init__(self,img_folder,ann_file,transforms,infrared_folder,
                 teacher_root,infrared_label_folder=None,infrared_index_offset=0,
                 return_masks=False,remap_mscoco_category=False,sam_mask_root=None):
        super().__init__(img_folder,ann_file,transforms,infrared_folder,
            infrared_label_folder,infrared_index_offset,return_masks,
            remap_mscoco_category,sam_mask_root)
        self.teacher_root = Path(teacher_root)
        geometry = json.loads((self.teacher_root/'geometry.json').read_text(encoding='utf-8'))
        self.teacher_rows = {r['image_id']:r for r in geometry['rows']}
        names = [type(t).__name__ for t in self._transforms.transforms]
        assert names == ['RandomPhotometricDistort','RandomHorizontalFlip','Resize',
                         'SanitizeBoundingBoxes','ConvertPILImage','ConvertBoxes'], names
        assert self.infrared_index_offset == 0

    def load_item(self,idx):
        image, target = super().load_item(idx)
        h,w = image.height,image.width
        anchor = torch.zeros((len(target['boxes']),h,w),dtype=torch.uint8)
        anchor[:,:,:w//2] = 1
        target['kd_flip_anchor'] = tv_tensors.Mask(anchor)
        return image,target

    def __getitem__(self,idx):
        samples,target = super().__getitem__(idx)
        anchor = target.pop('kd_flip_anchor')
        flipped = len(anchor) > 0 and not bool(anchor[0,0,0])
        image_id = int(target['image_id'][0])
        target['kd_flipped'] = torch.tensor([flipped],dtype=torch.bool)
        if image_id in self.teacher_rows and len(target['boxes']) == 1:
            row = self.teacher_rows[image_id]
            roi = torch.tensor(row['roi'],dtype=torch.float32)
            extent = torch.tensor(row['extent'],dtype=torch.float32)
            if flipped:
                for box in (roi,extent):
                    left,right = box[0].item(),box[2].item()
                    box[0],box[2] = 1-right,1-left
            target['kd_roi'],target['kd_extent'] = roi,extent
            if self.epoch < 10:
                teacher = np.load(self.teacher_root/'features'/f'{image_id:06d}.npy',allow_pickle=False)
                assert teacher.shape == (256,32,32) and teacher.dtype == np.float16
                target['kd_teacher'] = torch.from_numpy(teacher.copy())
                if flipped:
                    target['kd_teacher'] = target['kd_teacher'].flip(-1)
        return samples,target


class FeatureDetector(nn.Module):
    def __init__(self,detector,arm,normalization,log_path=None):
        super().__init__()
        assert arm in ARMS and not detector.sgc_enabled
        assert detector.backbone.return_idx == [2,3]
        assert detector.encoder.in_channels[0] == 512
        self.detector = detector
        self.kd_projection = nn.Conv2d(512,256,1,bias=False)
        self.register_buffer('teacher_mean',torch.tensor(normalization['mean']).view(1,256,1,1))
        self.register_buffer('teacher_std',torch.tensor(normalization['std']).view(1,256,1,1))
        self.arm,self.epoch,self.calls = arm,0,0
        self.log_path = str(log_path) if log_path else None
        self.captured = None
        self.detector.backbone.register_forward_hook(self._capture)

    def _capture(self,module,inputs,output):
        if self.training:
            self.captured = output[0]

    def set_training_epoch(self,epoch):
        self.epoch = int(epoch)
        self.detector.set_training_epoch(epoch)

    def forward(self,samples,targets=None):
        outputs = self.detector(samples,targets)
        source,self.captured = self.captured,None
        if not self.training or targets is None:
            return outputs
        strength = schedule(self.epoch)
        telemetry = dict(epoch=self.epoch,call=self.calls,arm=self.arm,batch=len(targets),
                         schedule=strength,valid=0,raw_mse=0.,weighted_loss=0.)
        loss = samples.new_zeros((),dtype=torch.float32)
        if strength > 0:
            assert source is not None and source.shape[1] == 512
            with torch.autocast(samples.device.type,enabled=False):
                choices = [common_weights(t) for t in targets]
                telemetry['valid'] = sum(c is not None for c in choices)
                if self.arm != 'none':
                    # Per-location normalization has no learned parameters.
                    normalized = F.layer_norm(source.float().permute(0,2,3,1),(512,)).permute(0,3,1,2)
                    projected = self.kd_projection(normalized)
                    for i,(target,choice) in enumerate(zip(targets,choices)):
                        if choice is None:
                            continue
                        pred = F.grid_sample(projected[i:i+1],roi_grid(target['kd_roi'],32),
                                             padding_mode='border',align_corners=False)
                        teacher = (target['kd_teacher'][None].float()-self.teacher_mean)/self.teacher_std
                        error = (pred-teacher).square().mean(1)[0]
                        loss = loss+(error*choice[0][self.arm]).sum()/len(targets)
                    telemetry['raw_mse'] = float(loss.detach())
                    loss = loss*strength  # locked initial KD weight = 1.0
        telemetry['weighted_loss'] = float(loss.detach())
        outputs['feature_kd_loss'] = loss
        if self.log_path:
            with Path(self.log_path).open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(telemetry)+'\n')
        if self.calls % 25 == 0:
            print('FEATURE_KD',json.dumps(telemetry),flush=True)
        self.calls += 1
        return outputs


class FeatureCriterion(nn.Module):
    def __init__(self,criterion):
        super().__init__()
        self.detector_criterion = criterion

    def forward(self,outputs,targets,**metas):
        losses = self.detector_criterion(outputs,targets,**metas)
        if 'feature_kd_loss' in outputs:
            losses['loss_feature_kd'] = outputs['feature_kd_loss']
        return losses
