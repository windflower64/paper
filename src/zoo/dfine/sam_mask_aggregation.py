"""MFAM-inspired detection adapter, not a complete MGNet reproduction.

Reference: https://arxiv.org/html/2605.25385v1, equations 9--13.
Only RGB coordinates are used. SAM is an offline training label provider.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def conv(in_channels, out_channels, kernel=3):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel,
                                   padding=kernel // 2, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=False))


class SAMMaskAggregation(nn.Module):
    def __init__(self, low_channels=256, high_channels=512, width=32):
        super().__init__()
        self.low_proj = conv(low_channels, width, 1)
        self.high_proj = conv(high_channels, width, 1)
        self.mask_head = nn.Sequential(conv(2 * width, width), nn.Conv2d(width, 1, 1))
        self.low_split = conv(2 * width, width)
        self.high_split = conv(2 * width, width)
        self.aggregate = conv(2 * width, width)
        self.output_proj = nn.Conv2d(width, high_channels, 1, bias=False)
        # Nonzero output lets detection gradients reach both mask and feature
        # paths immediately. Fixed scaling, not another learned near-zero gate.
        self.residual_scale = 0.1
        self.intervention = 'learned'

    def forward(self, low, high):
        if self.intervention == 'disabled':
            return high, None
        shallow = self.low_proj(low)
        deep = F.interpolate(self.high_proj(high), size=low.shape[-2:],
                             mode='bilinear', align_corners=False)
        logits = self.mask_head(torch.cat([shallow, deep], dim=1))
        mask = logits.sigmoid()
        if self.intervention == 'constant':
            mask = torch.full_like(mask, 0.5)
        elif self.intervention != 'learned':
            raise ValueError(self.intervention)
        shallow_split = self.low_split(torch.cat([mask * shallow, (1-mask) * shallow], 1))
        deep_split = self.high_split(torch.cat([mask * deep, (1-mask) * deep], 1))
        fused = self.aggregate(torch.cat([deep_split, shallow_split], 1)) + deep + shallow
        delta = self.output_proj(F.adaptive_avg_pool2d(fused, high.shape[-2:]))
        return high + self.residual_scale * delta, logits


def region_loss(logits, targets, source='sam'):
    """Area targets preserve fractional occupancy; rejected positives ignored.

    Box control shares SAM's image selection/weights, changing only shape.
    Empty annotated images remain valid all-background examples.
    """
    if source == 'none':
        return logits.float().sum() * 0.0
    if source not in ('sam', 'box', 'edge'):
        raise ValueError(source)
    logits = logits.float()
    total = logits.sum() * 0.0
    weights = logits.new_zeros(())
    for prediction, target in zip(logits, targets):
        if 'masks' not in target or 'sam_quality' not in target:
            raise RuntimeError('S-MFAM1 requires transformed SAM masks and sam_quality')
        masks = target['masks']
        empty = len(target['boxes']) == 0
        quality = 1.0 if empty else float(target['sam_quality'].item())
        if quality <= 0:
            continue
        height, width = masks.shape[-2:]
        union = prediction.new_zeros((height, width))
        if not empty:
            if source in ('sam', 'edge'):
                union = masks.to(prediction).amax(0)
                if source == 'edge':
                    # Extract at transformed mask resolution, before area resize.
                    # Keep fractional S8 occupancy; no low-resolution threshold.
                    field = union[None, None]
                    dilated = F.max_pool2d(field, 5, stride=1, padding=2)
                    eroded = -F.max_pool2d(-field, 5, stride=1, padding=2)
                    union = (dilated-eroded).clamp(0, 1)[0, 0]
            else:
                # Transformed training boxes are normalized cxcywh.
                for cx, cy, bw, bh in target['boxes'].detach().tolist():
                    x0 = max(0, int((cx - bw/2) * width))
                    y0 = max(0, int((cy - bh/2) * height))
                    x1 = min(width, int(math.ceil((cx + bw/2) * width)))
                    y1 = min(height, int(math.ceil((cy + bh/2) * height)))
                    union[y0:y1, x0:x1] = 1
        truth = F.interpolate(union[None, None], size=prediction.shape[-2:], mode='area')[0]
        bce = F.binary_cross_entropy_with_logits(prediction, truth, reduction='none')
        # Balance occupied/unoccupied area instead of letting tiny foreground
        # disappear in a uniform pixel mean. Dice only on positive images.
        if empty:
            item = bce.mean()
        else:
            foreground = (bce * truth).sum() / truth.sum().clamp_min(1)
            background = (bce * (1-truth)).sum() / (1-truth).sum().clamp_min(1)
            prob = prediction.sigmoid()
            dice = 1 - (2*(prob*truth).sum()+1)/(prob.sum()+truth.sum()+1)
            item = 0.5*(foreground+background) + dice
        total = total + quality * item
        weights = weights + quality
    return total / weights.clamp_min(1)
