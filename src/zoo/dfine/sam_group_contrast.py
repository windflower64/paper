"""Training-only local grouping objective, inspired by SAMFeat WSC.

No predicted mask, teacher feature input, or inference-time feature operation.
"""
import torch
from torch.nn import functional as F


def supervision_scale(epoch, decay_start=-1, decay_end=-1):
    """Return the training-only teacher scale for an optional linear retirement."""
    decay_start = int(decay_start)
    decay_end = int(decay_end)
    if decay_start == -1 and decay_end == -1:
        return 1.0
    if decay_start < 0 or decay_end < 0 or decay_start >= decay_end:
        raise ValueError(
            "SGC decay requires either (-1, -1) or 0 <= decay_start < decay_end"
        )
    epoch = int(epoch)
    if epoch <= decay_start:
        return 1.0
    if epoch >= decay_end:
        return 0.0
    return float(decay_end - epoch) / float(decay_end - decay_start)


def raster(boxes, height, width, expansion=1.):
    y = (torch.arange(height, device=boxes.device) + .5) / height
    x = (torch.arange(width, device=boxes.device) + .5) / width
    out = torch.zeros((height, width), device=boxes.device, dtype=torch.bool)
    for cx, cy, bw, bh in boxes:
        out |= ((y[:, None] - cy).abs() <= bh * expansion / 2) & (
            (x[None, :] - cx).abs() <= bw * expansion / 2)
    return out.float()


def selection(target, size, source):
    if source not in ('sam', 'box'):
        raise ValueError(source)
    masks = target['masks'].float()
    h, w = masks.shape[-2:]
    boxes = target['boxes']
    if len(boxes) != 1 or float(target['sam_quality']) <= 0:
        return None
    field = masks.amax(0) if source == 'sam' else raster(boxes, h, w)
    occupancy = F.interpolate(field[None, None], size=size, mode='area')[0, 0]
    region = raster(boxes, h, w, expansion=2.)
    local = F.interpolate(region[None, None], size=size, mode='area')[0, 0]
    positives = (occupancy >= .75).flatten().nonzero().flatten()
    negatives = ((occupancy <= .05) & (local >= .5)).flatten().nonzero().flatten()
    return positives, negatives


def choose(indices, maximum):
    if indices.numel() <= maximum:
        return indices
    return indices[torch.linspace(0, indices.numel()-1, maximum,
                                  device=indices.device).round().long()]


def contrast(feature, indices):
    positive, negative = indices
    if len(positive) < 2 or len(negative) < 4:
        return feature.float().sum() * 0., None
    # Explicit FP32 even if called within detector autocast.
    with torch.autocast(device_type=feature.device.type, enabled=False):
        tokens = F.normalize(feature.float().flatten(1).T, dim=-1)
        p = tokens[choose(positive, 32)]
        n = tokens[choose(negative, 64)]
        reference = F.normalize(p.sum(0, keepdim=True) - p, dim=-1)
        sim_positive = (p * reference).sum(-1)
        sim_negative = p @ n.T
        loss = F.softplus((sim_negative - sim_positive[:, None]) / .2).mean()
    return loss, dict(positive_cells=len(positive), negative_cells=len(negative),
                      separation=float((sim_positive.mean()-sim_negative.mean()).detach()))


def group_loss(features, targets, source):
    if source not in ('sam', 'box'):
        raise ValueError(source)
    total = features.float().sum()*0.
    count = 0
    for feature, target in zip(features, targets):
        # Identical eligible images in SAM and BOX arms, including at later epochs.
        choices = {a: selection(target, feature.shape[-2:], a) for a in ('sam', 'box')}
        if any(ind is None or len(ind[0]) < 2 or len(ind[1]) < 4 for ind in choices.values()):
            continue
        value, _ = contrast(feature, choices[source])
        total = total + value
        count += 1
    return total / max(1, count)
