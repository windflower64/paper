"""Per-image scale-normalized SAM grouping; no inference parameters."""
from .sam_group_contrast import selection, contrast


def multiscale_group_loss(features, targets, weights):
    if set(features) != set(weights) or not features or any(w <= 0 for w in weights.values()):
        raise ValueError('Every selected scale requires a positive weight')
    if any(len(f) != len(targets) for f in features.values()):
        raise ValueError('Feature/target batch mismatch')
    total = sum(f.float().sum()*0 for f in features.values())
    counts = {s: 0 for s in features}
    active_images = 0
    for i, target in enumerate(targets):
        terms = []
        for scale, feature in features.items():
            choices = {a: selection(target, feature.shape[-2:], a) for a in ('sam', 'box')}
            if any(c is None or len(c[0]) < 2 or len(c[1]) < 4 for c in choices.values()):
                continue
            value, _ = contrast(feature[i], choices['sam'])
            terms.append((weights[scale], value))
            counts[scale] += 1
        if terms:
            total = total + sum(w*v for w,v in terms)/sum(w for w,v in terms)
            active_images += 1
    return total/max(1, active_images), {'active_images': active_images, 'scale_eligible_images': counts}
