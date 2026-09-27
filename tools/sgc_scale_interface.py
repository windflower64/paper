"""Experimental training-only coverage control, not a production model module.

Keep the original S8 objective unchanged. Add S4 only for common SAM/BOX
eligible images that S8 would skip; normalize the addition by physical batch.
"""
from src.zoo.dfine.sam_group_contrast import selection, contrast, group_loss


def common_choices(target, size):
    choices = {arm: selection(target, size, arm) for arm in ('sam', 'box')}
    valid = all(ids is not None and len(ids[0]) >= 2 and len(ids[1]) >= 4
                for ids in choices.values())
    return choices if valid else None


def scale_losses(s4, s8, targets, arm):
    if arm not in ('sam', 'box'):
        raise ValueError(arm)
    if s4.shape[0] != s8.shape[0] or len(targets) != s8.shape[0]:
        raise ValueError('Batch dimensions disagree')
    old = group_loss(s8, targets, arm)
    rescue = s4.float().sum() * 0.
    routes = []
    for i, target in enumerate(targets):
        if common_choices(target, s8.shape[-2:]) is not None:
            routes.append('S8')
            continue
        choices = common_choices(target, s4.shape[-2:])
        if choices is None:
            routes.append('skip')
            continue
        value, _ = contrast(s4[i], choices[arm])
        rescue = rescue + value
        routes.append('S4_rescue')
    rescue = rescue / max(1, len(targets))
    return {'s8': old, 'rescue': rescue, 'total': old + rescue}, routes
