"""Training-only S4 fallback; preserve the existing S8 SAM objective."""
from .sam_group_contrast import selection, contrast, group_loss


def common_choices(target, size):
    choices = {arm: selection(target, size, arm) for arm in ('sam', 'box')}
    valid = all(ids is not None and len(ids[0]) >= 2 and len(ids[1]) >= 4
                for ids in choices.values())
    return choices if valid else None


def rescue_loss(s4, s8, targets, arm):
    if arm not in ('sam', 'box'):
        raise ValueError(arm)
    if s4 is None or s4.shape[0] != s8.shape[0] or len(targets) != s8.shape[0]:
        raise ValueError('Missing S4 feature or inconsistent batch dimensions')
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
    return rescue / max(1, len(targets)), routes


def scale_losses(s4, s8, targets, arm):
    """Compatibility API for the report132 prototype, not the BOX training arm."""
    old = group_loss(s8, targets, arm)
    rescue, routes = rescue_loss(s4, s8, targets, arm)
    return {'s8': old, 'rescue': rescue, 'total': old + rescue}, routes
