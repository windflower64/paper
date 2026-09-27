"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import random
from copy import deepcopy
from functools import partial

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as VT
from torch.utils.data import default_collate
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as VF

from ..core import register

torchvision.disable_beta_transforms_warning()


__all__ = [
    "DataLoader",
    "BaseCollateFunction",
    "BatchImageCollateFunction",
    "DenseO2OCollateFunction",
    "batch_image_collate_fn",
]


@register()
class DataLoader(data.DataLoader):
    __inject__ = ["dataset", "collate_fn"]

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ["dataset", "batch_size", "num_workers", "drop_last", "collate_fn"]:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), "shuffle must be a boolean"
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image"""
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1

    def __call__(self, items):
        raise NotImplementedError("")


def generate_scales(base_size, base_size_repeat):
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


@register()
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self,
        stop_epoch=None,
        ema_restart_decay=0.9999,
        base_size=640,
        base_size_repeat=None,
    ) -> None:
        super().__init__()
        self.base_size = base_size
        self.scales = (
            generate_scales(base_size, base_size_repeat) if base_size_repeat is not None else None
        )
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.ema_restart_decay = ema_restart_decay
        # self.interpolation = interpolation

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        if self.scales is not None and self.epoch < self.stop_epoch:
            # sz = random.choice(self.scales)
            # sz = [sz] if isinstance(sz, int) else list(sz)
            # VF.resize(inpt, sz, interpolation=self.interpolation)

            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if "masks" in targets[0]:
                for tg in targets:
                    tg["masks"] = F.interpolate(tg["masks"], size=sz, mode="nearest")
                raise NotImplementedError("")

        return images, targets


@register()
class DenseO2OCollateFunction(BatchImageCollateFunction):
    """DEIM batch-level MixUp isolated from the default D-FINE collator.

    Existing experiments keep using ``BatchImageCollateFunction``.  L-DQ1
    opts into this class explicitly, so the Dense O2O port cannot silently
    change earlier S/C/SC protocols.
    """

    def __init__(
        self,
        stop_epoch=None,
        ema_restart_decay=0.9999,
        base_size=640,
        base_size_repeat=None,
        mixup_prob=0.0,
        mixup_epochs=(0, 0),
    ) -> None:
        super().__init__(
            stop_epoch=stop_epoch,
            ema_restart_decay=ema_restart_decay,
            base_size=base_size,
            base_size_repeat=base_size_repeat,
        )
        if len(mixup_epochs) != 2:
            raise ValueError("mixup_epochs must contain [start, stop]")
        self.mixup_prob = float(mixup_prob)
        self.mixup_epochs = tuple(int(value) for value in mixup_epochs)
        if not 0.0 <= self.mixup_prob <= 1.0:
            raise ValueError("mixup_prob must be between 0 and 1")

    def apply_mixup(self, images, targets):
        if (
            len(targets) < 2
            or self.mixup_prob <= 0
            or not self.mixup_epochs[0] <= self.epoch < self.mixup_epochs[1]
            or random.random() >= self.mixup_prob
        ):
            return images, targets

        beta = round(random.uniform(0.45, 0.55), 6)
        shifted_images = images.roll(shifts=1, dims=0)
        mixed_images = shifted_images.mul(1.0 - beta).add(images.mul(beta))
        shifted_targets = targets[-1:] + targets[:-1]
        mixed_targets = deepcopy(targets)

        for index, (target, shifted) in enumerate(zip(targets, shifted_targets)):
            for key in ("boxes", "labels", "area", "iscrowd"):
                if key in target and key in shifted:
                    mixed_targets[index][key] = torch.cat((target[key], shifted[key]), dim=0)
            mixed_targets[index]["mixup"] = torch.tensor(
                [beta] * len(target["labels"])
                + [1.0 - beta] * len(shifted["labels"]),
                dtype=torch.float32,
            )

        return mixed_images, mixed_targets

    def __call__(self, items):
        images = torch.cat([item[0][None] for item in items], dim=0)
        targets = [item[1] for item in items]
        images, targets = self.apply_mixup(images, targets)

        if self.scales is not None and self.epoch < self.stop_epoch:
            size = random.choice(self.scales)
            images = F.interpolate(images, size=size)
            if "masks" in targets[0]:
                for target in targets:
                    target["masks"] = F.interpolate(target["masks"], size=size, mode="nearest")
                raise NotImplementedError("")

        return images, targets
