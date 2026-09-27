"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T

from ...core import GLOBAL_CONFIG, register
from ._transforms import EmptyTransform

torchvision.disable_beta_transforms_warning()


@register()
class Compose(T.Compose):
    def __init__(self, ops, policy=None, mosaic_prob=-0.1) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop("type")
                    transform = getattr(
                        GLOBAL_CONFIG[name]["_pymodule"], GLOBAL_CONFIG[name]["_name"]
                    )(**op)
                    transforms.append(transform)
                    op["type"] = name

                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError("")
        else:
            transforms = [
                EmptyTransform(),
            ]

        super().__init__(transforms=transforms)

        if policy is None:
            policy = {"name": "default"}

        self.policy = policy
        self.mosaic_prob = float(mosaic_prob)
        self.global_samples = 0

    def forward(self, *inputs: Any) -> Any:
        return self.get_forward(self.policy["name"])(*inputs)

    def get_forward(self, name):
        forwards = {
            "default": self.default_forward,
            "stop_epoch": self.stop_epoch_forward,
            "stop_sample": self.stop_sample_forward,
        }
        return forwards[name]

    def default_forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def stop_epoch_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]
        cur_epoch = dataset.epoch
        policy_ops = self.policy["ops"]
        policy_epoch = self.policy["epoch"]

        # DEIM Dense O2O uses four augmentation stages.  Keep the original
        # scalar stop-epoch behavior untouched for all existing experiments.
        if isinstance(policy_epoch, list):
            if len(policy_epoch) != 3:
                raise ValueError("Dense O2O epoch policy must contain [start, mosaic_end, stop]")
            with_mosaic = (
                policy_epoch[0] <= cur_epoch < policy_epoch[1]
                and random.random() <= self.mosaic_prob
            )
            for transform in self.transforms:
                name = type(transform).__name__
                if name in policy_ops and cur_epoch < policy_epoch[0]:
                    continue
                if name in policy_ops and cur_epoch >= policy_epoch[-1]:
                    continue
                if name == "DenseO2OMosaic" and not with_mosaic:
                    continue
                if name in {"RandomZoomOut", "RandomIoUCrop"} and with_mosaic:
                    continue
                sample = transform(sample)
            return sample

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and cur_epoch >= policy_epoch:
                pass
            else:
                sample = transform(sample)

        return sample

    def stop_sample_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]

        cur_epoch = dataset.epoch
        policy_ops = self.policy["ops"]
        policy_sample = self.policy["sample"]

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and self.global_samples >= policy_sample:
                pass
            else:
                sample = transform(sample)

        self.global_samples += 1

        return sample
