"""Training-only RGB feature preservation; no cross-modal spatial matching."""

import copy
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F


def checkpoint_weights(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("ema", {}).get("module") is not None:
        return state["ema"]["module"]
    return state["model"] if "model" in state else state


def load_rgb_features(model, checkpoint):
    weights = checkpoint_weights(checkpoint)
    for name in ("backbone", "encoder"):
        prefix = name + "."
        subset = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
        getattr(model, name).load_state_dict(subset, strict=True)


def box_occupancy(boxes, height, width, device):
    """Fractional cell coverage by normalized cxcywh boxes, including tiny boxes."""
    coverage = torch.zeros((height, width), device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32) / width
    ys = torch.arange(height, device=device, dtype=torch.float32) / height
    for box in boxes.detach().float():
        lo = (box[:2] - box[2:] / 2).clamp(0, 1)
        hi = (box[:2] + box[2:] / 2).clamp(0, 1)
        ox = (torch.minimum(xs + 1 / width, hi[0]) - torch.maximum(xs, lo[0])).clamp_min(0) * width
        oy = (torch.minimum(ys + 1 / height, hi[1]) - torch.maximum(ys, lo[1])).clamp_min(0) * height
        # Maximum avoids double counting overlapping boxes. Not an exact union
        # for multiple disjoint boxes sharing a cell; this is a weighting map.
        coverage = torch.maximum(coverage, oy[:, None] * ox[None, :])
    return coverage


def preservation_loss(student_levels, teacher_levels, targets):
    if len(student_levels) != len(teacher_levels):
        raise ValueError("teacher/student level count mismatch")
    terms = []
    for student, teacher in zip(student_levels, teacher_levels):
        if student.shape != teacher.shape:
            raise ValueError("teacher/student shape mismatch")
        if student.shape[0] != len(targets):
            raise ValueError("target batch mismatch")
        # Match channel-vector direction, not absolute feature magnitude.
        distance = (F.normalize(student.float(), dim=1) -
                    F.normalize(teacher.detach().float(), dim=1)).square().sum(1) / 2
        for index, target in enumerate(targets):
            fg = box_occupancy(target["boxes"], *distance.shape[-2:], student.device)
            regions = []
            for mask in (fg, 1 - fg):
                mass = mask.sum()
                if mass.item() > 0:
                    regions.append((distance[index] * mask).sum() / mass)
            terms.append(torch.stack(regions).mean())
    return torch.stack(terms).mean()


class RGBPreserver:
    """Kept outside the model, optimizer and EMA; capture pre-fusion encoder."""

    def __init__(self, student, spec, device):
        self.weight = float(spec.get("loss_weight", 1.0))
        if self.weight <= 0:
            raise ValueError("loss_weight must be positive")
        self.checkpoint = Path(spec["teacher_checkpoint"])
        # Deep copying deterministic modules does not consume RNG state.
        self.teacher = torch.nn.Module()
        self.teacher.backbone = copy.deepcopy(student.backbone)
        self.teacher.encoder = copy.deepcopy(student.encoder)
        load_rgb_features(self.teacher, self.checkpoint)
        self.teacher.to(device).eval().requires_grad_(False)
        self.cache = None
        self.hook = student.encoder.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        if module.training:
            self.cache = tuple(output)

    def __call__(self, samples, targets):
        if self.cache is None:
            raise RuntimeError("RGB encoder features were not captured")
        levels, self.cache = self.cache, None
        self.teacher.eval()
        with torch.no_grad(), torch.autocast(
            device_type=samples.device.type, enabled=samples.is_cuda,
            dtype=torch.float16,
        ):
            teacher_levels = self.teacher.encoder(self.teacher.backbone(samples[:, :3]))
        return self.weight * preservation_loss(levels, teacher_levels, targets)

    def describe(self):
        return {
            "teacher_checkpoint": str(self.checkpoint),
            "teacher_sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            "teacher_source": "EMA when available, otherwise model",
            "loss_weight": self.weight,
            "location": "RGB encoder outputs S16/S32 before M",
            "loss": "channel-direction distance, per-image balanced box/background",
            "sam_used": False, "inference_parameters_added": 0,
        }

    def close(self):
        self.hook.remove()
        self.cache = None
