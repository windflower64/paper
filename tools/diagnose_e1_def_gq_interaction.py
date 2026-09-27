#!/usr/bin/env python3
"""Diagnose the representation interface between DEF1@S16 and GQ1@S32.

This script is evaluation-only.  It never updates parameters and never uses
Test.  The strongest causal comparison is paired within the trained E1 model:
the same Val batch is forwarded with learned DGFE and with DGFE truly bypassed.
Standalone DEF1 and GQ1 are descriptive references only; their channel bases
were learned independently and are therefore not compared element by element.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--e1-config", type=Path, required=True)
    p.add_argument("--e1-checkpoint", type=Path, required=True)
    p.add_argument("--def1-config", type=Path, required=True)
    p.add_argument("--def1-checkpoint", type=Path, required=True)
    p.add_argument("--gq1-config", type=Path, required=True)
    p.add_argument("--gq1-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-batches", type=int)
    p.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    p.add_argument(
        "--e1-mode",
        choices=("student", "teacher"),
        default="student",
        help=(
            "Use the deployable soft student mask or the reconstruction-teacher "
            "hard mask for the paired E1 intervention."
        ),
    )
    return p.parse_args()


def load_model(repo, config_path, checkpoint_path, hgnet_overrides):
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(config_path))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    # YAMLConfig's registered component dictionaries are process-global.  This
    # diagnostic intentionally creates three HGNet variants in one process, so
    # explicitly reset every S/C switch to prevent the previous variant from
    # leaking into the next model instance.
    cfg.yaml_cfg["HGNetv2"].update(hgnet_overrides)
    model = cfg.model.cuda().eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = checkpoint.get("ema", {}).get("module")
    source = "ema.module"
    if weights is None:
        weights = checkpoint.get("model")
        source = "model"
    if weights is None:
        weights = checkpoint
        source = "raw"
    own = model.state_dict()
    compatible = {
        key: value
        for key, value in weights.items()
        if key in own and own[key].shape == value.shape
    }
    result = model.load_state_dict(compatible, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch for {config_path}: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    return cfg, model, {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "weight_source": source,
        "loaded_tensors": len(compatible),
    }


class FeatureCapture:
    def __init__(self, backbone, gq_type):
        self.backbone = backbone
        self.stage2 = 2
        self.stage3 = 3
        self.data = {}
        self.handles = []
        self.handles.append(
            backbone.stages[self.stage2].register_forward_hook(self._stage2_hook)
        )
        self.handles.append(
            backbone.stages[self.stage3].register_forward_pre_hook(self._stage3_pre_hook)
        )
        self.handles.append(
            backbone.stages[self.stage3].register_forward_hook(self._stage3_hook)
        )
        self.gq_names = []
        for name, module in backbone.named_modules():
            if isinstance(module, gq_type):
                self.gq_names.append(name)
                self.handles.append(
                    module.register_forward_pre_hook(self._make_gq_pre(name))
                )
                self.handles.append(
                    module.register_forward_hook(self._make_gq_post(name))
                )

    def clear(self):
        self.data.clear()

    def _stage2_hook(self, module, inputs, output):
        self.data["s16_raw"] = output.detach()

    def _stage3_pre_hook(self, module, inputs):
        self.data["s16_to_s32"] = inputs[0].detach()

    def _stage3_hook(self, module, inputs, output):
        self.data["s32_out"] = output.detach()

    def _make_gq_pre(self, name):
        def hook(module, inputs):
            self.data[f"gq_input::{name}"] = inputs[0].detach()

        return hook

    def _make_gq_post(self, name):
        def hook(module, inputs, output):
            self.data[f"gq_output::{name}"] = output.detach()

        return hook

    def snapshot(self):
        return {key: value.detach().float().clone() for key, value in self.data.items()}

    def close(self):
        for handle in self.handles:
            handle.remove()


def batch_ranks(values):
    order = values.argsort(dim=1)
    ranks = torch.empty_like(order, dtype=torch.float32)
    base = torch.arange(values.shape[1], device=values.device, dtype=torch.float32)
    ranks.scatter_(1, order, base.unsqueeze(0).expand_as(ranks))
    return ranks


def batch_corr(a, b, eps=1e-12):
    a = a.flatten(1).float()
    b = b.flatten(1).float()
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return (a * b).sum(dim=1) / (
        a.square().sum(dim=1).sqrt() * b.square().sum(dim=1).sqrt()
    ).clamp_min(eps)


def pair_metrics(before, after):
    before = before.float()
    after = after.float()
    residual = after - before
    before_flat = before.flatten(1)
    after_flat = after.flatten(1)
    residual_flat = residual.flatten(1)
    before_norm = before_flat.norm(dim=1).clamp_min(1e-12)
    after_norm = after_flat.norm(dim=1)
    cosine = F.cosine_similarity(before_flat, after_flat, dim=1)

    before_std = before_flat.std(dim=1, unbiased=False).clamp_min(1e-12)
    mean_shift = (after_flat.mean(dim=1) - before_flat.mean(dim=1)).abs()

    before_channel = before.square().mean(dim=(-2, -1)).sqrt()
    after_channel = after.square().mean(dim=(-2, -1)).sqrt()
    before_rank = batch_ranks(before_channel)
    after_rank = batch_ranks(after_channel)
    rank_corr = batch_corr(before_rank, after_rank)
    keep = max(1, before_channel.shape[1] // 4)
    before_top = before_channel.topk(keep, dim=1).indices
    after_top = after_channel.topk(keep, dim=1).indices
    overlap = []
    for old, new in zip(before_top, after_top):
        overlap.append(torch.isin(old, new).float().mean())
    overlap = torch.stack(overlap)

    before_energy = before.square().mean(dim=1).sqrt()
    after_energy = after.square().mean(dim=1).sqrt()
    return {
        "relative_l2": residual_flat.norm(dim=1) / before_norm,
        "cosine": cosine,
        "rms_ratio": after_norm / before_norm,
        "mean_shift_in_before_std": mean_shift / before_std,
        "std_ratio": after_flat.std(dim=1, unbiased=False) / before_std,
        "channel_rms_rank_spearman": rank_corr,
        "channel_top25_overlap": overlap,
        "spatial_energy_pearson": batch_corr(before_energy, after_energy),
    }


def make_target_mask(targets, height, width, image_height, image_width, device):
    mask = torch.zeros((len(targets), 1, height, width), dtype=torch.bool, device=device)
    for batch_index, target in enumerate(targets):
        boxes = target["boxes"]
        absolute_xyxy = bool(boxes.numel() and boxes.max() > 2)
        for box in boxes:
            if absolute_xyxy:
                x1, y1, x2, y2 = box
                x1, x2 = x1 / image_width * width, x2 / image_width * width
                y1, y2 = y1 / image_height * height, y2 / image_height * height
            else:
                cx, cy, bw, bh = box
                x1, x2 = (cx - bw / 2) * width, (cx + bw / 2) * width
                y1, y2 = (cy - bh / 2) * height, (cy + bh / 2) * height
            ix1 = max(0, min(width - 1, int(torch.floor(x1).item())))
            iy1 = max(0, min(height - 1, int(torch.floor(y1).item())))
            ix2 = max(ix1 + 1, min(width, int(torch.ceil(x2).item())))
            iy2 = max(iy1 + 1, min(height, int(torch.ceil(y2).item())))
            mask[batch_index, 0, iy1:iy2, ix1:ix2] = True
    return mask


def region_metrics(before, after, target_mask):
    residual_map = (after - before).float().square().mean(dim=1, keepdim=True).sqrt()
    base_map = before.float().square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
    ratio_map = residual_map / base_map
    after_ratio_map = (
        after.float().square().mean(dim=1, keepdim=True).sqrt() / base_map
    )
    ring = F.max_pool2d(target_mask.float(), 5, 1, 2).bool() & ~target_mask
    background = ~F.max_pool2d(target_mask.float(), 5, 1, 2).bool()
    result = {}
    for name, area in (("target", target_mask), ("ring", ring), ("background", background)):
        residual_values = []
        after_values = []
        for batch_index in range(before.shape[0]):
            selected = area[batch_index]
            if selected.any():
                residual_values.append(ratio_map[batch_index][selected].mean())
                after_values.append(after_ratio_map[batch_index][selected].mean())
            else:
                residual_values.append(torch.tensor(float("nan"), device=before.device))
                after_values.append(torch.tensor(float("nan"), device=before.device))
        result[f"residual_to_raw_rms::{name}"] = torch.stack(residual_values)
        result[f"enhanced_to_raw_rms::{name}"] = torch.stack(after_values)
    return result


def mask_region_metrics(probability, target_mask):
    probability = probability.float()
    if probability.shape[-2:] != target_mask.shape[-2:]:
        target_mask = F.interpolate(
            target_mask.float(), size=probability.shape[-2:], mode="nearest"
        ).bool()
    ring = F.max_pool2d(target_mask.float(), 5, 1, 2).bool() & ~target_mask
    background = ~F.max_pool2d(target_mask.float(), 5, 1, 2).bool()
    result = {}
    for name, area in (("target", target_mask), ("ring", ring), ("background", background)):
        values = []
        for batch_index in range(probability.shape[0]):
            selected = area[batch_index]
            values.append(
                probability[batch_index][selected].mean()
                if selected.any()
                else torch.tensor(float("nan"), device=probability.device)
            )
        result[f"mask_mean::{name}"] = torch.stack(values)
    result["mask_target_minus_background"] = (
        result["mask_mean::target"] - result["mask_mean::background"]
    )
    return result


def gq_details(module, feature, target_mask):
    local_x, global_x = torch.split(
        feature.float(), [module.dim_conv3, module.dim_untouched], dim=1
    )
    normalized = module.norm(global_x)
    batch, channels, height, width = normalized.shape
    tokens = normalized.flatten(2).transpose(1, 2)
    token_count = tokens.shape[1]
    heads = module.attn.num_heads
    head_dim = module.attn.head_dim
    global_token = tokens.mean(dim=1, keepdim=True)
    query = module.attn.q(global_token).reshape(batch, 1, heads, head_dim).transpose(1, 2)
    query = query * module.attn.scale
    key_value = module.attn.kv(tokens).reshape(
        batch, token_count, 2, heads, head_dim
    ).permute(2, 0, 3, 1, 4)
    key, value = key_value[0], key_value[1]
    logits = query @ key.transpose(-2, -1)
    attention = logits.softmax(dim=-1)
    context = (attention @ value).transpose(1, 2).reshape(batch, 1, channels)
    context = module.attn.proj(context).squeeze(1)

    entropy = -(attention * attention.clamp_min(1e-12).log()).sum(dim=-1).squeeze(-2)
    entropy = entropy / math.log(token_count)
    top_count = max(1, int(math.ceil(token_count * 0.1)))
    top_mass = attention.topk(top_count, dim=-1).values.sum(dim=-1).squeeze(-2)

    resized_target = F.interpolate(
        target_mask.float(), size=(height, width), mode="nearest"
    ).bool().flatten(2)[:, 0]
    target_mass = []
    target_lift = []
    for batch_index in range(batch):
        selected = resized_target[batch_index]
        fraction = selected.float().mean().clamp_min(1.0 / token_count)
        if selected.any():
            mass = attention[batch_index, :, 0, selected].sum(dim=-1)
            target_mass.append(mass.mean())
            target_lift.append((mass / fraction).mean())
        else:
            target_mass.append(torch.tensor(float("nan"), device=feature.device))
            target_lift.append(torch.tensor(float("nan"), device=feature.device))

    return {
        "attention": attention,
        "context": context,
        "metrics": {
            "attention_entropy_normalized": entropy.mean(dim=1),
            "attention_max_weight": attention.max(dim=-1).values.squeeze(-1).mean(dim=1),
            "attention_top10_mass": top_mass.mean(dim=1),
            "attention_logit_std": logits.flatten(2).std(dim=-1, unbiased=False).mean(dim=1),
            "query_norm": query.flatten(2).norm(dim=-1).mean(dim=1),
            "context_norm": context.norm(dim=1),
            "input_global_to_local_rms": (
                global_x.square().mean(dim=(1, 2, 3)).sqrt()
                / local_x.square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-12)
            ),
            "attention_target_mass": torch.stack(target_mass),
            "attention_target_lift": torch.stack(target_lift),
        },
    }


def gq_pair_metrics(normal, bypass):
    p = normal["attention"].clamp_min(1e-12)
    q = bypass["attention"].clamp_min(1e-12)
    midpoint = 0.5 * (p + q)
    js = 0.5 * (
        (p * (p.log() - midpoint.log())).sum(dim=-1)
        + (q * (q.log() - midpoint.log())).sum(dim=-1)
    )
    attention_cosine = F.cosine_similarity(p.flatten(1), q.flatten(1), dim=1)
    context_cosine = F.cosine_similarity(normal["context"], bypass["context"], dim=1)
    context_relative_l2 = (
        (normal["context"] - bypass["context"]).norm(dim=1)
        / normal["context"].norm(dim=1).clamp_min(1e-12)
    )
    return {
        "attention_js_divergence": js.mean(dim=1).squeeze(-1),
        "attention_cosine": attention_cosine,
        "context_cosine": context_cosine,
        "context_relative_l2": context_relative_l2,
    }


class Collector:
    def __init__(self):
        self.values = defaultdict(lambda: defaultdict(list))

    def add(self, key, tensor, image_groups):
        values = tensor.detach().float().cpu().numpy().reshape(-1)
        if len(values) != len(image_groups):
            raise ValueError((key, len(values), len(image_groups)))
        for value, groups in zip(values, image_groups):
            if not np.isfinite(value):
                continue
            for group in groups:
                self.values[key][group].append(float(value))

    def summary(self):
        result = {}
        for key, groups in sorted(self.values.items()):
            result[key] = {}
            for group, values in sorted(groups.items()):
                array = np.asarray(values, dtype=np.float64)
                result[key][group] = {
                    "n": int(array.size),
                    "mean": float(array.mean()),
                    "std": float(array.std()),
                    "median": float(np.median(array)),
                    "q25": float(np.quantile(array, 0.25)),
                    "q75": float(np.quantile(array, 0.75)),
                }
        return result

    def raw(self):
        return {
            key: {group: values for group, values in groups.items()}
            for key, groups in self.values.items()
        }


def image_groups_from_coco(coco, targets):
    groups = []
    for target in targets:
        image_id = int(target["image_id"].item())
        annotations = coco.imgToAnns.get(image_id, [])
        labels = {"all"}
        for annotation in annotations:
            edge = math.sqrt(float(annotation.get("area", 0.0)))
            if edge < 8:
                labels.add("lt8")
            elif edge < 16:
                labels.add("8to16")
            elif edge < 32:
                labels.add("16to32")
            elif edge < 48:
                labels.add("32to48")
            else:
                labels.add("ge48")
        groups.append(labels)
    return groups


def add_metrics(collector, prefix, metrics, groups):
    for name, values in metrics.items():
        collector.add(f"{prefix}/{name}", values, groups)


def forward_backbone(model, capture, samples, use_amp):
    capture.clear()
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        model.backbone(samples)
    return capture.snapshot()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.repo))
    from src.nn.backbone.partialnet_pat_sf import PartialGlobalQueryConv

    e1_cfg, e1, e1_meta = load_model(
        args.repo,
        args.e1_config,
        args.e1_checkpoint,
        {
            "srtod_stage": 2,
            "srtod_student_only": args.e1_mode == "student",
            "srtod_student_hidden_channels": 32,
            "pat_stage": -1,
            "pat_sf_stage": 3,
            "pat_sf_n_div": 4,
            "pat_sf_variant": "global_query",
        },
    )
    _, def1, def1_meta = load_model(
        args.repo,
        args.def1_config,
        args.def1_checkpoint,
        {
            "srtod_stage": 2,
            "srtod_student_only": True,
            "srtod_student_hidden_channels": 32,
            "pat_stage": -1,
            "pat_sf_stage": -1,
            "pat_sf_n_div": 4,
            "pat_sf_variant": "self",
        },
    )
    _, gq1, gq1_meta = load_model(
        args.repo,
        args.gq1_config,
        args.gq1_checkpoint,
        {
            "srtod_stage": -1,
            "srtod_student_only": False,
            "srtod_student_hidden_channels": 32,
            "pat_stage": -1,
            "pat_sf_stage": 3,
            "pat_sf_n_div": 4,
            "pat_sf_variant": "global_query",
        },
    )
    loader = e1_cfg.val_dataloader
    coco = loader.dataset.coco
    use_amp = args.precision == "fp16"

    e1_capture = FeatureCapture(e1.backbone, PartialGlobalQueryConv)
    def_capture = FeatureCapture(def1.backbone, PartialGlobalQueryConv)
    gq_capture = FeatureCapture(gq1.backbone, PartialGlobalQueryConv)
    if len(e1_capture.gq_names) != 3 or len(gq_capture.gq_names) != 3:
        raise RuntimeError(
            f"expected three GQ modules: e1={e1_capture.gq_names}, "
            f"gq1={gq_capture.gq_names}"
        )
    if def_capture.gq_names:
        raise RuntimeError(f"standalone DEF1 unexpectedly contains GQ: {def_capture.gq_names}")
    dgfe = e1.backbone.srtod_dgfe
    original_forward_with_mask = dgfe.forward_with_mask

    def bypass_forward_with_mask(module, feature, probability):
        module.last_applied_mask = probability.detach()
        return feature

    collector = Collector()
    e1_mask_prefix = (
        "e1_student_mask" if args.e1_mode == "student" else "e1_teacher_mask"
    )
    started = time.time()
    images_seen = 0

    try:
        with torch.inference_mode():
            for batch_index, (samples, targets) in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                samples = samples.cuda(non_blocking=True)
                targets = [
                    {
                        key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
                        for key, value in target.items()
                    }
                    for target in targets
                ]
                groups = image_groups_from_coco(coco, targets)
                images_seen += len(targets)
                image_height, image_width = samples.shape[-2:]

                # Normal E1.
                dgfe.forward_with_mask = original_forward_with_mask
                e1_normal = forward_backbone(e1, e1_capture, samples, use_amp)
                e1_probability = e1.backbone.srtod_difference_mask.detach().float().clone()

                # Same E1 weights and images, with the complete DGFE transform bypassed.
                dgfe.forward_with_mask = types.MethodType(bypass_forward_with_mask, dgfe)
                e1_bypass = forward_backbone(e1, e1_capture, samples, use_amp)
                dgfe.forward_with_mask = original_forward_with_mask

                # Independent references.
                def_normal = forward_backbone(def1, def_capture, samples, use_amp)
                def_probability = def1.backbone.srtod_difference_mask.detach().float().clone()
                gq_normal = forward_backbone(gq1, gq_capture, samples, use_amp)

                target_s16 = make_target_mask(
                    targets,
                    e1_normal["s16_raw"].shape[-2],
                    e1_normal["s16_raw"].shape[-1],
                    image_height,
                    image_width,
                    samples.device,
                )

                add_metrics(
                    collector,
                    "e1_s16_raw_to_enhanced",
                    pair_metrics(e1_normal["s16_raw"], e1_normal["s16_to_s32"]),
                    groups,
                )
                add_metrics(
                    collector,
                    "e1_s16_regions",
                    region_metrics(
                        e1_normal["s16_raw"], e1_normal["s16_to_s32"], target_s16
                    ),
                    groups,
                )
                add_metrics(
                    collector,
                    e1_mask_prefix,
                    mask_region_metrics(e1_probability, target_s16),
                    groups,
                )
                add_metrics(
                    collector,
                    "def1_s16_raw_to_enhanced",
                    pair_metrics(def_normal["s16_raw"], def_normal["s16_to_s32"]),
                    groups,
                )
                add_metrics(
                    collector,
                    "def1_s16_regions",
                    region_metrics(
                        def_normal["s16_raw"], def_normal["s16_to_s32"], target_s16
                    ),
                    groups,
                )
                add_metrics(
                    collector,
                    "def1_student_mask",
                    mask_region_metrics(def_probability, target_s16),
                    groups,
                )

                # Sanity: bypass should feed exactly stage-2 output into stage 3.
                add_metrics(
                    collector,
                    "e1_bypass_s16_sanity",
                    pair_metrics(e1_bypass["s16_raw"], e1_bypass["s16_to_s32"]),
                    groups,
                )
                add_metrics(
                    collector,
                    "e1_normal_vs_bypass_s32_out",
                    pair_metrics(e1_normal["s32_out"], e1_bypass["s32_out"]),
                    groups,
                )

                for module_index, name in enumerate(e1_capture.gq_names):
                    normal_input = e1_normal[f"gq_input::{name}"]
                    bypass_input = e1_bypass[f"gq_input::{name}"]
                    target_gq = make_target_mask(
                        targets,
                        normal_input.shape[-2],
                        normal_input.shape[-1],
                        image_height,
                        image_width,
                        samples.device,
                    )
                    e1_module = dict(e1.backbone.named_modules())[name]
                    normal_details = gq_details(e1_module, normal_input, target_gq)
                    bypass_details = gq_details(e1_module, bypass_input, target_gq)
                    prefix = f"e1_gq{module_index}"
                    add_metrics(
                        collector,
                        f"{prefix}_input_normal_vs_bypass",
                        pair_metrics(normal_input, bypass_input),
                        groups,
                    )
                    add_metrics(
                        collector,
                        f"{prefix}_attention_normal",
                        normal_details["metrics"],
                        groups,
                    )
                    add_metrics(
                        collector,
                        f"{prefix}_attention_bypass",
                        bypass_details["metrics"],
                        groups,
                    )
                    add_metrics(
                        collector,
                        f"{prefix}_normal_vs_bypass",
                        gq_pair_metrics(normal_details, bypass_details),
                        groups,
                    )

                for module_index, name in enumerate(gq_capture.gq_names):
                    feature = gq_normal[f"gq_input::{name}"]
                    target_gq = make_target_mask(
                        targets,
                        feature.shape[-2],
                        feature.shape[-1],
                        image_height,
                        image_width,
                        samples.device,
                    )
                    module = dict(gq1.backbone.named_modules())[name]
                    details = gq_details(module, feature, target_gq)
                    add_metrics(
                        collector,
                        f"gq1_gq{module_index}_attention",
                        details["metrics"],
                        groups,
                    )

                if batch_index == 0 or (batch_index + 1) % 10 == 0:
                    print(
                        f"batch={batch_index + 1}/{len(loader)} images={images_seen}",
                        flush=True,
                    )
    finally:
        dgfe.forward_with_mask = original_forward_with_mask
        e1_capture.close()
        def_capture.close()
        gq_capture.close()

    summary = {
        "protocol": {
            "split": "Val only",
            "precision": args.precision,
            "e1_mode": args.e1_mode,
            "max_batches": args.max_batches,
            "images_seen": images_seen,
            "elapsed_seconds": time.time() - started,
            "paired_causal_comparison": "same E1 weights/images: learned DGFE vs true bypass",
            "standalone_reference_warning": (
                "DEF1 and GQ1 were trained independently; compare aggregate mechanisms, "
                "not their channel identities."
            ),
        },
        "models": {"e1": e1_meta, "def1": def1_meta, "gq1": gq1_meta},
        "gq_module_names": {
            "e1": e1_capture.gq_names,
            "gq1": gq_capture.gq_names,
        },
        "metrics": collector.summary(),
    }
    (args.output_dir / "interaction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with gzip.open(args.output_dir / "interaction_raw_metrics.json.gz", "wt", encoding="utf-8") as f:
        json.dump(collector.raw(), f, ensure_ascii=False)
    print(json.dumps(summary["protocol"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
