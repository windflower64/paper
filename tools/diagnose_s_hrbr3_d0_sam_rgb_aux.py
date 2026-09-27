#!/usr/bin/env python3
"""S-HRBR3-D0：训练期正确/错位/无SAM对RGB框校准探针的配对诊断。

检测器和HRBR1始终冻结。三个探针推理时都只读取RGB裁剪；SAM仅作为训练期
分割辅助目标，因此本实验不会把SAM直接泄漏到推理框。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.ops import roi_align

from train_s_hrbr1_refinebox import (
    BackboneFeatureTap,
    RefineBoxHead,
    aligned_iou,
    box_cxcywh_to_xyxy,
    inverse_sigmoid,
    load_frozen_detector,
    regression_loss,
    save_json,
)


MODES = ("aligned_sam", "shifted_sam", "no_sam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=Path(
            "E:/two_paper/outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth"
        ),
    )
    parser.add_argument(
        "--hrbr1-checkpoint",
        type=Path,
        default=Path(
            "E:/two_paper/outputs/S_HRBR1_REFINEBOX_OFFICIAL_FPN_SEED0/best.pth"
        ),
    )
    parser.add_argument(
        "--train-mask-root",
        type=Path,
        default=Path(
            "E:/two_paper/reports/20_spatial_importance/"
            "S_TNDP2_RETENTION_DISTILLATION/masks_train"
        ),
    )
    parser.add_argument(
        "--val-mask-root",
        type=Path,
        default=Path(
            "E:/two_paper/reports/20_spatial_importance/"
            "S_DIAG3_TRUE_CONTOUR/masks_val"
        ),
    )
    parser.add_argument(
        "--train-image-root",
        type=Path,
        default=Path("E:/two_paper/data/antiuav6k_common/images/train"),
    )
    parser.add_argument(
        "--val-image-root",
        type=Path,
        default=Path("E:/two_paper/data/antiuav6k_common/images/val"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "E:/two_paper/reports/24_high_resolution_box_refinement/"
            "S_HRBR3_D0_SAM_RGB_AUX"
        ),
    )
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--crop-size", type=int, default=48)
    parser.add_argument("--context-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-manual-records", type=int)
    parser.add_argument("--max-heldout-records", type=int)
    return parser.parse_args()


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1
    )


def translate_no_wrap(mask: torch.Tensor, dx: int) -> torch.Tensor:
    result = torch.zeros_like(mask)
    width = mask.shape[-1]
    source_start, source_end = max(0, -dx), min(width, width - dx)
    target_start, target_end = max(0, dx), min(width, width + dx)
    if source_end > source_start:
        result[..., target_start:target_end] = mask[..., source_start:source_end]
    return result


def select_records(mask_root: Path, split: str, maximum: int | None):
    records = json.loads((mask_root / "records.json").read_text(encoding="utf-8"))
    manual_ids = {
        int(path.stem) for path in (mask_root / "audit_overlays").glob("*.webp")
    }
    if split == "train":
        selected = [record for record in records if bool(record.get("accepted"))]
    elif split == "manual_audit":
        selected = [record for record in records if int(record["image_id"]) in manual_ids]
    elif split == "heldout_accepted":
        selected = [
            record
            for record in records
            if bool(record.get("accepted"))
            and int(record["image_id"]) not in manual_ids
        ]
    else:
        raise ValueError(split)
    selected.sort(key=lambda item: int(item["image_id"]))
    return selected if maximum is None else selected[:maximum]


class RecordDataset(Dataset):
    def __init__(self, records, image_root: Path, mask_root: Path) -> None:
        self.records = records
        self.image_root = image_root
        self.mask_root = mask_root

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = np.asarray(
            Image.open(self.image_root / record["file_name"]).convert("RGB"),
            dtype=np.uint8,
        ).copy()
        mask_path = self.mask_root / "masks" / f"{int(record['image_id']):06d}.png"
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8).copy()
        if image.shape[:2] != (512, 640) or mask.shape != (512, 640):
            raise RuntimeError(
                f"非预期图像尺寸 image={image.shape} mask={mask.shape} id={record['image_id']}"
            )
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1),
            "mask": torch.from_numpy(mask > 127).to(torch.uint8)[None],
            "box": torch.tensor(record["bbox_xyxy"], dtype=torch.float32),
            "image_id": int(record["image_id"]),
        }


def load_hrbr1(detector, checkpoint_path: Path):
    channels = tuple(detector.backbone._out_channels[index] for index in (0, 1, 2, 3))
    refiner = RefineBoxHead(channels, d_model=64, roi_size=7, refine_steps=3).cuda()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    refiner.load_state_dict(saved["refiner"], strict=True)
    refiner.force_identity = False
    refiner.requires_grad_(False).eval()
    return refiner


def crop_rois(tensor, boxes, batch_indices, crop_size, context_scale):
    expanded = boxes.clone()
    expanded[:, 2:] *= context_scale
    xyxy = box_cxcywh_to_xyxy(expanded).clamp(0.0, 1.0)
    height, width = tensor.shape[-2:]
    xyxy = xyxy * xyxy.new_tensor((width, height, width, height))
    rois = torch.cat((batch_indices[:, None].to(xyxy), xyxy), dim=1)
    return roi_align(
        tensor,
        rois,
        output_size=crop_size,
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=True,
    )


@torch.inference_mode()
def build_split_cache(
    records,
    image_root,
    mask_root,
    detector,
    hrbr1,
    tap,
    matcher,
    batch_size,
    crop_size,
    context_scale,
):
    dataset = RecordDataset(records, image_root, mask_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    storage = {key: [] for key in ("rgb", "aligned_mask", "shifted_mask", "base", "truth", "image_id")}
    top100_hits = 0
    total = 0
    for batch_index, batch in enumerate(loader):
        images = batch["image"].cuda(non_blocking=True).float() / 255.0
        masks = batch["mask"].cuda(non_blocking=True).float()
        boxes_xyxy = batch["box"].cuda(non_blocking=True)
        width, height = 640.0, 512.0
        truth = xyxy_to_cxcywh(
            boxes_xyxy / boxes_xyxy.new_tensor((width, height, width, height))
        )
        targets = [
            {
                "labels": torch.zeros(1, dtype=torch.long, device="cuda"),
                "boxes": truth[index : index + 1],
            }
            for index in range(len(images))
        ]
        tap.clear()
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = detector(images)
        features = tuple(feature.float() for feature in tap.features())
        core = {
            "pred_logits": outputs["pred_logits"].float(),
            "pred_boxes": outputs["pred_boxes"].float(),
        }
        indices = matcher(core, targets)["indices"]
        query_ids = torch.stack([pair[0][0].to("cuda") for pair in indices])
        batch_ids = torch.arange(len(images), device="cuda")
        predicted = core["pred_boxes"][batch_ids, query_ids]
        refined = hrbr1(features, predicted, batch_ids)[-1]

        ranked = core["pred_logits"].max(-1).values.topk(100, dim=1).indices
        top100_hits += int((ranked == query_ids[:, None]).any(1).sum())
        total += len(images)

        rgb_crop = crop_rois(images, refined, batch_ids, crop_size, context_scale)
        aligned_crop = crop_rois(masks, refined, batch_ids, crop_size, context_scale)
        # 等面积同形位置对照：只改变位置，不改变掩码面积或复杂度。
        shifted_crop = torch.roll(
            aligned_crop, shifts=crop_size // 3, dims=-1
        )
        storage["rgb"].append((rgb_crop.clamp(0, 1) * 255).round().byte().cpu())
        storage["aligned_mask"].append((aligned_crop > 0.5).byte().cpu())
        storage["shifted_mask"].append((shifted_crop > 0.5).byte().cpu())
        storage["base"].append(refined.cpu())
        storage["truth"].append(truth.cpu())
        storage["image_id"].append(batch["image_id"].cpu())
        if batch_index == 0 or (batch_index + 1) % 20 == 0:
            print(f"cache batch={batch_index + 1}/{len(loader)} samples={total}", flush=True)
    result = {key: torch.cat(value, 0) for key, value in storage.items()}
    result["top100_fraction"] = top100_hits / max(1, total)
    return result


class SAMAuxRGBProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, groups=8, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
        )
        self.box_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.mask_head = nn.Conv2d(64, 1, 1)
        nn.init.zeros_(self.box_head[-1].weight)
        nn.init.zeros_(self.box_head[-1].bias)

    def forward(self, rgb, base):
        feature = self.encoder(rgb)
        delta = self.box_head(feature)
        refined = (inverse_sigmoid(base) + delta).sigmoid()
        return refined, self.mask_head(feature)


def mask_loss(logits, target):
    resized = F.interpolate(target, size=logits.shape[-2:], mode="area")
    positives = resized.sum().clamp_min(1.0)
    negatives = (1.0 - resized).sum().clamp_min(1.0)
    pos_weight = (negatives / positives).clamp(1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(logits, resized, pos_weight=pos_weight)
    probability = logits.sigmoid()
    dice = 1.0 - (2 * (probability * resized).sum() + 1.0) / (
        probability.sum() + resized.sum() + 1.0
    )
    return 0.5 * bce + 0.5 * dice


def tensor_loader(split, batch_size, shuffle, seed):
    dataset = TensorDataset(
        split["rgb"], split["aligned_mask"], split["shifted_mask"],
        split["base"], split["truth"], split["image_id"],
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )


def gradient_norm(grads):
    values = [gradient.detach().float().square().sum() for gradient in grads if gradient is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def preflight_gradient_ratio(model, batch, aux_weight):
    rgb, aligned, _shifted, base, truth, _image_id = batch
    rgb = rgb.cuda().float() / 255.0
    aligned = aligned.cuda().float()
    base, truth = base.cuda(), truth.cuda()
    refined, logits = model(rgb, base)
    box, _, _ = regression_loss(refined, truth)
    auxiliary = aux_weight * mask_loss(logits, aligned)
    parameters = [parameter for parameter in model.encoder.parameters() if parameter.requires_grad]
    box_grads = torch.autograd.grad(box, parameters, retain_graph=True, allow_unused=True)
    aux_grads = torch.autograd.grad(auxiliary, parameters, allow_unused=True)
    box_norm, aux_norm = gradient_norm(box_grads), gradient_norm(aux_grads)
    return {
        "box_shared_gradient_norm": box_norm,
        "weighted_aux_shared_gradient_norm": aux_norm,
        "weighted_aux_over_box": aux_norm / max(box_norm, 1e-12),
    }


def train_probe(mode, initial_state, train_split, args, epochs, aux_weight, phase):
    torch.manual_seed(args.seed)
    model = SAMAuxRGBProbe().cuda()
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, int(epochs * 2 / 3)), max(2, int(epochs * 5 / 6))],
        gamma=0.1,
    )
    loader = tensor_loader(train_split, args.batch_size, True, args.seed)
    curve = []
    for epoch in range(epochs):
        model.train()
        totals = {"count": 0, "box": 0.0, "mask": 0.0, "total": 0.0}
        for rgb, aligned, shifted, base, truth, _image_id in loader:
            rgb = rgb.cuda(non_blocking=True).float() / 255.0
            aligned = aligned.cuda(non_blocking=True).float()
            shifted = shifted.cuda(non_blocking=True).float()
            base = base.cuda(non_blocking=True)
            truth = truth.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            refined, logits = model(rgb, base)
            box, _, _ = regression_loss(refined, truth)
            if mode == "aligned_sam":
                auxiliary = mask_loss(logits, aligned)
            elif mode == "shifted_sam":
                auxiliary = mask_loss(logits, shifted)
            else:
                auxiliary = box.new_zeros(())
            total = box + aux_weight * auxiliary
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1, error_if_nonfinite=True)
            optimizer.step()
            count = len(rgb)
            totals["count"] += count
            totals["box"] += float(box.detach()) * count
            totals["mask"] += float(auxiliary.detach()) * count
            totals["total"] += float(total.detach()) * count
        scheduler.step()
        curve.append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "box_loss": totals["box"] / totals["count"],
                "mask_loss": totals["mask"] / totals["count"],
                "total_loss": totals["total"] / totals["count"],
            }
        )
        print(
            f"phase={phase} mode={mode} epoch={epoch:02d} "
            f"{json.dumps(curve[-1])}", flush=True
        )
    return model.eval(), curve


@torch.inference_mode()
def evaluate_probe(model, split, batch_size):
    loader = tensor_loader(split, batch_size, False, 0)
    losses, base_ious, refined_ious = [], [], []
    side_errors, direction_correct, direction_total = [], 0, 0
    for rgb, _aligned, _shifted, base, truth, _image_id in loader:
        rgb = rgb.cuda(non_blocking=True).float() / 255.0
        base = base.cuda(non_blocking=True)
        truth = truth.cuda(non_blocking=True)
        refined, _ = model(rgb, base)
        loss, _, _ = regression_loss(refined, truth)
        losses.append((float(loss), len(rgb)))
        base_xyxy = box_cxcywh_to_xyxy(base)
        refined_xyxy = box_cxcywh_to_xyxy(refined)
        truth_xyxy = box_cxcywh_to_xyxy(truth)
        base_ious.extend(aligned_iou(base_xyxy, truth_xyxy).cpu().tolist())
        refined_ious.extend(aligned_iou(refined_xyxy, truth_xyxy).cpu().tolist())
        error_pixels = (refined_xyxy - truth_xyxy).abs() * refined_xyxy.new_tensor(
            (640.0, 512.0, 640.0, 512.0)
        )
        side_errors.extend(error_pixels.mean(-1).cpu().tolist())
        target_change = (truth_xyxy - base_xyxy) * target_change_scale(base_xyxy)
        predicted_change = (refined_xyxy - base_xyxy) * target_change_scale(base_xyxy)
        valid = target_change.abs() >= 0.5
        direction_correct += int(((target_change * predicted_change) > 0)[valid].sum())
        direction_total += int(valid.sum())
    base_array = np.asarray(base_ious, dtype=np.float64)
    refined_array = np.asarray(refined_ious, dtype=np.float64)
    return {
        "count": int(len(base_array)),
        "box_loss": sum(value * count for value, count in losses)
        / max(1, sum(count for _, count in losses)),
        "base_iou_mean": float(base_array.mean()),
        "refined_iou_mean": float(refined_array.mean()),
        "iou_gain_mean": float((refined_array - base_array).mean()),
        "iou_improved_fraction": float((refined_array > base_array).mean()),
        "mean_side_error_pixels": float(np.mean(side_errors)),
        "direction_accuracy": direction_correct / max(1, direction_total),
        "direction_count": direction_total,
    }


def target_change_scale(boxes_xyxy):
    return boxes_xyxy.new_tensor((640.0, 512.0, 640.0, 512.0))


def gate(results):
    decision = {}
    for split in ("manual_audit", "heldout_accepted"):
        aligned = results["aligned_sam"][split]
        shifted = results["shifted_sam"][split]
        no_sam = results["no_sam"][split]
        decision[split] = {
            "aligned_lower_box_loss_than_both": aligned["box_loss"]
            < min(shifted["box_loss"], no_sam["box_loss"]),
            "aligned_higher_iou_gain_than_both": aligned["iou_gain_mean"]
            > max(shifted["iou_gain_mean"], no_sam["iou_gain_mean"]),
            "aligned_higher_direction_accuracy_than_both": aligned[
                "direction_accuracy"
            ] > max(shifted["direction_accuracy"], no_sam["direction_accuracy"]),
        }
    aligned_h = results["aligned_sam"]["heldout_accepted"]
    shifted_h = results["shifted_sam"]["heldout_accepted"]
    no_h = results["no_sam"]["heldout_accepted"]
    margins = {
        "heldout_iou_gain_advantage_over_best_control": aligned_h["iou_gain_mean"]
        - max(shifted_h["iou_gain_mean"], no_h["iou_gain_mean"]),
        "heldout_direction_advantage_over_best_control": aligned_h[
            "direction_accuracy"
        ] - max(shifted_h["direction_accuracy"], no_h["direction_accuracy"]),
    }
    passed = all(all(values.values()) for values in decision.values()) and (
        margins["heldout_iou_gain_advantage_over_best_control"] >= 0.001
        and margins["heldout_direction_advantage_over_best_control"] >= 0.01
    )
    return {"split_checks": decision, "margins": margins, "pass": bool(passed)}


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo))
    from src.core import YAMLConfig

    if not torch.cuda.is_available():
        raise RuntimeError("S-HRBR3-D0需要CUDA")
    if args.warmup_epochs < 1 or args.warmup_epochs >= args.epochs:
        raise ValueError("warmup-epochs必须至少为1且小于总epochs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache or (args.output_dir / "probe_cache.pt")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_records = select_records(
        args.train_mask_root, "train", args.max_train_records
    )
    manual_records = select_records(
        args.val_mask_root, "manual_audit", args.max_manual_records
    )
    heldout_records = select_records(
        args.val_mask_root, "heldout_accepted", args.max_heldout_records
    )
    selection = {
        "train_accepted": len(train_records),
        "manual_audit": len(manual_records),
        "heldout_accepted": len(heldout_records),
    }
    print(json.dumps(selection, ensure_ascii=False), flush=True)

    if cache_path.exists() and not args.rebuild_cache:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        cfg = YAMLConfig(str(args.config))
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        detector, _ = load_frozen_detector(cfg, args.detector_checkpoint)
        detector.requires_grad_(False).eval()
        hrbr1 = load_hrbr1(detector, args.hrbr1_checkpoint)
        tap = BackboneFeatureTap(detector.backbone)
        matcher = cfg.criterion.cuda().eval().matcher
        cache = {
            "train": build_split_cache(
                train_records, args.train_image_root, args.train_mask_root,
                detector, hrbr1, tap, matcher, args.cache_batch_size,
                args.crop_size, args.context_scale,
            ),
            "manual_audit": build_split_cache(
                manual_records, args.val_image_root, args.val_mask_root,
                detector, hrbr1, tap, matcher, args.cache_batch_size,
                args.crop_size, args.context_scale,
            ),
            "heldout_accepted": build_split_cache(
                heldout_records, args.val_image_root, args.val_mask_root,
                detector, hrbr1, tap, matcher, args.cache_batch_size,
                args.crop_size, args.context_scale,
            ),
            "selection": selection,
        }
        torch.save(cache, cache_path)

    torch.manual_seed(args.seed)
    template = SAMAuxRGBProbe()
    initial_state = copy.deepcopy(template.state_dict())
    warmup_model, warmup_curve = train_probe(
        "no_sam",
        initial_state,
        cache["train"],
        args,
        epochs=args.warmup_epochs,
        aux_weight=0.0,
        phase="shared_warmup",
    )
    warmup_state = {
        key: value.detach().cpu().clone()
        for key, value in warmup_model.state_dict().items()
    }
    del warmup_model
    first_batch = next(iter(tensor_loader(cache["train"], args.batch_size, False, 0)))
    preflight_model = SAMAuxRGBProbe().cuda()
    preflight_model.load_state_dict(warmup_state, strict=True)
    gradient_report = preflight_gradient_ratio(
        preflight_model, first_batch, args.aux_weight
    )
    del preflight_model
    effective_aux_weight = args.aux_weight * min(
        1.0, 0.30 / max(gradient_report["weighted_aux_over_box"], 1e-12)
    )
    gradient_report["configured_aux_weight"] = args.aux_weight
    gradient_report["effective_aux_weight"] = effective_aux_weight
    gradient_report["normalized_ratio_upper_bound"] = 0.30

    results, curves = {}, {}
    started = time.time()
    for mode in MODES:
        model, curve = train_probe(
            mode,
            warmup_state,
            cache["train"],
            args,
            epochs=args.epochs - args.warmup_epochs,
            aux_weight=effective_aux_weight,
            phase="sam_comparison",
        )
        curves[mode] = {"shared_warmup": warmup_curve, "comparison": curve}
        results[mode] = {
            "manual_audit": evaluate_probe(model, cache["manual_audit"], args.batch_size),
            "heldout_accepted": evaluate_probe(
                model, cache["heldout_accepted"], args.batch_size
            ),
        }
        torch.save(
            {"model": model.state_dict(), "mode": mode, "curve": curve},
            args.output_dir / f"{mode}.pth",
        )
        del model
        torch.cuda.empty_cache()

    report = {
        "experiment": "S-HRBR3-D0-SAM-RGB-AUX",
        "protocol": {
            "detector_and_hrbr1_frozen": True,
            "sam_used_at_inference": False,
            "modes": list(MODES),
            "epochs": args.epochs,
            "shared_warmup_epochs": args.warmup_epochs,
            "comparison_epochs": args.epochs - args.warmup_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "configured_aux_weight": args.aux_weight,
            "effective_aux_weight": effective_aux_weight,
            "crop_size": args.crop_size,
            "context_scale": args.context_scale,
            "seed": args.seed,
        },
        "selection": selection,
        "cache_top100_fraction": {
            split: cache[split]["top100_fraction"]
            for split in ("train", "manual_audit", "heldout_accepted")
        },
        "gradient_preflight": gradient_report,
        "results": results,
        "curves": curves,
        "gate": gate(results),
        "elapsed_seconds": time.time() - started,
    }
    save_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
