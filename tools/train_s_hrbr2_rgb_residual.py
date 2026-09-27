#!/usr/bin/env python3
"""S-HRBR2：冻结A00和HRBR1，只训练原始RGB裁剪的二级框残差。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align

from train_s_hrbr1_refinebox import (
    BackboneFeatureTap,
    RefineBoxHead,
    box_cxcywh_to_xyxy,
    coco_metrics,
    collect_detections,
    inverse_sigmoid,
    load_frozen_detector,
    matched_training_boxes,
    move_targets,
    regression_loss,
    save_json,
)


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
        "--output-dir",
        type=Path,
        default=Path("E:/two_paper/outputs/S_HRBR2_RGB_RESIDUAL_SEED0"),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--crop-size", type=int, default=48)
    parser.add_argument("--context-scale", type=float, default=3.0)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--rgb-checkpoint", type=Path)
    return parser.parse_args()


class RGBResidualRefiner(nn.Module):
    def __init__(self, crop_size: int = 48, context_scale: float = 3.0) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.context_scale = context_scale
        self.force_identity = True
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
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def crop(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        expanded = boxes.clone()
        expanded[:, 2:] = expanded[:, 2:] * self.context_scale
        xyxy = box_cxcywh_to_xyxy(expanded).clamp(0.0, 1.0)
        height, width = images.shape[-2:]
        xyxy = xyxy * xyxy.new_tensor((width, height, width, height))
        rois = torch.cat((batch_indices[:, None].to(xyxy), xyxy), dim=1)
        return roi_align(
            images,
            rois,
            output_size=self.crop_size,
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )

    def forward(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.force_identity:
            return boxes
        crops = self.crop(images, boxes, batch_indices)
        delta = self.regressor(self.encoder(crops))
        return (inverse_sigmoid(boxes) + delta).sigmoid()


def load_hrbr1(detector, checkpoint_path: Path):
    channels = tuple(detector.backbone._out_channels[index] for index in (0, 1, 2, 3))
    refiner = RefineBoxHead(channels, d_model=64, roi_size=7, refine_steps=3).cuda()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    refiner.load_state_dict(saved["refiner"], strict=True)
    refiner.force_identity = False
    refiner.requires_grad_(False).eval()
    return refiner


@torch.inference_mode()
def evaluate(
    detector,
    hrbr1,
    rgb_refiner,
    tap,
    loader,
    postprocessor,
    topk,
    precision,
    max_batches=None,
):
    detector.eval()
    hrbr1.eval()
    rgb_refiner.eval()
    coco = loader.dataset.coco
    category_ids = sorted(coco.getCatIds())
    baseline_detections, rgb_detections = [], []
    for batch_index, (samples, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        samples = samples.cuda(non_blocking=True)
        targets_cuda = move_targets(targets, "cuda")
        tap.clear()
        with torch.autocast(
            "cuda", dtype=torch.float16, enabled=precision == "fp16"
        ):
            outputs = detector(samples)
        features = tuple(feature.float() for feature in tap.features())
        scores = outputs["pred_logits"].float().sigmoid().max(-1).values
        selected = scores.topk(min(topk, scores.shape[1]), dim=1).indices
        batch_grid = torch.arange(samples.shape[0], device=samples.device)[:, None]
        base_boxes = outputs["pred_boxes"].float()[batch_grid, selected]
        flat_batch = batch_grid.expand_as(selected).flatten()
        hrbr1_boxes = hrbr1(features, base_boxes.flatten(0, 1), flat_batch)[-1]
        rgb_boxes = rgb_refiner(samples.float(), hrbr1_boxes, flat_batch)

        common = {
            "pred_logits": outputs["pred_logits"].float(),
            "pred_boxes": outputs["pred_boxes"].float(),
        }
        hrbr1_outputs = {key: value.clone() for key, value in common.items()}
        rgb_outputs = {key: value.clone() for key, value in common.items()}
        hrbr1_outputs["pred_boxes"][batch_grid, selected] = hrbr1_boxes.view_as(base_boxes)
        rgb_outputs["pred_boxes"][batch_grid, selected] = rgb_boxes.view_as(base_boxes)
        sizes = torch.stack([target["orig_size"] for target in targets_cuda])
        hrbr1_results = postprocessor(hrbr1_outputs, sizes)
        rgb_results = postprocessor(rgb_outputs, sizes)
        collect_detections(
            baseline_detections, targets_cuda, hrbr1_results, category_ids
        )
        collect_detections(rgb_detections, targets_cuda, rgb_results, category_ids)

    if max_batches is not None:
        return {
            "partial_batches": min(max_batches, len(loader)),
            "hrbr1_detections": len(baseline_detections),
            "rgb_detections": len(rgb_detections),
        }
    baseline = coco_metrics(coco, baseline_detections)
    rgb = coco_metrics(coco, rgb_detections)
    rgb["hrbr1_baseline"] = baseline
    rgb["delta"] = {key: rgb[key] - baseline[key] for key in baseline}
    return rgb


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo))
    from src.core import YAMLConfig

    if not torch.cuda.is_available():
        raise RuntimeError("S-HRBR2需要CUDA")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = args.batch_size
    detector, weight_source = load_frozen_detector(cfg, args.detector_checkpoint)
    detector.requires_grad_(False).eval()
    hrbr1 = load_hrbr1(detector, args.hrbr1_checkpoint)
    tap = BackboneFeatureTap(detector.backbone)
    matcher = cfg.criterion.cuda().eval().matcher
    train_loader = cfg.train_dataloader
    val_loader = cfg.val_dataloader
    postprocessor = cfg.postprocessor
    rgb_refiner = RGBResidualRefiner(args.crop_size, args.context_scale).cuda()
    if args.rgb_checkpoint is not None:
        saved = torch.load(args.rgb_checkpoint, map_location="cpu", weights_only=False)
        rgb_refiner.load_state_dict(saved.get("rgb_refiner", saved), strict=True)
        rgb_refiner.force_identity = False

    metadata = {
        "experiment": "S-HRBR2-RGB-Residual",
        "purpose": "冻结A00和HRBR1，只检验下采样前RGB裁剪的额外框校准价值",
        "detector_checkpoint": str(args.detector_checkpoint.resolve()),
        "hrbr1_checkpoint": str(args.hrbr1_checkpoint.resolve()),
        "weight_source": weight_source,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "crop_size": args.crop_size,
        "context_scale": args.context_scale,
        "topk": args.topk,
        "rgb_trainable_parameters": sum(p.numel() for p in rgb_refiner.parameters()),
        "hrbr1_frozen_parameters": sum(p.numel() for p in hrbr1.parameters()),
        "detector_frozen_parameters": sum(p.numel() for p in detector.parameters()),
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)

    if args.eval_only:
        metrics = evaluate(
            detector, hrbr1, rgb_refiner, tap, val_loader, postprocessor,
            args.topk, args.precision, args.max_val_batches,
        )
        save_json(args.output_dir / "eval_only.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        return

    rgb_refiner.force_identity = False
    optimizer = torch.optim.AdamW(
        rgb_refiner.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, int(args.epochs * 2 / 3)), max(2, int(args.epochs * 5 / 6))],
        gamma=0.1,
    )
    history = []
    best_ap = -1.0
    started = time.time()
    for epoch in range(args.epochs):
        rgb_refiner.train()
        running = defaultdict(float)
        positives = 0
        for batch_index, (samples, targets) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            samples = samples.cuda(non_blocking=True)
            targets = move_targets(targets, "cuda")
            tap.clear()
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=args.precision == "fp16"
            ):
                outputs = detector(samples)
            features = tuple(feature.detach().float() for feature in tap.features())
            predicted, truth, batch_ids = matched_training_boxes(outputs, targets, matcher)
            if predicted is None:
                continue
            with torch.no_grad():
                hrbr1_boxes = hrbr1(features, predicted.float(), batch_ids)[-1]
            optimizer.zero_grad(set_to_none=True)
            rgb_boxes = rgb_refiner(samples.float(), hrbr1_boxes, batch_ids)
            loss, l1, giou = regression_loss(rgb_boxes, truth.float())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"非有限训练损失: {float(loss)}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                rgb_refiner.parameters(), 0.1, error_if_nonfinite=True
            )
            optimizer.step()
            count = len(predicted)
            positives += count
            running["loss"] += float(loss.detach()) * count
            running["l1"] += float(l1.detach()) * count
            running["giou"] += float(giou.detach()) * count
            running["grad_norm"] += float(gradient_norm) * count
            if batch_index == 0 or (batch_index + 1) % 20 == 0:
                print(
                    f"epoch={epoch:02d} batch={batch_index + 1:03d}/{len(train_loader)} "
                    f"loss={float(loss):.5f} positives={positives}", flush=True,
                )
        scheduler.step()
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "positives": positives,
            "train_loss": running["loss"] / max(1, positives),
            "train_l1": running["l1"] / max(1, positives),
            "train_giou": running["giou"] / max(1, positives),
            "gradient_norm": running["grad_norm"] / max(1, positives),
            "elapsed_seconds": time.time() - started,
        }
        metrics = evaluate(
            detector, hrbr1, rgb_refiner, tap, val_loader, postprocessor,
            args.topk, args.precision, args.max_val_batches,
        )
        record["validation"] = metrics
        history.append(record)
        save_json(args.output_dir / "history.json", history)
        payload = {
            "rgb_refiner": rgb_refiner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metadata": metadata,
            "validation": metrics,
        }
        torch.save(payload, args.output_dir / "last.pth")
        current_ap = metrics.get("AP", -1.0)
        if current_ap > best_ap:
            best_ap = current_ap
            torch.save(payload, args.output_dir / "best.pth")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    best = max(history, key=lambda item: item["validation"].get("AP", -1.0))
    summary = {
        "metadata": metadata,
        "best_epoch": best["epoch"],
        "best_validation": best["validation"],
        "history_path": str((args.output_dir / "history.json").resolve()),
        "elapsed_seconds": time.time() - started,
    }
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
