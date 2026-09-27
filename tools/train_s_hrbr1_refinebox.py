#!/usr/bin/env python3
"""S-HRBR1：在冻结的 D-FINE 上训练 RefineBox 风格的框校准器。

第一阶段刻意不使用 SAM。它只回答一个问题：已有检测框周围的 S8/S16/S32
局部特征，能否在不改类别分数的情况下进一步改善框的位置。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import MultiScaleRoIAlign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "E:/two_paper/outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "E:/two_paper/outputs/S_HRBR1_REFINEBOX_OFFICIAL_FPN_SEED0"
        ),
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--roi-size", type=int, default=7)
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument(
        "--eval-residual-scale",
        type=float,
        default=1.0,
        help="验证时保留的框残差比例；训练目标不变，默认1.0",
    )
    parser.add_argument(
        "--feature-stages",
        default="0,1,2,3",
        help="逗号分隔的HGNetv2 stage索引；默认0,1,2,3对应S4/S8/S16/S32",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("full", "lowpass", "shifted_detail"),
        default="full",
        help="HRBR5-D0 P2因果控制；默认full保持HRBR1行为不变",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--refiner-checkpoint", type=Path)
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="额外保存每个epoch的refiner checkpoint，便于固定轮次多seed复核",
    )
    parser.add_argument(
        "--experiment-name",
        default="S-HRBR1-RefineBox-OfficialFPN-S4S8S16S32",
        help="写入metadata的实验名称；不改变训练逻辑",
    )
    parser.add_argument(
        "--purpose",
        default=None,
        help="写入metadata的实验目的；不改变训练逻辑",
    )
    parser.add_argument(
        "--detector-name",
        default="A00",
        help="冻结检测器名称；未显式传入purpose时用于在Python内部生成中文说明",
    )
    return parser.parse_args()


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0.0, max=1.0)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def parse_feature_stages(specification: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item.strip()) for item in specification.split(","))
    except ValueError as error:
        raise ValueError(f"无法解析feature_stages: {specification}") from error
    if not indices or indices[0] != 0:
        raise ValueError("feature_stages必须从stage 0（S4）开始")
    if indices != tuple(sorted(set(indices))):
        raise ValueError("feature_stages必须严格递增且不能重复")
    if any(index < 0 or index > 3 for index in indices):
        raise ValueError("feature_stages只允许0～3")
    return indices


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
        dim=-1,
    )


def aligned_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes_a[:, :2], boxes_b[:, :2])
    bottom_right = torch.minimum(boxes_a[:, 2:], boxes_b[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(-1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp_min(0).prod(-1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp_min(0).prod(-1)
    return intersection / (area_a + area_b - intersection).clamp_min(1e-9)


def aligned_giou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    iou = aligned_iou(boxes_a, boxes_b)
    top_left = torch.minimum(boxes_a[:, :2], boxes_b[:, :2])
    bottom_right = torch.maximum(boxes_a[:, 2:], boxes_b[:, 2:])
    enclosing = (bottom_right - top_left).clamp_min(0).prod(-1)
    intersection_top_left = torch.maximum(boxes_a[:, :2], boxes_b[:, :2])
    intersection_bottom_right = torch.minimum(boxes_a[:, 2:], boxes_b[:, 2:])
    intersection = (intersection_bottom_right - intersection_top_left).clamp_min(0).prod(-1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp_min(0).prod(-1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp_min(0).prod(-1)
    union = area_a + area_b - intersection
    return iou - (enclosing - union) / enclosing.clamp_min(1e-9)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        # 对齐官方 Detectron2 BottleneckBlock：1×1、3×3、1×1，GN，groups=1。
        groups = 32 if channels % 32 == 0 else math.gcd(channels, 8)
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv3 = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm3 = nn.GroupNorm(groups, channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.activation(self.norm1(self.conv1(x)))
        residual = self.activation(self.norm2(self.conv2(residual)))
        residual = self.norm3(self.conv3(residual))
        return self.activation(x + residual)


class RefineBoxFPN(nn.Module):
    """官方RBFPN的最小等价适配：自顶向下相加，输出P2/P3/P4/P5。"""

    def __init__(self, in_channels: tuple[int, ...], out_channels: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(channels, out_channels, 1) for channels in in_channels
        )
        self.output = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels
        )
        for module in list(self.lateral) + list(self.output):
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.zeros_(module.bias)

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        results = [None] * len(features)
        inner = self.lateral[-1](features[-1])
        results[-1] = self.output[-1](inner)
        for index in range(len(features) - 2, -1, -1):
            lateral = self.lateral[index](features[index])
            inner = lateral + F.interpolate(inner, size=lateral.shape[-2:], mode="nearest")
            results[index] = self.output[index](inner)
        return tuple(results)


class RefineBoxHead(nn.Module):
    """RefineBox 核心的轻量 PyTorch 适配：多尺度 ROI、残差块、迭代回归。"""

    def __init__(
        self,
        in_channels: tuple[int, ...],
        d_model: int = 64,
        roi_size: int = 7,
        refine_steps: int = 3,
        feature_mode: str = "full",
    ) -> None:
        super().__init__()
        self.roi_size = roi_size
        self.refine_steps = refine_steps
        if feature_mode not in {"full", "lowpass", "shifted_detail"}:
            raise ValueError(f"未知feature_mode: {feature_mode}")
        self.feature_mode = feature_mode
        # 只用于“未训练挂载”的严格等价预检；正式训练前必须关闭。
        self.force_identity = True
        self.fpn = RefineBoxFPN(in_channels, d_model)
        self.roi_pooler = MultiScaleRoIAlign(
            featmap_names=[str(index) for index in range(len(in_channels))],
            output_size=roi_size,
            sampling_ratio=2,
            canonical_scale=224,
            canonical_level=4,
        )
        self.residual = nn.Sequential(*(ResidualBlock(d_model) for _ in range(3)))
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model, bias=False),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )
        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)

    def pool(
        self,
        features: tuple[torch.Tensor, ...],
        boxes: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        normalized_xyxy = box_cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)
        image_height = features[0].shape[-2] * 4
        image_width = features[0].shape[-1] * 4
        scale = normalized_xyxy.new_tensor(
            (image_width, image_height, image_width, image_height)
        )
        absolute_xyxy = normalized_xyxy * scale
        boxes_per_image = [
            absolute_xyxy[batch_indices == image_index]
            for image_index in range(features[0].shape[0])
        ]
        feature_dict = OrderedDict(
            (str(index), feature) for index, feature in enumerate(features)
        )
        image_shapes = [
            (image_height, image_width) for _ in range(features[0].shape[0])
        ]
        return self.roi_pooler(feature_dict, boxes_per_image, image_shapes)

    def refine_once(
        self,
        features: tuple[torch.Tensor, ...],
        boxes: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.force_identity:
            return boxes
        pooled = self.residual(self.pool(features, boxes, batch_indices))
        vector = F.adaptive_avg_pool2d(pooled, 1).flatten(1)
        delta = self.regressor(vector)
        return (inverse_sigmoid(boxes) + delta).sigmoid()

    def forward(
        self,
        features: tuple[torch.Tensor, ...],
        boxes: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> list[torch.Tensor]:
        sequence = []
        refined = boxes
        pyramid = list(self.fpn(features))
        if self.feature_mode != "full":
            p2 = pyramid[0]
            lowpass = F.interpolate(
                F.avg_pool2d(p2, kernel_size=2, stride=2),
                size=p2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if self.feature_mode == "lowpass":
                pyramid[0] = lowpass
            else:
                detail = p2 - lowpass
                shift_y = max(1, p2.shape[-2] // 3)
                shift_x = max(1, p2.shape[-1] // 5)
                shifted = torch.roll(
                    detail, shifts=(shift_y, shift_x), dims=(-2, -1)
                )
                pyramid[0] = lowpass + shifted
        pyramid = tuple(pyramid)
        for _ in range(self.refine_steps):
            refined = self.refine_once(pyramid, refined, batch_indices)
            sequence.append(refined)
        return sequence


class BackboneFeatureTap:
    def __init__(self, backbone: nn.Module, stage_indices=(0, 1, 2, 3)) -> None:
        self.outputs: dict[int, torch.Tensor] = {}
        self.handles = []
        for index in stage_indices:
            self.handles.append(
                backbone.stages[index].register_forward_hook(self._make_hook(index))
            )
        self.stage_indices = tuple(stage_indices)

    def _make_hook(self, index: int):
        def hook(_module, _inputs, output):
            self.outputs[index] = output

        return hook

    def features(self) -> tuple[torch.Tensor, ...]:
        missing = [index for index in self.stage_indices if index not in self.outputs]
        if missing:
            raise RuntimeError(f"未捕获到 backbone stages: {missing}")
        return tuple(self.outputs[index] for index in self.stage_indices)

    def clear(self) -> None:
        self.outputs.clear()


def load_frozen_detector(cfg, checkpoint_path: Path):
    detector = cfg.model.cuda().eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = checkpoint.get("ema", {}).get("module")
    weight_source = "ema.module"
    if weights is None:
        weights = checkpoint.get("model", checkpoint)
        weight_source = "model_or_raw"
    detector.load_state_dict(weights, strict=True)
    detector.requires_grad_(False)
    return detector, weight_source


def move_targets(targets, device):
    return [
        {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def matched_training_boxes(outputs, targets, matcher):
    core = {
        "pred_logits": outputs["pred_logits"].float(),
        "pred_boxes": outputs["pred_boxes"].float(),
    }
    indices = matcher(core, targets)["indices"]
    predicted, truth, batch_ids = [], [], []
    for batch_index, ((query_ids, target_ids), target) in enumerate(zip(indices, targets)):
        query_ids = query_ids.to(core["pred_boxes"].device)
        target_ids = target_ids.to(target["boxes"].device)
        if not len(query_ids):
            continue
        predicted.append(core["pred_boxes"][batch_index, query_ids])
        truth.append(target["boxes"][target_ids])
        batch_ids.append(
            torch.full_like(query_ids, batch_index, dtype=torch.long)
        )
    if not predicted:
        return None, None, None
    return torch.cat(predicted), torch.cat(truth), torch.cat(batch_ids)


def regression_loss(predicted: torch.Tensor, truth: torch.Tensor):
    l1 = F.l1_loss(predicted, truth)
    predicted_xyxy = box_cxcywh_to_xyxy(predicted)
    truth_xyxy = box_cxcywh_to_xyxy(truth)
    giou = (1.0 - aligned_giou(predicted_xyxy, truth_xyxy)).mean()
    return 5.0 * l1 + 2.0 * giou, l1, giou


def collect_detections(storage, targets, results, category_ids):
    for target, result in zip(targets, results):
        boxes = result["boxes"].detach().cpu().clone()
        boxes[:, 2:] -= boxes[:, :2]
        for box, score, label in zip(
            boxes.tolist(), result["scores"].tolist(), result["labels"].tolist()
        ):
            storage.append(
                {
                    "image_id": int(target["image_id"].item()),
                    "category_id": int(category_ids[int(label)]),
                    "bbox": box,
                    "score": float(score),
                }
            )


def coco_metrics(coco_gt, detections):
    from faster_coco_eval import COCOeval_faster

    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = (
        "AP", "AP50", "AP75", "APS", "APM", "APL",
        "AR1", "AR10", "AR100", "ARS", "ARM", "ARL",
    )
    return {name: float(value) for name, value in zip(names, evaluator.stats)}


@torch.inference_mode()
def evaluate(
    detector,
    refiner,
    tap,
    loader,
    postprocessor,
    topk,
    precision,
    residual_scale=1.0,
    max_batches=None,
):
    detector.eval()
    refiner.eval()
    coco = loader.dataset.coco
    category_ids = sorted(coco.getCatIds())
    detections = []
    baseline_detections = []
    use_amp = precision == "fp16"
    for batch_index, (samples, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        samples = samples.cuda(non_blocking=True)
        targets_cuda = move_targets(targets, "cuda")
        tap.clear()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            outputs = detector(samples)
        # ROIAlign 和微小框回归保持 FP32；检测器仍可用 FP16 冻结推理。
        features = tuple(feature.float() for feature in tap.features())
        scores = outputs["pred_logits"].float().sigmoid().max(-1).values
        selected = scores.topk(min(topk, scores.shape[1]), dim=1).indices
        batch_ids = torch.arange(samples.shape[0], device=samples.device)[:, None]
        boxes = outputs["pred_boxes"].float()[batch_ids, selected]
        flat_boxes = boxes.flatten(0, 1)
        flat_batch = batch_ids.expand_as(selected).flatten()
        refined = refiner(features, flat_boxes, flat_batch)[-1]
        refined = (
            flat_boxes + residual_scale * (refined - flat_boxes)
        ).clamp(0.0, 1.0)
        adjusted = {
            "pred_logits": outputs["pred_logits"].float().clone(),
            "pred_boxes": outputs["pred_boxes"].float().clone(),
        }
        adjusted["pred_boxes"][batch_ids, selected] = refined.view_as(boxes)
        sizes = torch.stack([target["orig_size"] for target in targets_cuda])
        baseline_outputs = {
            "pred_logits": outputs["pred_logits"].float(),
            "pred_boxes": outputs["pred_boxes"].float(),
        }
        baseline_results = postprocessor(baseline_outputs, sizes)
        results = postprocessor(adjusted, sizes)
        collect_detections(
            baseline_detections, targets_cuda, baseline_results, category_ids
        )
        collect_detections(detections, targets_cuda, results, category_ids)
    if max_batches is not None:
        return {
            "partial_batches": min(max_batches, len(loader)),
            "detections": len(detections),
            "baseline_detections": len(baseline_detections),
        }
    baseline_metrics = coco_metrics(coco, baseline_detections)
    refined_metrics = coco_metrics(coco, detections)
    refined_metrics["baseline"] = baseline_metrics
    refined_metrics["delta"] = {
        key: refined_metrics[key] - baseline_metrics[key]
        for key in baseline_metrics
    }
    return refined_metrics


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    stage_indices = parse_feature_stages(args.feature_stages)
    if not 0.0 <= args.eval_residual_scale <= 1.0:
        raise ValueError("eval_residual_scale必须位于[0, 1]")
    sys.path.insert(0, str(args.repo))
    from src.core import YAMLConfig

    if not torch.cuda.is_available():
        raise RuntimeError("S-HRBR1 需要 CUDA")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = args.batch_size
    detector, weight_source = load_frozen_detector(cfg, args.checkpoint)
    criterion = cfg.criterion.cuda().eval()
    matcher = criterion.matcher
    train_loader = cfg.train_dataloader
    val_loader = cfg.val_dataloader
    postprocessor = cfg.postprocessor
    tap = BackboneFeatureTap(detector.backbone, stage_indices=stage_indices)
    channels = tuple(detector.backbone._out_channels[index] for index in stage_indices)
    refiner = RefineBoxHead(
        channels, args.d_model, args.roi_size, args.refine_steps, args.feature_mode
    ).cuda()

    if args.refiner_checkpoint is not None:
        saved = torch.load(args.refiner_checkpoint, map_location="cpu", weights_only=False)
        refiner.load_state_dict(saved.get("refiner", saved), strict=True)
        refiner.force_identity = False

    trainable_parameters = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
    frozen_parameters = sum(p.numel() for p in detector.parameters())
    metadata = {
        "experiment": args.experiment_name,
        "purpose": args.purpose or (
            f"冻结{args.detector_name}并从零训练HRBR1，只校准框；"
            "不使用SAM、不修改类别分数"
        ),
        "detector_name": args.detector_name,
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "weight_source": weight_source,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "feature_stage_indices": list(stage_indices),
        "feature_strides": [4 * (2 ** index) for index in stage_indices],
        "feature_channels": list(channels),
        "d_model": args.d_model,
        "roi_size": args.roi_size,
        "refine_steps": args.refine_steps,
        "topk": args.topk,
        "eval_residual_scale": args.eval_residual_scale,
        "feature_mode": args.feature_mode,
        "trainable_parameters": trainable_parameters,
        "frozen_detector_parameters": frozen_parameters,
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)

    if args.eval_only:
        metrics = evaluate(
            detector, refiner, tap, val_loader, postprocessor,
            args.topk, args.precision, args.eval_residual_scale,
            args.max_val_batches,
        )
        save_json(args.output_dir / "eval_only.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        return

    optimizer = torch.optim.AdamW(
        refiner.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, int(args.epochs * 2 / 3)), max(2, int(args.epochs * 5 / 6))],
        gamma=0.1,
    )
    history = []
    best_ap = -1.0
    started = time.time()
    # 从第一批起真正启用逆 sigmoid 增量回归，保留完整训练梯度。
    refiner.force_identity = False

    for epoch in range(args.epochs):
        detector.eval()
        refiner.train()
        running = defaultdict(float)
        positives = 0
        for batch_index, (samples, targets) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            samples = samples.cuda(non_blocking=True)
            targets = move_targets(targets, "cuda")
            tap.clear()
            # 使用 no_grad 而非 inference_mode：冻结特征本身不求梯度，但后续
            # 可训练 ROI 头仍需把这些普通张量保存用于自身参数的反向传播。
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=args.precision == "fp16"
            ):
                outputs = detector(samples)
                features = tuple(feature.detach().float() for feature in tap.features())
            predicted, truth, batch_ids = matched_training_boxes(outputs, targets, matcher)
            if predicted is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            sequence = refiner(features, predicted.float(), batch_ids)
            step_losses = []
            final_l1 = final_giou = None
            for refined in sequence:
                loss, final_l1, final_giou = regression_loss(refined, truth.float())
                step_losses.append(loss)
            total_loss = torch.stack(step_losses).sum()
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"非有限训练损失: {float(total_loss)}")
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                refiner.parameters(), 0.1, error_if_nonfinite=True
            )
            optimizer.step()

            count = len(predicted)
            positives += count
            running["loss"] += float(total_loss.detach()) * count
            running["l1"] += float(final_l1.detach()) * count
            running["giou"] += float(final_giou.detach()) * count
            running["grad_norm"] += float(gradient_norm) * count
            if batch_index == 0 or (batch_index + 1) % 20 == 0:
                print(
                    f"epoch={epoch:02d} batch={batch_index + 1:03d}/{len(train_loader)} "
                    f"loss={float(total_loss):.5f} positives={positives}",
                    flush=True,
                )

        scheduler.step()
        epoch_record = {
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
            detector, refiner, tap, val_loader, postprocessor,
            args.topk, args.precision, args.eval_residual_scale,
            args.max_val_batches,
        )
        epoch_record["validation"] = metrics
        history.append(epoch_record)
        save_json(args.output_dir / "history.json", history)
        checkpoint_payload = {
            "refiner": refiner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metadata": metadata,
            "validation": metrics,
        }
        torch.save(checkpoint_payload, args.output_dir / "last.pth")
        if args.save_every_epoch:
            torch.save(
                checkpoint_payload,
                args.output_dir / f"epoch_{epoch:02d}.pth",
            )
        current_ap = metrics.get("AP", -1.0)
        if current_ap > best_ap:
            best_ap = current_ap
            torch.save(checkpoint_payload, args.output_dir / "best.pth")
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)

    summary = {
        "metadata": metadata,
        "best_epoch": max(
            history, key=lambda item: item["validation"].get("AP", -1.0)
        )["epoch"] if history else None,
        "best_validation": max(
            history, key=lambda item: item["validation"].get("AP", -1.0)
        )["validation"] if history else None,
        "history_path": str((args.output_dir / "history.json").resolve()),
        "elapsed_seconds": time.time() - started,
    }
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
