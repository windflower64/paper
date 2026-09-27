#!/usr/bin/env python3
"""S-QBDM0：冻结A00，只训练预测框四边的下采样前细节读取器。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.core import YAMLConfig
from train_s_hrbr1_refinebox import (
    BackboneFeatureTap,
    box_cxcywh_to_xyxy,
    coco_metrics,
    collect_detections,
    load_frozen_detector,
    matched_training_boxes,
    move_targets,
    regression_loss,
)


MODES = ("aligned", "shifted", "phase_permuted", "zero", "low_only")
METRICS = ("AP", "AP50", "AP75", "APS", "APM", "AR100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/phase_s/visible_60e_base_local.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT.parent
        / "outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT.parent / "outputs/S_QBDM0_ALIGNED_FROZEN_A00_SEED0",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--detail-channels", type=int, default=16)
    parser.add_argument("--points-per-side", type=int, default=3)
    parser.add_argument("--max-relative-offset", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class QBDMDetailReader(nn.Module):
    """把S8相位细节保存为独立记忆，并只在预测框四边读取。"""

    def __init__(
        self,
        in_channels: int,
        detail_channels: int = 16,
        points_per_side: int = 3,
        max_relative_offset: float = 0.25,
    ) -> None:
        super().__init__()
        if points_per_side < 1:
            raise ValueError("points_per_side必须为正整数")
        self.detail_channels = detail_channels
        self.points_per_side = points_per_side
        self.max_relative_offset = max_relative_offset
        self.reduce = nn.Conv2d(in_channels, detail_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.reduce.weight, a=1)
        side_input = detail_channels * 3 * points_per_side
        self.side_heads = nn.ModuleList()
        for _ in range(4):
            head = nn.Sequential(
                nn.Linear(side_input, 32, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1, bias=False),
            )
            nn.init.zeros_(head[-1].weight)
            self.side_heads.append(head)

    @staticmethod
    def decompose(z8: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if z8.shape[-2] % 2 or z8.shape[-1] % 2:
            raise ValueError(f"S8空间尺寸必须为偶数，当前为{tuple(z8.shape[-2:])}")
        a = z8[..., 0::2, 0::2]
        b = z8[..., 0::2, 1::2]
        c = z8[..., 1::2, 0::2]
        d = z8[..., 1::2, 1::2]
        # 1/2使四个分量构成正交、等能量的2×2 Haar变换。
        ll = (a + b + c + d) * 0.5
        lh = (a - b + c - d) * 0.5
        hl = (a + b - c - d) * 0.5
        hh = (a - b - c + d) * 0.5
        return ll, lh, hl, hh

    @staticmethod
    def reconstruct(
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        a = (ll + lh + hl + hh) * 0.5
        b = (ll - lh + hl - hh) * 0.5
        c = (ll + lh - hl - hh) * 0.5
        d = (ll - lh - hl + hh) * 0.5
        output = ll.new_empty(
            ll.shape[0], ll.shape[1], ll.shape[-2] * 2, ll.shape[-1] * 2
        )
        output[..., 0::2, 0::2] = a
        output[..., 0::2, 1::2] = b
        output[..., 1::2, 0::2] = c
        output[..., 1::2, 1::2] = d
        return output

    def build_memory(
        self, s8: torch.Tensor, mode: str = "aligned"
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if mode not in MODES:
            raise ValueError(f"未知细节模式：{mode}")
        z8 = self.reduce(s8)
        ll, lh, hl, hh = self.decompose(z8)
        detail = torch.cat((lh, hl, hh), dim=1)
        if mode == "aligned":
            memory = detail
        elif mode == "shifted":
            memory = torch.roll(
                detail,
                shifts=(max(1, detail.shape[-2] // 3), max(1, detail.shape[-1] // 5)),
                dims=(-2, -1),
            )
        elif mode == "phase_permuted":
            memory = torch.cat((hl, hh, lh), dim=1)
        elif mode == "zero":
            memory = torch.zeros_like(detail)
        else:
            low = torch.cat((ll, ll, ll), dim=1) / math.sqrt(3.0)
            detail_norm = detail.flatten(1).norm(dim=1)
            low_norm = low.flatten(1).norm(dim=1).clamp_min(1e-12)
            memory = low * (detail_norm / low_norm)[:, None, None, None]
        return memory, {
            "z8": z8,
            "ll": ll,
            "lh": lh,
            "hl": hl,
            "hh": hh,
            "detail": detail,
        }

    def side_points(self, boxes: torch.Tensor) -> torch.Tensor:
        xyxy = box_cxcywh_to_xyxy(boxes).clamp(0.0, 1.0)
        x1, y1, x2, y2 = xyxy.unbind(-1)
        offsets = torch.linspace(
            -0.25,
            0.25,
            self.points_per_side,
            device=boxes.device,
            dtype=boxes.dtype,
        )
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        width = x2 - x1
        height = y2 - y1
        vertical_y = cy[:, None] + height[:, None] * offsets[None]
        horizontal_x = cx[:, None] + width[:, None] * offsets[None]
        left = torch.stack((x1[:, None].expand_as(vertical_y), vertical_y), dim=-1)
        top = torch.stack((horizontal_x, y1[:, None].expand_as(horizontal_x)), dim=-1)
        right = torch.stack((x2[:, None].expand_as(vertical_y), vertical_y), dim=-1)
        bottom = torch.stack((horizontal_x, y2[:, None].expand_as(horizontal_x)), dim=-1)
        return torch.stack((left, top, right, bottom), dim=1).clamp(0.0, 1.0)

    @staticmethod
    def sample_memory(
        memory: torch.Tensor,
        points: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> torch.Tensor:
        sampled = memory.new_zeros(
            points.shape[0], points.shape[1], points.shape[2], memory.shape[1]
        )
        grid = points * 2.0 - 1.0
        for image_index in batch_indices.unique(sorted=True).tolist():
            mask = batch_indices == image_index
            image_grid = grid[mask].reshape(1, -1, 1, 2)
            values = F.grid_sample(
                memory[image_index : image_index + 1],
                image_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            values = values[0, :, :, 0].transpose(0, 1).reshape(
                int(mask.sum()), points.shape[1], points.shape[2], memory.shape[1]
            )
            sampled[mask] = values
        return sampled

    def forward(
        self,
        s8: torch.Tensor,
        boxes: torch.Tensor,
        batch_indices: torch.Tensor,
        mode: str = "aligned",
    ) -> torch.Tensor:
        # ZERO 是严格的原检测器对照：不能让后续坐标裁剪悄悄改变原始预测框。
        if mode == "zero":
            return boxes
        memory, _ = self.build_memory(s8, mode)
        sampled = self.sample_memory(memory, self.side_points(boxes), batch_indices)
        side_vectors = sampled.flatten(2)
        raw_offsets = torch.cat(
            [self.side_heads[index](side_vectors[:, index]) for index in range(4)],
            dim=1,
        )
        relative = self.max_relative_offset * torch.tanh(raw_offsets)
        xyxy = box_cxcywh_to_xyxy(boxes)
        x1, y1, x2, y2 = xyxy.unbind(-1)
        width = (x2 - x1).clamp_min(1e-5)
        height = (y2 - y1).clamp_min(1e-5)
        new_x1 = (x1 + relative[:, 0] * width).clamp(0.0, 1.0)
        new_y1 = (y1 + relative[:, 1] * height).clamp(0.0, 1.0)
        new_x2 = (x2 + relative[:, 2] * width).clamp(0.0, 1.0)
        new_y2 = (y2 + relative[:, 3] * height).clamp(0.0, 1.0)
        new_width = (new_x2 - new_x1).clamp_min(1e-5)
        new_height = (new_y2 - new_y1).clamp_min(1e-5)
        return torch.stack(
            (
                (new_x1 + new_x2) * 0.5,
                (new_y1 + new_y2) * 0.5,
                new_width,
                new_height,
            ),
            dim=-1,
        )


def custom_size_metrics(coco_gt, detections) -> dict:
    from faster_coco_eval import COCOeval_faster

    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")
    labels = ("all", "lt8", "8to16", "16to32", "32to48", "ge48")
    edges = (
        (0, 1e10),
        (0, 8**2),
        (8**2, 16**2),
        (16**2, 32**2),
        (32**2, 48**2),
        (48**2, 1e10),
    )
    evaluator.params.areaRng = [list(edge) for edge in edges]
    evaluator.params.areaRngLbl = list(labels)
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    i75 = int(np.argmin(np.abs(evaluator.params.iouThrs - 0.75)))

    def mean_valid(array):
        valid = array[array > -1]
        return float(valid.mean()) if valid.size else None

    result = {}
    for area_index, label in enumerate(labels):
        result[label] = {
            "AP50_95": mean_valid(precision[:, :, :, area_index, -1]),
            "AP50": mean_valid(precision[0, :, :, area_index, -1]),
            "AP75": mean_valid(precision[i75, :, :, area_index, -1]),
            "AR100": mean_valid(recall[:, :, area_index, -1]),
        }
    return result


@torch.inference_mode()
def evaluate(
    detector,
    reader,
    tap,
    loader,
    postprocessor,
    topk,
    precision,
    modes=("aligned",),
    include_custom=False,
    max_batches=None,
) -> dict:
    detector.eval()
    reader.eval()
    coco = loader.dataset.coco
    category_ids = sorted(coco.getCatIds())
    baseline_detections = []
    detections = {mode: [] for mode in modes}
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
        s8 = tap.outputs[1].float()
        logits = outputs["pred_logits"].float()
        detector_boxes = outputs["pred_boxes"].float()
        scores = logits.sigmoid().max(-1).values
        selected = scores.topk(min(topk, scores.shape[1]), dim=1).indices
        image_ids = torch.arange(samples.shape[0], device=samples.device)[:, None]
        selected_boxes = detector_boxes[image_ids, selected]
        flat_boxes = selected_boxes.flatten(0, 1)
        flat_batch = image_ids.expand_as(selected).flatten()
        sizes = torch.stack([target["orig_size"] for target in targets_cuda])
        baseline_results = postprocessor(
            {"pred_logits": logits, "pred_boxes": detector_boxes}, sizes
        )
        collect_detections(
            baseline_detections, targets_cuda, baseline_results, category_ids
        )
        for mode in modes:
            refined = reader(s8, flat_boxes, flat_batch, mode=mode)
            adjusted = detector_boxes.clone()
            adjusted[image_ids, selected] = refined.view_as(selected_boxes)
            results = postprocessor(
                {"pred_logits": logits, "pred_boxes": adjusted}, sizes
            )
            collect_detections(detections[mode], targets_cuda, results, category_ids)

    if max_batches is not None:
        return {"partial_batches": min(max_batches, len(loader))}
    baseline = coco_metrics(coco, baseline_detections)
    report = {"baseline": {name: baseline[name] for name in METRICS}, "modes": {}}
    for mode in modes:
        metrics = coco_metrics(coco, detections[mode])
        report["modes"][mode] = {
            "absolute": {name: metrics[name] for name in METRICS},
            "paired_delta": {
                name: metrics[name] - baseline[name] for name in METRICS
            },
        }
        if include_custom:
            report["modes"][mode]["custom_size"] = custom_size_metrics(
                coco, detections[mode]
            )
    if include_custom:
        report["baseline_custom_size"] = custom_size_metrics(
            coco, baseline_detections
        )
    return report


def run_preflight(args, detector, matcher, tap, train_loader, channels) -> dict:
    samples, targets = next(iter(train_loader))
    if samples.shape[0] != args.batch_size:
        raise RuntimeError(f"预检需要batch={args.batch_size}，实际为{samples.shape[0]}")
    samples = samples.cuda(non_blocking=True)
    targets = move_targets(targets, "cuda")
    tap.clear()
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.float16, enabled=args.precision == "fp16"
    ):
        outputs = detector(samples)
        s8 = tap.outputs[1].detach().float()
    predicted, truth, batch_ids = matched_training_boxes(outputs, targets, matcher)
    if predicted is None:
        raise RuntimeError("预检batch没有匹配正样本")
    reader = QBDMDetailReader(
        channels,
        args.detail_channels,
        args.points_per_side,
        args.max_relative_offset,
    ).cuda()
    parameter_count = sum(parameter.numel() for parameter in reader.parameters())
    with torch.no_grad():
        aligned, parts = reader.build_memory(s8, "aligned")
        reconstructed = reader.reconstruct(
            parts["ll"], parts["lh"], parts["hl"], parts["hh"]
        )
        reconstruction_error = float((reconstructed - parts["z8"]).abs().max())
        mode_energy = {}
        for mode in MODES:
            memory, _ = reader.build_memory(s8, mode)
            mode_energy[mode] = float(memory.square().sum().sqrt())
        shifted_error = abs(mode_energy["aligned"] - mode_energy["shifted"]) / max(
            mode_energy["aligned"], 1e-12
        )
        phase_error = abs(
            mode_energy["aligned"] - mode_energy["phase_permuted"]
        ) / max(mode_energy["aligned"], 1e-12)
        identity = reader(s8, predicted.float(), batch_ids, mode="zero")
        identity_error = float((identity - predicted.float()).abs().max())
    if reconstruction_error > 1e-6 or shifted_error > 1e-6 or phase_error > 1e-6:
        raise RuntimeError("相位分解或等能量控制预检失败")
    if identity_error != 0.0:
        raise RuntimeError(f"ZERO恒等框预检失败：{identity_error}")

    reader.train()
    reader.zero_grad(set_to_none=True)
    refined = reader(s8, predicted.float(), batch_ids, mode="aligned")
    loss = regression_loss(refined, truth.float())[0]
    loss.backward()
    gradients = [
        parameter.grad.detach().float().square().sum()
        for parameter in reader.parameters()
        if parameter.grad is not None
    ]
    gradient_norm = float(torch.stack(gradients).sum().sqrt())
    if not torch.isfinite(loss) or not math.isfinite(gradient_norm) or gradient_norm <= 0:
        raise RuntimeError("batch16反向数值预检失败")
    return {
        "experiment": "S-QBDM0-PREFLIGHT",
        "batch_size": int(samples.shape[0]),
        "matched_boxes": int(len(predicted)),
        "s8_shape": list(s8.shape),
        "parameter_count": parameter_count,
        "reconstruction_max_error": reconstruction_error,
        "memory_l2": mode_energy,
        "shifted_energy_relative_error": shifted_error,
        "phase_permuted_energy_relative_error": phase_error,
        "zero_identity_box_max_error": identity_error,
        "aligned_loss": float(loss.detach()),
        "gradient_norm": gradient_norm,
        "pass": True,
    }


def final_gate(control_report: dict) -> dict:
    modes = control_report["modes"]
    aligned = modes["aligned"]["absolute"]
    zero = modes["zero"]["absolute"]
    shifted = modes["shifted"]["absolute"]
    phase = modes["phase_permuted"]["absolute"]
    aligned_16 = modes["aligned"]["custom_size"]["16to32"]
    controls_16 = [
        modes[name]["custom_size"]["16to32"]
        for name in ("zero", "shifted", "phase_permuted")
    ]
    gate = {
        "aligned_ap_over_zero_at_least_0_001": aligned["AP"] - zero["AP"] >= 0.001,
        "aligned_ap_over_shifted_at_least_0_001": aligned["AP"] - shifted["AP"] >= 0.001,
        "aligned_ap_over_phase_at_least_0_001": aligned["AP"] - phase["AP"] >= 0.001,
        "aligned_ap75_over_controls_at_least_0_002": min(
            aligned["AP75"] - zero["AP75"],
            aligned["AP75"] - shifted["AP75"],
            aligned["AP75"] - phase["AP75"],
        )
        >= 0.002,
        "aligned_16to32_ap75_over_controls": all(
            aligned_16["AP75"] > item["AP75"] for item in controls_16
        ),
    }
    gate["pass"] = all(gate.values())
    return gate


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("S-QBDM0需要CUDA")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = args.batch_size
    detector, weight_source = load_frozen_detector(cfg, args.checkpoint)
    matcher = cfg.criterion.cuda().eval().matcher
    train_loader = cfg.train_dataloader
    val_loader = cfg.val_dataloader
    postprocessor = cfg.postprocessor
    tap = BackboneFeatureTap(detector.backbone)
    s8_channels = int(detector.backbone._out_channels[1])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    preflight = run_preflight(
        args, detector, matcher, tap, train_loader, s8_channels
    )
    save_json(args.output_dir / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return

    # 预检会消耗随机数；正式头重新固定种子，确保协议可复现。
    torch.manual_seed(args.seed)
    reader = QBDMDetailReader(
        s8_channels,
        args.detail_channels,
        args.points_per_side,
        args.max_relative_offset,
    ).cuda()
    optimizer = torch.optim.AdamW(
        reader.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, int(args.epochs * 2 / 3)), max(2, int(args.epochs * 5 / 6))],
        gamma=0.1,
    )
    metadata = {
        "experiment": "S-QBDM0-ALIGNED-FROZEN-A00",
        "purpose": "保存S8到S16压缩前相位细节，仅在预测框四边供定位读取",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_weight_source": weight_source,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "detail_channels": args.detail_channels,
        "points_per_side": args.points_per_side,
        "max_relative_offset": args.max_relative_offset,
        "topk": args.topk,
        "trainable_parameters": sum(p.numel() for p in reader.parameters()),
        "seed": args.seed,
    }
    save_json(args.output_dir / "metadata.json", metadata)
    history = []
    best_ap = -1.0
    started = time.time()

    for epoch in range(args.epochs):
        detector.eval()
        reader.train()
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
                s8 = tap.outputs[1].detach().float()
            predicted, truth, batch_ids = matched_training_boxes(outputs, targets, matcher)
            if predicted is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            refined = reader(s8, predicted.float(), batch_ids, mode="aligned")
            loss, l1, giou = regression_loss(refined, truth.float())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"epoch={epoch}出现非有限损失")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                reader.parameters(), 0.1, error_if_nonfinite=True
            )
            optimizer.step()
            count = len(predicted)
            positives += count
            running["loss"] += float(loss.detach()) * count
            running["l1"] += float(l1.detach()) * count
            running["giou"] += float(giou.detach()) * count
            running["gradient_norm"] += float(gradient_norm) * count
            if batch_index == 0 or (batch_index + 1) % 20 == 0:
                print(
                    f"epoch={epoch:02d} batch={batch_index + 1:03d}/{len(train_loader)} "
                    f"loss={float(loss):.6f} positives={positives}",
                    flush=True,
                )
        scheduler.step()
        validation = evaluate(
            detector,
            reader,
            tap,
            val_loader,
            postprocessor,
            args.topk,
            args.precision,
            modes=("aligned",),
            max_batches=args.max_val_batches,
        )
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "positives": positives,
            "train_loss": running["loss"] / max(1, positives),
            "train_l1": running["l1"] / max(1, positives),
            "train_giou": running["giou"] / max(1, positives),
            "gradient_norm": running["gradient_norm"] / max(1, positives),
            "validation": validation,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        save_json(args.output_dir / "history.json", history)
        payload = {
            "reader": reader.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metadata": metadata,
            "validation": validation,
        }
        torch.save(payload, args.output_dir / "last.pth")
        current_ap = validation.get("modes", {}).get("aligned", {}).get(
            "absolute", {}
        ).get("AP", -1.0)
        if current_ap > best_ap:
            best_ap = current_ap
            torch.save(payload, args.output_dir / "best.pth")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    best_payload = torch.load(
        args.output_dir / "best.pth", map_location="cpu", weights_only=False
    )
    reader.load_state_dict(best_payload["reader"], strict=True)
    controls = evaluate(
        detector,
        reader,
        tap,
        val_loader,
        postprocessor,
        args.topk,
        args.precision,
        modes=MODES,
        include_custom=True,
    )
    controls["best_epoch"] = int(best_payload["epoch"])
    controls["preregistered_gate"] = final_gate(controls)
    controls["interpretation"] = (
        "通过：正确位置与相位的压缩前细节对定位不可替代，允许进入QBDM1联合训练。"
        if controls["preregistered_gate"]["pass"]
        else "未通过：细节读取器没有形成正确位置/相位依赖，关闭QBDM，不启动完整联合训练。"
    )
    save_json(args.output_dir / "control_report.json", controls)
    summary = {
        "metadata": metadata,
        "best_epoch": int(best_payload["epoch"]),
        "best_validation": best_payload["validation"],
        "control_report": str((args.output_dir / "control_report.json").resolve()),
        "gate": controls["preregistered_gate"],
        "elapsed_seconds": time.time() - started,
    }
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
