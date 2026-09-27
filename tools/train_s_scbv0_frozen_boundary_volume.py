#!/usr/bin/env python3
"""S14-SCBV0：冻结A00，只训练语义条件的S8框边界代价体。"""

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.core import YAMLConfig
from s_scbv import (
    SCBV_MODES,
    SemanticConditionedBoundaryVolume,
    box_cxcywh_to_xyxy,
    gradient_l2,
)
from train_s_hrbr1_refinebox import (
    BackboneFeatureTap,
    coco_metrics,
    collect_detections,
    load_frozen_detector,
    matched_training_boxes,
    move_targets,
    regression_loss,
)
from train_s_qbdm0_frozen_detail_reader import custom_size_metrics, save_json


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
        default=ROOT.parent / "outputs/S_SCBV0_ALIGNED_FROZEN_A00_SEED0",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bin-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--objective", choices=("hard_bin", "continuous_box"), default="hard_bin"
    )
    parser.add_argument("--detail-channels", type=int, default=16)
    parser.add_argument("--tangent-points", type=int, default=5)
    parser.add_argument("--offset-bins", type=int, default=9)
    parser.add_argument("--max-relative-offset", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def build_reader(args, detector) -> SemanticConditionedBoundaryVolume:
    return SemanticConditionedBoundaryVolume(
        s8_channels=int(detector.backbone._out_channels[1]),
        s16_channels=int(detector.backbone._out_channels[2]),
        detail_channels=args.detail_channels,
        tangent_points=args.tangent_points,
        offset_bins=args.offset_bins,
        max_relative_offset=args.max_relative_offset,
    )


def core_outputs(outputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return outputs["pred_logits"].float(), outputs["pred_boxes"].float()


def training_objective(args, box_loss, bin_loss):
    if args.objective == "continuous_box":
        return box_loss
    return box_loss + args.bin_loss_weight * bin_loss


def aligned_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    intersection_lt = torch.maximum(left[:, :2], right[:, :2])
    intersection_rb = torch.minimum(left[:, 2:], right[:, 2:])
    intersection_wh = (intersection_rb - intersection_lt).clamp_min(0)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    left_wh = (left[:, 2:] - left[:, :2]).clamp_min(0)
    right_wh = (right[:, 2:] - right[:, :2]).clamp_min(0)
    union = (
        left_wh[:, 0] * left_wh[:, 1]
        + right_wh[:, 0] * right_wh[:, 1]
        - intersection
    )
    return intersection / union.clamp_min(1e-12)


def validation_targets_for_matcher(targets, samples) -> list[dict]:
    """把验证集Resize后的绝对XYXY转换为matcher要求的归一化CXCYWH。

    训练流水线含ConvertBoxes；验证流水线为了COCO评估保留绝对XYXY，不能直接复用
    ``matched_training_boxes``。这里只复制boxes，原targets仍供postprocessor/COCO使用。
    """
    height, width = samples.shape[-2:]
    scale = samples.new_tensor((width, height, width, height))
    converted = []
    for target in targets:
        boxes = target["boxes"].as_subclass(torch.Tensor).float() / scale
        x1, y1, x2, y2 = boxes.unbind(-1)
        cxcywh = torch.stack(
            ((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), dim=-1
        )
        item = dict(target)
        item["boxes"] = cxcywh
        converted.append(item)
    return converted


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
    processed = 0
    for batch_index, (samples, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        processed += 1
        samples = samples.cuda(non_blocking=True)
        targets_cuda = move_targets(targets, "cuda")
        tap.clear()
        with torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16"):
            outputs = detector(samples)
        s8 = tap.outputs[1].float()
        s16 = tap.outputs[2].float()
        logits, detector_boxes = core_outputs(outputs)
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
            refined = reader(s8, s16, flat_boxes, flat_batch, mode=mode)
            adjusted = detector_boxes.clone()
            adjusted[image_ids, selected] = refined.view_as(selected_boxes)
            results = postprocessor(
                {"pred_logits": logits, "pred_boxes": adjusted}, sizes
            )
            collect_detections(detections[mode], targets_cuda, results, category_ids)

    if max_batches is not None:
        return {"partial_batches": processed}
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


@torch.inference_mode()
def evaluate_matched_quality(
    detector,
    reader,
    tap,
    loader,
    matcher,
    precision,
    modes=("aligned",),
    max_batches=None,
) -> dict[str, dict[str, float]]:
    totals = {
        mode: {
            "correct": 0.0,
            "side_count": 0,
            "box_count": 0,
            "baseline_iou_sum": 0.0,
            "refined_iou_sum": 0.0,
            "improved": 0,
            "worsened": 0,
        }
        for mode in modes
    }
    for batch_index, (samples, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        samples = samples.cuda(non_blocking=True)
        targets_cuda = move_targets(targets, "cuda")
        tap.clear()
        with torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16"):
            outputs = detector(samples)
        matcher_targets = validation_targets_for_matcher(targets_cuda, samples)
        predicted, truth, batch_ids = matched_training_boxes(
            outputs, matcher_targets, matcher
        )
        if predicted is None:
            continue
        s8 = tap.outputs[1].float()
        s16 = tap.outputs[2].float()
        for mode in modes:
            refined = reader(s8, s16, predicted.float(), batch_ids, mode=mode)
            logits = reader.last_logits
            if logits is None:
                continue
            _, bins = reader.target_offsets(predicted.float(), truth.float())
            base_iou = aligned_iou(
                box_cxcywh_to_xyxy(predicted.float()),
                box_cxcywh_to_xyxy(truth.float()),
            )
            refined_iou = aligned_iou(
                box_cxcywh_to_xyxy(refined), box_cxcywh_to_xyxy(truth.float())
            )
            delta = refined_iou - base_iou
            totals[mode]["correct"] += float((logits.argmax(-1) == bins).sum())
            totals[mode]["side_count"] += int(bins.numel())
            totals[mode]["box_count"] += int(len(predicted))
            totals[mode]["baseline_iou_sum"] += float(base_iou.sum())
            totals[mode]["refined_iou_sum"] += float(refined_iou.sum())
            totals[mode]["improved"] += int((delta > 0).sum())
            totals[mode]["worsened"] += int((delta < 0).sum())
    return {
        mode: {
            "bin_accuracy": values["correct"] / max(1, values["side_count"]),
            "side_count": values["side_count"],
            "box_count": values["box_count"],
            "baseline_mean_iou": values["baseline_iou_sum"]
            / max(1, values["box_count"]),
            "refined_mean_iou": values["refined_iou_sum"]
            / max(1, values["box_count"]),
            "mean_iou_delta": (
                values["refined_iou_sum"] - values["baseline_iou_sum"]
            )
            / max(1, values["box_count"]),
            "improved_fraction": values["improved"] / max(1, values["box_count"]),
            "worsened_fraction": values["worsened"] / max(1, values["box_count"]),
        }
        for mode, values in totals.items()
    }


def run_preflight(args, detector, matcher, tap, train_loader) -> dict:
    samples, targets = next(iter(train_loader))
    if samples.shape[0] != args.batch_size:
        raise RuntimeError(f"预检要求batch={args.batch_size}，实际{samples.shape[0]}")
    samples = samples.cuda(non_blocking=True)
    targets_cuda = move_targets(targets, "cuda")
    tap.clear()
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.float16, enabled=args.precision == "fp16"
    ):
        outputs = detector(samples)
    s8 = tap.outputs[1].detach().float()
    s16 = tap.outputs[2].detach().float()
    predicted, truth, batch_ids = matched_training_boxes(outputs, targets_cuda, matcher)
    if predicted is None:
        raise RuntimeError("预检batch没有匹配正样本")

    reader = build_reader(args, detector).cuda().train()
    with torch.no_grad():
        zero = reader(s8, s16, predicted.float(), batch_ids, "zero_update")
        zero_error = float((zero - predicted.float()).abs().max())
        initial = reader(s8, s16, predicted.float(), batch_ids, "aligned")
        initial_error = float((initial - predicted.float()).abs().max())
        detail_energy_error = reader.detail_energy_control_error(s8)
    if zero_error != 0.0 or initial_error > 1e-6:
        raise RuntimeError(
            f"恒等预检失败：zero={zero_error}, zero_init_aligned={initial_error}"
        )
    if detail_energy_error > 1e-6:
        raise RuntimeError(f"SHIFT_DETAIL等能量预检失败：{detail_energy_error}")

    reader.zero_grad(set_to_none=True)
    refined = reader(s8, s16, predicted.float(), batch_ids, "aligned")
    logits = reader.last_logits
    if logits is None:
        raise RuntimeError("ALIGNED没有生成候选代价体")
    box_loss, l1, giou = regression_loss(refined, truth.float())
    bin_loss, bin_accuracy = reader.bin_loss_and_accuracy(
        logits, predicted.float(), truth.float()
    )
    loss = training_objective(args, box_loss, bin_loss)
    loss.backward()
    reader_gradient = gradient_l2(reader.parameters())
    detector_gradient = gradient_l2(detector.parameters())
    if (
        not torch.isfinite(loss)
        or not math.isfinite(reader_gradient)
        or reader_gradient <= 0
        or detector_gradient != 0.0
    ):
        raise RuntimeError(
            f"反向预检失败：loss={float(loss)}, reader_grad={reader_gradient}, "
            f"detector_grad={detector_gradient}"
        )

    return {
        "experiment": "S-SCBV0-PREFLIGHT",
        "batch_size": int(samples.shape[0]),
        "matched_boxes": int(len(predicted)),
        "s8_shape": list(s8.shape),
        "s16_shape": list(s16.shape),
        "parameter_count": sum(p.numel() for p in reader.parameters()),
        "zero_update_box_max_error": zero_error,
        "zero_initialized_aligned_box_max_error": initial_error,
        "shift_detail_global_energy_relative_error": detail_energy_error,
        "loss": float(loss.detach()),
        "box_loss": float(box_loss.detach()),
        "l1": float(l1.detach()),
        "giou": float(giou.detach()),
        "bin_loss": float(bin_loss.detach()),
        "initial_bin_accuracy": float(bin_accuracy.detach()),
        "reader_gradient_l2": reader_gradient,
        "frozen_detector_gradient_l2": detector_gradient,
        "mode_statistics": reader.last_mode_statistics,
        "pass": True,
    }


def final_gate(control_report: dict, matched_report: dict, objective: str) -> dict:
    modes = control_report["modes"]
    aligned = modes["aligned"]["absolute"]
    zero = modes["zero_update"]["absolute"]
    low = modes["low_only"]["absolute"]
    shifted = modes["shift_detail"]["absolute"]
    swapped = modes["swap_direction"]["absolute"]
    s16 = modes["s16_only"]["absolute"]
    aligned_16 = modes["aligned"]["custom_size"]["16to32"]["AP75"]
    controls_16 = {
        name: modes[name]["custom_size"]["16to32"]["AP75"]
        for name in ("low_only", "shift_detail", "swap_direction", "s16_only")
    }
    gate = {
        "aligned_ap_over_zero_at_least_0_001": aligned["AP"] - zero["AP"] >= 0.001,
        "aligned_ap75_over_zero_at_least_0_002": aligned["AP75"] - zero["AP75"] >= 0.002,
        "aligned_ap75_over_low_at_least_0_0015": aligned["AP75"] - low["AP75"] >= 0.0015,
        "aligned_ap75_over_shift_at_least_0_002": aligned["AP75"] - shifted["AP75"] >= 0.002,
        "aligned_ap75_over_swap_at_least_0_002": aligned["AP75"] - swapped["AP75"] >= 0.002,
        "aligned_ap75_over_s16_at_least_0_002": aligned["AP75"] - s16["AP75"] >= 0.002,
        "aligned_16to32_ap75_over_all_controls": all(
            aligned_16 > value for value in controls_16.values()
        ),
    }
    if objective == "hard_bin":
        gate["aligned_bin_accuracy_at_least_0_35"] = (
            matched_report["aligned"]["bin_accuracy"] >= 0.35
        )
    else:
        gate["aligned_matched_mean_iou_positive"] = (
            matched_report["aligned"]["mean_iou_delta"] > 0.0
        )
        gate["aligned_matched_improved_majority"] = (
            matched_report["aligned"]["improved_fraction"] > 0.5
        )
    gate["pass"] = all(gate.values())
    gate["deltas"] = {
        "aligned_minus_zero_ap": aligned["AP"] - zero["AP"],
        "aligned_minus_zero_ap75": aligned["AP75"] - zero["AP75"],
        "aligned_minus_low_ap75": aligned["AP75"] - low["AP75"],
        "aligned_minus_shift_ap75": aligned["AP75"] - shifted["AP75"],
        "aligned_minus_swap_ap75": aligned["AP75"] - swapped["AP75"],
        "aligned_minus_s16_ap75": aligned["AP75"] - s16["AP75"],
        "aligned_16to32_ap75": aligned_16,
        "control_16to32_ap75": controls_16,
        "aligned_bin_accuracy": matched_report["aligned"]["bin_accuracy"],
        "aligned_matched_mean_iou_delta": matched_report["aligned"]["mean_iou_delta"],
        "aligned_matched_improved_fraction": matched_report["aligned"]["improved_fraction"],
    }
    return gate


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("S-SCBV0需要CUDA")
    if not args.config.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"配置或权重不存在：config={args.config}, checkpoint={args.checkpoint}"
        )
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
    tap = BackboneFeatureTap(detector.backbone, stage_indices=(1, 2))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    preflight = run_preflight(args, detector, matcher, tap, train_loader)
    save_json(args.output_dir / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return

    torch.manual_seed(args.seed)
    reader = build_reader(args, detector).cuda()
    optimizer = torch.optim.AdamW(
        reader.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, int(args.epochs * 2 / 3)), max(2, int(args.epochs * 5 / 6))],
        gamma=0.1,
    )
    metadata = {
        "experiment": "S-SCBV0-ALIGNED-FROZEN-A00",
        "purpose": "目标语义限定对象，沿四条框边法线直接读取S8候选偏移",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_weight_source": weight_source,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "bin_loss_weight": args.bin_loss_weight,
        "objective": args.objective,
        "detail_channels": args.detail_channels,
        "tangent_points": args.tangent_points,
        "offset_bins": args.offset_bins,
        "max_relative_offset": args.max_relative_offset,
        "topk": args.topk,
        "trainable_parameters": sum(p.numel() for p in reader.parameters()),
        "seed": args.seed,
        "modes": SCBV_MODES,
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
            targets_cuda = move_targets(targets, "cuda")
            tap.clear()
            with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=args.precision == "fp16"
            ):
                outputs = detector(samples)
            predicted, truth, batch_ids = matched_training_boxes(
                outputs, targets_cuda, matcher
            )
            if predicted is None:
                continue
            s8 = tap.outputs[1].detach().float()
            s16 = tap.outputs[2].detach().float()
            optimizer.zero_grad(set_to_none=True)
            refined = reader(s8, s16, predicted.float(), batch_ids, "aligned")
            logits = reader.last_logits
            if logits is None:
                raise RuntimeError("训练时ALIGNED没有产生候选代价体")
            box_loss, l1, giou = regression_loss(refined, truth.float())
            bin_loss, bin_accuracy = reader.bin_loss_and_accuracy(
                logits, predicted.float(), truth.float()
            )
            loss = training_objective(args, box_loss, bin_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"epoch={epoch}出现非有限损失")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                reader.parameters(), 0.1, error_if_nonfinite=True
            )
            optimizer.step()
            count = len(predicted)
            positives += count
            for name, value in (
                ("loss", loss),
                ("box_loss", box_loss),
                ("l1", l1),
                ("giou", giou),
                ("bin_loss", bin_loss),
                ("bin_accuracy", bin_accuracy),
                ("gradient_norm", gradient_norm),
            ):
                running[name] += float(value.detach()) * count
            if batch_index == 0 or (batch_index + 1) % 20 == 0:
                print(
                    f"epoch={epoch:02d} batch={batch_index + 1:03d}/{len(train_loader)} "
                    f"loss={float(loss):.6f} bin_acc={float(bin_accuracy):.4f} "
                    f"positives={positives}",
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
            **{name: value / max(1, positives) for name, value in running.items()},
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
        current_ap = (
            validation.get("modes", {})
            .get("aligned", {})
            .get("absolute", {})
            .get("AP", -1.0)
        )
        if current_ap > best_ap:
            best_ap = current_ap
            torch.save(payload, args.output_dir / "best.pth")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    if not (args.output_dir / "best.pth").is_file():
        raise RuntimeError("没有生成best权重；请勿使用max-val-batches执行正式训练")
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
        modes=SCBV_MODES,
        include_custom=True,
    )
    matched_report = evaluate_matched_quality(
        detector,
        reader,
        tap,
        val_loader,
        matcher,
        args.precision,
        modes=tuple(mode for mode in SCBV_MODES if mode != "zero_update"),
    )
    controls["best_epoch"] = int(best_payload["epoch"])
    controls["matched_quality"] = matched_report
    controls["preregistered_gate"] = final_gate(
        controls, matched_report, args.objective
    )
    controls["interpretation"] = (
        "通过：正确位置和方向的S8边界证据不可由低频、错位或S16替代，允许进入SCBV1联合训练。"
        if controls["preregistered_gate"]["pass"]
        else "未通过：SCBV没有建立正确S8细节的不可替代性，不启动SCBV1，不搜索局部超参数。"
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
