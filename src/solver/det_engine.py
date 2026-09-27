"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import math
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    max_norm: float = 0,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    epochs = kwargs.get("epochs", None)
    header = "Epoch: [{}]".format(epoch) if epochs is None else "Epoch: [{}/{}]".format(epoch, epochs)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    losses = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)
    frequency_distiller = kwargs.get("frequency_distiller", None)
    rgb_preserver = kwargs.get("rgb_preserver", None)
    accumulation_steps = max(
        1, int(kwargs.get("gradient_accumulation_steps", 1))
    )
    optimizer.zero_grad(set_to_none=True)

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            visual_samples = samples[:, :3] if samples.shape[1] == 6 else samples
            save_samples(
                visual_samples,
                targets,
                output_dir,
                "train",
                normalized=True,
                box_fmt="cxcywh",
            )

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        accumulation_window_start = (i // accumulation_steps) * accumulation_steps
        accumulation_window_size = min(
            accumulation_steps,
            len(data_loader) - accumulation_window_start,
        )
        should_update = (
            (i + 1) % accumulation_steps == 0 or i + 1 == len(data_loader)
        )

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                print(outputs["pred_boxes"])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace("module.", "")
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state["model"] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)
                if frequency_distiller is not None:
                    loss_dict["loss_sfreq"] = frequency_distiller(samples, targets)
                if rgb_preserver is not None:
                    loss_dict["loss_rgb_preserve"] = rgb_preserver(samples, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss / accumulation_window_size).backward()

            if should_update:
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)
            if frequency_distiller is not None:
                loss_dict["loss_sfreq"] = frequency_distiller(samples, targets)
            if rgb_preserver is not None:
                loss_dict["loss_rgb_preserve"] = rgb_preserver(samples, targets)

            loss: torch.Tensor = sum(loss_dict.values())
            (loss / accumulation_window_size).backward()

            if should_update:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if should_update:
            # EMA and warmup follow optimizer updates, not physical micro-batches.
            if ema is not None:
                ema.update(model)

            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        losses.append(loss_value.detach().cpu().numpy())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if "stql_response_margin" in outputs:
            metric_logger.update(
                stql_fg=outputs["stql_foreground_response"].detach(),
                stql_bg=outputs["stql_background_response"].detach(),
                stql_margin=outputs["stql_response_margin"].detach(),
                stql_valid=outputs["stql_valid_instances"].detach(),
                stql_schedule=outputs["stql_schedule"].detach(),
            )
        if "sqer_matched_topk_coverage" in outputs:
            sqer_telemetry = {
                key: value.detach()
                for key, value in outputs.items()
                if key.startswith("sqer_")
                and isinstance(value, torch.Tensor)
                and value.numel() == 1
            }
            metric_logger.update(**sqer_telemetry)
        if "mote_update_ratio" in outputs:
            metric_logger.update(
                mote_candidate_confidence=outputs[
                    "mote_candidate_scores"
                ].detach().max(dim=1).values.mean(),
                mote_gate=outputs["mote_gate_mean"].detach(),
                mote_object_attention=outputs[
                    "mote_object_attention_mean"
                ].detach(),
                mote_update_ratio=outputs["mote_update_ratio"].detach(),
            )
        if "qcer_delta_logits" in outputs:
            telemetry = {
                "qcer_delta_abs": outputs["qcer_delta_logits"].detach().abs().mean(),
                "qcer_gate": outputs["qcer_gate"].detach().mean(),
                "qcer_entropy": outputs["qcer_attention_entropy"].detach().mean(),
                "qcer_availability": outputs["qcer_availability"].detach().float().mean(),
            }
            if "qcer_positive_tokens" in outputs:
                telemetry.update(
                    qcer_positive_tokens=outputs["qcer_positive_tokens"].detach(),
                    qcer_fallback_tokens=outputs["qcer_fallback_tokens"].detach(),
                )
            metric_logger.update(**telemetry)
        if "qdmf_gate_mean" in outputs:
            telemetry = {
                key: value.detach()
                for key, value in outputs.items()
                if key.startswith("qdmf_")
                and isinstance(value, torch.Tensor)
                and value.numel() == 1
            }
            metric_logger.update(**telemetry)

        # SDTEC protocol telemetry is deliberately kept outside ``loss_dict``
        # so it is logged without changing the optimization objective.
        if "msd2_rms_ratio_by_level" in outputs:
            telemetry = {
                "msd2_rms_ratio": outputs[
                    "msd2_rms_ratio_by_level"
                ].detach().mean(),
                "msd2_rms_ratio_max": outputs[
                    "msd2_rms_ratio_by_level"
                ].detach().max(),
                "msd2_scale_abs": outputs[
                    "msd2_scale_by_level"
                ].detach().abs().mean(),
                "msd2_content_ratio": outputs[
                    "msd2_content_ratio"
                ].detach(),
                "msd2_context_abs_mean": outputs[
                    "msd2_context_abs_mean"
                ].detach(),
            }
            for level, ratio in enumerate(outputs["msd2_rms_ratio_by_level"].detach().unbind(-1)):
                telemetry[f"msd2_rms_level{level}"] = ratio.mean()
                telemetry[f"msd2_rms_level{level}_max"] = ratio.max()
            if "msd2_token_diversity" in outputs:
                telemetry["msd2_token_diversity"] = outputs[
                    "msd2_token_diversity"
                ].detach()
            if "msd2_null_raw_rms_by_level" in outputs:
                telemetry["msd2_null_raw_rms"] = outputs[
                    "msd2_null_raw_rms_by_level"
                ].detach().mean()
                telemetry["msd2_contrast_raw_rms"] = outputs[
                    "msd2_contrast_raw_rms_by_level"
                ].detach().mean()
            metric_logger.update(**telemetry)
        elif "ma1_gate" in outputs:
            effective_offset = outputs["ma1_effective_offset"].detach()
            telemetry = {
                "ma1_gate": outputs["ma1_gate"].detach().mean(),
                "ma1_abs_delta": outputs["ma1_logit_delta"].detach().abs().mean(),
                "ma1_offset_norm": effective_offset.norm(dim=-1).mean(),
                "ma1_max_residual": outputs[
                    "ma1_max_abs_residual_offset"
                ].detach(),
                "ma1_attention_entropy": outputs[
                    "ma1_attention_entropy"
                ].detach().mean(),
                "ma1_content_ratio": outputs[
                    "ma1_thermal_content_mask"
                ].detach().float().mean(),
                "ma1_progress": outputs["ma1_fusion_progress"].detach(),
            }
            metric_logger.update(**telemetry)
        elif "sdtec_gate_by_layer" in outputs:
            telemetry = {
                "sdtec_gate": outputs["sdtec_gate_by_layer"].detach().mean(),
                "sdtec_quality": outputs["sdtec_token_quality"].detach().mean(),
                "sdtec_uncertainty": outputs["sdtec_uncertainty"].detach().mean(),
                "sdtec_availability": outputs["sdtec_presence_logits"]
                .detach()
                .sigmoid()
                .mean(),
                "sdtec_progress": outputs["sdtec_fusion_progress"].detach(),
            }
            for layer_index, scale in enumerate(
                outputs["sdtec_effective_scale_by_layer"].detach()
            ):
                telemetry[f"sdtec_scale_{layer_index}"] = scale
            tokens = outputs["sdtec_tokens"].detach()
            tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            similarity = torch.matmul(tokens, tokens.transpose(1, 2)).abs()
            token_count = similarity.shape[-1]
            off_diagonal = (
                similarity.sum(dim=(-1, -2))
                - similarity.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            ) / max(token_count * (token_count - 1), 1)
            telemetry["sdtec_token_similarity"] = off_diagonal.mean()
            metric_logger.update(**telemetry)
        elif "sdtec_spatial_gate" in outputs:
            telemetry = {
                "sdtec_gate": outputs["sdtec_spatial_gate"].detach().mean(),
                "sdtec_progress": outputs["sdtec_fusion_progress"].detach(),
            }
            for layer_index, scale in enumerate(
                outputs["sdtec_effective_scale_by_layer"].detach()
            ):
                telemetry[f"sdtec_scale_{layer_index}"] = scale
            metric_logger.update(**telemetry)

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)

    if use_wandb:
        wandb.log(
            {"lr": optimizer.param_groups[0]["lr"], "epoch": epoch, "train/loss": np.mean(losses)}
        )
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    epoch: int,
    use_wandb: bool,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        global_step = epoch * len(data_loader) + i

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            visual_samples = samples[:, :3] if samples.shape[1] == 6 else samples
            save_samples(
                visual_samples,
                targets,
                output_dir,
                "val",
                normalized=False,
                box_fmt="xyxy",
            )

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # orig_target_sizes = torch.tensor([[samples.shape[-1], samples.shape[-2]]], device=samples.device)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # validator format for metrics
        for idx, (target, result) in enumerate(zip(targets, results)):
            gt.append(
                {
                    "boxes": scale_boxes(  # from model input size to original img size
                        target["boxes"],
                        (target["orig_size"][1], target["orig_size"][0]),
                        (samples[idx].shape[-2], samples[idx].shape[-1]),
                    ),
                    "labels": target["labels"],
                }
            )
            labels = (
                torch.tensor([mscoco_category2label[int(x.item())] for x in result["labels"].flatten()])
                .to(result["labels"].device)
                .reshape(result["labels"].shape)
            ) if postprocessor.remap_mscoco_category else result["labels"]
            preds.append(
                {"boxes": result["boxes"], "labels": labels, "scores": result["scores"]}
            )

    # Conf matrix, F1, Precision, Recall, box IoU
    metrics = Validator(gt, preds).compute_metrics()
    print("Metrics:", metrics)
    if use_wandb:
        metrics = {f"metrics/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = epoch
        wandb.log(metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator
