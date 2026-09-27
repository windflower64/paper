"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
import math

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .dfine_utils import bbox2distance
from .sam_query_evidence_reader import sqer_shape_loss
from .stql_qcer import qcer_objectness_loss, stql_shape_loss


@register()
class DFINECriterion(nn.Module):
    """This class computes the loss for D-FINE."""

    __share__ = [
        "num_classes",
    ]
    __inject__ = [
        "matcher",
    ]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        spatial_aux_weight=0.0,
        spatial_aux_dice_weight=1.0,
        pdbr_boundary_aux_weight=0.0,
        pdbr_boundary_dice_weight=1.0,
        pdbr_boundary_sigma=0.75,
        tndp_detail_weight=0.0,
        tndp_neighborhood_radius=1,
        bpc_boundary_aux_weight=0.0,
        bpc_boundary_dice_weight=1.0,
        bpc_boundary_radius_pixels=4,
        qrl_region_aux_weight=0.0,
        qrl_region_dice_weight=1.0,
        qrl_region_decay_start=20,
        qrl_region_decay_end=45,
        qrl_region_radius_fraction=0.1,
        qrl_region_radius_min=1,
        qrl_region_radius_max=4,
        spar_aux_weight=0.0,
        spar_l1_weight=1.0,
        spar_dice_weight=2.0,
        spar_target_mode="mask",
        spar_mask_shift_fraction=0.0,
        spar_loss_resolution="feature",
        spar_decay_start=-1,
        spar_decay_end=-1,
        sabr_aux_weight=0.0,
        sabr_body_weight=0.5,
        sabr_boundary_weight=0.5,
        sabr_l1_weight=1.0,
        sabr_dice_weight=2.0,
        sabr_boundary_radius_fraction=0.25,
        sabr_boundary_radius_min=1,
        sabr_boundary_radius_max=3,
        sabr_decay_start=-1,
        sabr_decay_end=-1,
        sbox_aux_weight=0.0,
        sbox_dice_weight=1.0,
        sbox_focal_alpha=0.25,
        sbox_focal_gamma=2.0,
        sbox_extreme_band_pixels=3,
        sbox_neighborhood_radius=4,
        sbox_max_side_offset_cells=1.0,
        sbox_decay_start=30,
        sbox_decay_end=45,
        mdqa_aux_weight=0.0,
        mdqa_focal_weight=1.0,
        mdqa_dice_weight=1.0,
        mdqa_focal_alpha=0.25,
        mdqa_focal_gamma=2.0,
        sqmi_aux_weight=0.0,
        sbrd_aux_weight=0.0,
        sbrd_boundary_radius=1,
        sbrd_mask_shift_fraction=0.0,
        sbrd_region_mode="boundary",
        qcsr_teacher_weight=0.0,
        qcsr_transfer_weight=0.0,
        qcsr_boundary_radius=2,
        qcsr_incoherence_floor=0.25,
        qcsr_transfer_mode="raw_huber",
        qcsr_shape_eps=1e-4,
        srtod_reconstruction_weight=0.0,
        hbs_aux_weight=0.0,
        sdtec_presence_weight=0.0,
        sdtec_diversity_weight=0.0,
        sdtec_fallback_weight=0.0,
        sdtec_pair_rank_weight=0.0,
        sdtec_pair_rank_margin=0.02,
        sdtec_mismatch_suppression_weight=0.0,
        sdtec_delta_l2_weight=0.0,
        ma1_alignment_weight=0.0,
        ma1_gate_weight=0.0,
        ma1_mismatch_weight=0.0,
        stql_weight=0.0,
        stql_supervision="sam",
        stql_total_epochs=20,
        sqer_shape_weight=0.0,
        sqer_shape_start_epoch=0,
        sqer_supervision="sam",
        qcer_objectness_weight=0.0,
        fgl_edge_weight_mode="none",
        ddf_edge_reliability_mode="none",
        mal_alpha=None,
    ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            reg_max (int): Max number of the discrete bins in D-FINE.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.mal_alpha = mal_alpha
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.spatial_aux_weight = float(spatial_aux_weight)
        self.spatial_aux_dice_weight = float(spatial_aux_dice_weight)
        self.pdbr_boundary_aux_weight = float(pdbr_boundary_aux_weight)
        self.pdbr_boundary_dice_weight = float(pdbr_boundary_dice_weight)
        self.pdbr_boundary_sigma = float(pdbr_boundary_sigma)
        if self.pdbr_boundary_sigma <= 0:
            raise ValueError("pdbr_boundary_sigma must be positive")
        self.tndp_detail_weight = float(tndp_detail_weight)
        self.tndp_neighborhood_radius = int(tndp_neighborhood_radius)
        if self.tndp_detail_weight < 0:
            raise ValueError("tndp_detail_weight must be non-negative")
        if self.tndp_neighborhood_radius < 0:
            raise ValueError("tndp_neighborhood_radius must be non-negative")
        self.bpc_boundary_aux_weight = float(bpc_boundary_aux_weight)
        self.bpc_boundary_dice_weight = float(bpc_boundary_dice_weight)
        self.bpc_boundary_radius_pixels = int(bpc_boundary_radius_pixels)
        if self.bpc_boundary_aux_weight < 0:
            raise ValueError("bpc_boundary_aux_weight must be non-negative")
        if self.bpc_boundary_radius_pixels <= 0:
            raise ValueError("bpc_boundary_radius_pixels must be positive")
        self.qrl_region_aux_weight = float(qrl_region_aux_weight)
        self.qrl_region_dice_weight = float(qrl_region_dice_weight)
        self.qrl_region_decay_start = int(qrl_region_decay_start)
        self.qrl_region_decay_end = int(qrl_region_decay_end)
        self.qrl_region_radius_fraction = float(qrl_region_radius_fraction)
        self.qrl_region_radius_min = int(qrl_region_radius_min)
        self.qrl_region_radius_max = int(qrl_region_radius_max)
        if self.qrl_region_aux_weight < 0:
            raise ValueError("qrl_region_aux_weight must be non-negative")
        if not 0 <= self.qrl_region_decay_start < self.qrl_region_decay_end:
            raise ValueError("QRL region decay requires 0 <= start < end")
        if self.qrl_region_radius_fraction <= 0:
            raise ValueError("qrl_region_radius_fraction must be positive")
        if not 0 < self.qrl_region_radius_min <= self.qrl_region_radius_max:
            raise ValueError("QRL region radius bounds must satisfy 0 < min <= max")
        self.spar_aux_weight = float(spar_aux_weight)
        self.spar_l1_weight = float(spar_l1_weight)
        self.spar_dice_weight = float(spar_dice_weight)
        self.spar_target_mode = str(spar_target_mode)
        self.spar_mask_shift_fraction = float(spar_mask_shift_fraction)
        self.spar_loss_resolution = str(spar_loss_resolution)
        self.spar_decay_start = int(spar_decay_start)
        self.spar_decay_end = int(spar_decay_end)
        if min(self.spar_aux_weight, self.spar_l1_weight, self.spar_dice_weight) < 0:
            raise ValueError("SPAR loss weights must be non-negative")
        if self.spar_target_mode not in {"mask", "box"}:
            raise ValueError("spar_target_mode must be mask or box")
        if not -1.0 < self.spar_mask_shift_fraction < 1.0:
            raise ValueError("spar_mask_shift_fraction must be between -1 and 1")
        if self.spar_loss_resolution not in {"feature", "mask"}:
            raise ValueError("spar_loss_resolution must be feature or mask")
        if not (
            (self.spar_decay_start == -1 and self.spar_decay_end == -1)
            or 0 <= self.spar_decay_start < self.spar_decay_end
        ):
            raise ValueError(
                "SPAR decay must be disabled with -1/-1 or satisfy 0 <= start < end"
            )
        self.sabr_aux_weight = float(sabr_aux_weight)
        self.sabr_body_weight = float(sabr_body_weight)
        self.sabr_boundary_weight = float(sabr_boundary_weight)
        self.sabr_l1_weight = float(sabr_l1_weight)
        self.sabr_dice_weight = float(sabr_dice_weight)
        self.sabr_boundary_radius_fraction = float(sabr_boundary_radius_fraction)
        self.sabr_boundary_radius_min = int(sabr_boundary_radius_min)
        self.sabr_boundary_radius_max = int(sabr_boundary_radius_max)
        self.sabr_decay_start = int(sabr_decay_start)
        self.sabr_decay_end = int(sabr_decay_end)
        if min(
            self.sabr_aux_weight,
            self.sabr_body_weight,
            self.sabr_boundary_weight,
            self.sabr_l1_weight,
            self.sabr_dice_weight,
        ) < 0:
            raise ValueError("SABR loss weights must be non-negative")
        if self.sabr_aux_weight > 0 and self.sabr_body_weight + self.sabr_boundary_weight <= 0:
            raise ValueError("SABR requires a positive body or boundary weight")
        if self.sabr_boundary_radius_fraction <= 0:
            raise ValueError("sabr_boundary_radius_fraction must be positive")
        if not 0 < self.sabr_boundary_radius_min <= self.sabr_boundary_radius_max:
            raise ValueError("SABR boundary radius bounds must satisfy 0 < min <= max")
        if not (
            (self.sabr_decay_start == -1 and self.sabr_decay_end == -1)
            or 0 <= self.sabr_decay_start < self.sabr_decay_end
        ):
            raise ValueError(
                "SABR decay must be disabled with -1/-1 or satisfy 0 <= start < end"
            )
        self.sbox_aux_weight = float(sbox_aux_weight)
        self.sbox_dice_weight = float(sbox_dice_weight)
        self.sbox_focal_alpha = float(sbox_focal_alpha)
        self.sbox_focal_gamma = float(sbox_focal_gamma)
        self.sbox_extreme_band_pixels = int(sbox_extreme_band_pixels)
        self.sbox_neighborhood_radius = int(sbox_neighborhood_radius)
        self.sbox_max_side_offset_cells = float(sbox_max_side_offset_cells)
        self.sbox_decay_start = int(sbox_decay_start)
        self.sbox_decay_end = int(sbox_decay_end)
        if min(
            self.sbox_aux_weight,
            self.sbox_dice_weight,
            self.sbox_focal_gamma,
            self.sbox_max_side_offset_cells,
        ) < 0:
            raise ValueError("S-BOX weights, gamma, and offset must be non-negative")
        if not 0.0 <= self.sbox_focal_alpha <= 1.0:
            raise ValueError("sbox_focal_alpha must be in [0, 1]")
        if self.sbox_extreme_band_pixels <= 0:
            raise ValueError("sbox_extreme_band_pixels must be positive")
        if self.sbox_neighborhood_radius < 0:
            raise ValueError("sbox_neighborhood_radius must be non-negative")
        if not 0 <= self.sbox_decay_start < self.sbox_decay_end:
            raise ValueError("S-BOX decay requires 0 <= start < end")
        self.mdqa_aux_weight = float(mdqa_aux_weight)
        self.mdqa_focal_weight = float(mdqa_focal_weight)
        self.mdqa_dice_weight = float(mdqa_dice_weight)
        self.mdqa_focal_alpha = float(mdqa_focal_alpha)
        self.mdqa_focal_gamma = float(mdqa_focal_gamma)
        if min(
            self.mdqa_aux_weight,
            self.mdqa_focal_weight,
            self.mdqa_dice_weight,
            self.mdqa_focal_gamma,
        ) < 0:
            raise ValueError("MDQA loss weights and gamma must be non-negative")
        if not 0.0 <= self.mdqa_focal_alpha <= 1.0:
            raise ValueError("mdqa_focal_alpha must be in [0, 1]")
        self.sqmi_aux_weight = float(sqmi_aux_weight)
        if self.sqmi_aux_weight < 0:
            raise ValueError("sqmi_aux_weight must be non-negative")
        self.sbrd_aux_weight = float(sbrd_aux_weight)
        self.sbrd_boundary_radius = int(sbrd_boundary_radius)
        self.sbrd_mask_shift_fraction = float(sbrd_mask_shift_fraction)
        self.sbrd_region_mode = str(sbrd_region_mode)
        if self.sbrd_aux_weight < 0:
            raise ValueError("sbrd_aux_weight must be non-negative")
        if self.sbrd_boundary_radius <= 0:
            raise ValueError("sbrd_boundary_radius must be positive")
        if not -1.0 < self.sbrd_mask_shift_fraction < 1.0:
            raise ValueError("sbrd_mask_shift_fraction must be between -1 and 1")
        if self.sbrd_region_mode not in {"boundary", "incoherent_boundary"}:
            raise ValueError(
                "sbrd_region_mode must be boundary or incoherent_boundary"
            )
        self.qcsr_teacher_weight = float(qcsr_teacher_weight)
        self.qcsr_transfer_weight = float(qcsr_transfer_weight)
        self.qcsr_boundary_radius = int(qcsr_boundary_radius)
        self.qcsr_incoherence_floor = float(qcsr_incoherence_floor)
        self.qcsr_transfer_mode = str(qcsr_transfer_mode)
        self.qcsr_shape_eps = float(qcsr_shape_eps)
        if min(self.qcsr_teacher_weight, self.qcsr_transfer_weight) < 0:
            raise ValueError("QCSR loss weights must be non-negative")
        if self.qcsr_boundary_radius <= 0:
            raise ValueError("qcsr_boundary_radius must be positive")
        if not 0.0 <= self.qcsr_incoherence_floor <= 1.0:
            raise ValueError("qcsr_incoherence_floor must be in [0, 1]")
        if self.qcsr_transfer_mode not in {"raw_huber", "shape_correlation"}:
            raise ValueError(
                "qcsr_transfer_mode must be raw_huber or shape_correlation"
            )
        if self.qcsr_shape_eps <= 0:
            raise ValueError("qcsr_shape_eps must be positive")
        self.srtod_reconstruction_weight = float(srtod_reconstruction_weight)
        self.hbs_aux_weight = float(hbs_aux_weight)
        self.sdtec_presence_weight = float(sdtec_presence_weight)
        self.sdtec_diversity_weight = float(sdtec_diversity_weight)
        self.sdtec_fallback_weight = float(sdtec_fallback_weight)
        self.sdtec_pair_rank_weight = float(sdtec_pair_rank_weight)
        self.sdtec_pair_rank_margin = float(sdtec_pair_rank_margin)
        self.sdtec_mismatch_suppression_weight = float(
            sdtec_mismatch_suppression_weight
        )
        self.sdtec_delta_l2_weight = float(sdtec_delta_l2_weight)
        self.ma1_alignment_weight = float(ma1_alignment_weight)
        self.ma1_gate_weight = float(ma1_gate_weight)
        self.ma1_mismatch_weight = float(ma1_mismatch_weight)
        self.stql_weight = float(stql_weight)
        self.stql_supervision = str(stql_supervision)
        self.stql_total_epochs = int(stql_total_epochs)
        self.sqer_shape_weight = float(sqer_shape_weight)
        self.sqer_shape_start_epoch = int(sqer_shape_start_epoch)
        self.training_epoch = 0
        self.sqer_supervision = str(sqer_supervision)
        self.qcer_objectness_weight = float(qcer_objectness_weight)
        if min(self.stql_weight, self.sqer_shape_weight, self.qcer_objectness_weight) < 0:
            raise ValueError("STQL/S-QER/QCER loss weights must be non-negative")
        if self.stql_supervision not in {"sam", "box"}:
            raise ValueError("stql_supervision must be 'sam' or 'box'")
        if self.sqer_supervision not in {"sam", "box"}:
            raise ValueError("sqer_supervision must be 'sam' or 'box'")
        if self.stql_total_epochs <= 0:
            raise ValueError("stql_total_epochs must be positive")
        if self.sqer_shape_start_epoch < 0:
            raise ValueError("sqer_shape_start_epoch must be non-negative")
        if min(
            self.sdtec_presence_weight,
            self.sdtec_diversity_weight,
            self.sdtec_fallback_weight,
            self.sdtec_pair_rank_weight,
            self.sdtec_pair_rank_margin,
            self.sdtec_mismatch_suppression_weight,
            self.sdtec_delta_l2_weight,
            self.ma1_alignment_weight,
            self.ma1_gate_weight,
            self.ma1_mismatch_weight,
        ) < 0:
            raise ValueError("SDTEC/MA1 loss weights must be non-negative")
        if fgl_edge_weight_mode not in {"none", "shape", "shape_reverse"}:
            raise ValueError(
                "fgl_edge_weight_mode must be none, shape, or shape_reverse, "
                f"got {fgl_edge_weight_mode!r}"
            )
        if ddf_edge_reliability_mode not in {"none", "entropy"}:
            raise ValueError(
                "ddf_edge_reliability_mode must be none or entropy, "
                f"got {ddf_edge_reliability_mode!r}"
            )
        self.fgl_edge_weight_mode = str(fgl_edge_weight_mode)
        self.ddf_edge_reliability_mode = str(ddf_edge_reliability_mode)
        self.num_pos, self.num_neg = None, None

    def _fgl_edge_weights(self, target_boxes, dtype):
        """Return mean-one weights in FDR order: left, top, right, bottom."""
        width = target_boxes[:, 2].clamp_min(1e-12)
        height = target_boxes[:, 3].clamp_min(1e-12)
        horizontal_edge = 2 * height / (width + height)
        vertical_edge = 2 * width / (width + height)
        if self.fgl_edge_weight_mode == "shape_reverse":
            horizontal_edge, vertical_edge = vertical_edge, horizontal_edge
        weights = torch.stack(
            (horizontal_edge, vertical_edge, horizontal_edge, vertical_edge), dim=-1
        )
        return weights.to(dtype=dtype).reshape(-1).detach()

    def _ddf_edge_reliability(self, teacher_corners):
        """Mean-one detached reliability from normalized teacher edge entropy."""
        probabilities = F.softmax(teacher_corners.detach(), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        reliability = 1 - entropy / math.log(self.reg_max + 1)
        reliability = reliability.clamp_min(1e-6)
        reliability = reliability / reliability.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        return reliability.detach()

    def _spar_union_target(self, instance_masks, device):
        """Build the class-agnostic SPAR target for one transformed image."""
        if instance_masks.numel() == 0:
            return torch.zeros(
                (1, *instance_masks.shape[-2:]),
                device=device,
                dtype=torch.float32,
            )

        instance_masks = instance_masks.to(device)
        if self.spar_target_mode == "mask":
            return instance_masks.float().amax(dim=0, keepdim=True)

        # Use each accepted SAM instance's own enclosing rectangle. This
        # preserves its transformed position and extent while removing contour
        # shape, making a cleaner causal control than detector boxes whose
        # annotation may be inaccurate.
        binary = instance_masks > 0.5
        valid_instances = binary.flatten(1).any(dim=1)
        height, width = binary.shape[-2:]
        rows = binary.any(dim=2)
        columns = binary.any(dim=1)
        y = torch.arange(height, device=binary.device)
        x = torch.arange(width, device=binary.device)
        y_min = torch.where(rows, y[None], height).amin(dim=1)
        y_max = torch.where(rows, y[None], -1).amax(dim=1)
        x_min = torch.where(columns, x[None], width).amin(dim=1)
        x_max = torch.where(columns, x[None], -1).amax(dim=1)
        box_masks = (
            (y[None, :, None] >= y_min[:, None, None])
            & (y[None, :, None] <= y_max[:, None, None])
            & (x[None, None, :] >= x_min[:, None, None])
            & (x[None, None, :] <= x_max[:, None, None])
            & valid_instances[:, None, None]
        )
        return box_masks.any(dim=0, keepdim=True).float()

    def _spar_loss(self, fused_features, targets):
        """FALCON-SPAR L1+Dice loss with batch-safe normalization.

        The official repository computes this map and loss but returns zero.
        REP1 follows the paper equation and excludes samples without an
        accepted foreground mask.
        """
        fused_features = fused_features.float()
        activation = fused_features.mean(dim=1, keepdim=True)
        flat = activation.flatten(1)
        minimum = flat.amin(dim=1).view(-1, 1, 1, 1)
        maximum = flat.amax(dim=1).view(-1, 1, 1, 1)
        activation = 1.0 - (activation - minimum) / (maximum - minimum).clamp_min(1e-6)

        predictions = []
        masks = []
        valid = []
        for batch_index, target in enumerate(targets):
            if "masks" not in target:
                raise RuntimeError("SPAR training requires transformed SAM masks")
            instance_masks = target["masks"]
            union = self._spar_union_target(instance_masks, activation.device)
            if self.spar_mask_shift_fraction:
                shift_pixels = int(
                    round(union.shape[-1] * self.spar_mask_shift_fraction)
                )
                union = torch.roll(union, shifts=shift_pixels, dims=-1)
            valid.append(bool(union.sum().detach() > 0))
            if self.spar_loss_resolution == "mask":
                # Match the public FALCON implementation: preserve the full
                # transformed SAM target and upsample the prediction to it.
                prediction = F.interpolate(
                    activation[batch_index : batch_index + 1],
                    size=union.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                target_mask = union.clamp(0.0, 1.0)
            else:
                prediction = activation[batch_index]
                target_mask = F.interpolate(
                    union.unsqueeze(0),
                    size=activation.shape[-2:],
                    mode="area",
                ).squeeze(0).clamp(0.0, 1.0)
            predictions.append(prediction)
            masks.append(target_mask)

        activation = torch.stack(predictions, dim=0)
        target_mask = torch.stack(masks, dim=0)
        valid = torch.as_tensor(valid, device=activation.device, dtype=torch.bool)
        if not bool(valid.any()):
            return activation.sum() * 0.0

        l1 = (activation - target_mask).abs().mean(dim=(1, 2, 3))
        intersection = (activation * target_mask).sum(dim=(1, 2, 3))
        denominator = activation.sum(dim=(1, 2, 3)) + target_mask.sum(dim=(1, 2, 3))
        dice = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        per_sample = self.spar_l1_weight * l1 + self.spar_dice_weight * dice
        return per_sample[valid].mean()

    def _sabr_targets(self, boundary_logits, body_logits, targets):
        """Build scale-specific targets: a wide contour at S8 and body at S16."""
        boundary_targets = []
        body_targets = []
        valid = []
        for target in targets:
            if "masks" not in target:
                raise RuntimeError("SABR training requires transformed SAM masks")
            masks = target["masks"].float().to(boundary_logits.device)
            has_foreground = masks.numel() > 0 and bool(masks.any())
            valid.append(has_foreground)
            if not has_foreground:
                boundary_targets.append(boundary_logits.new_zeros(boundary_logits.shape[-3:]))
                body_targets.append(body_logits.new_zeros(body_logits.shape[-3:]))
                continue

            body = masks.amax(dim=0, keepdim=True)
            body = F.interpolate(
                body.unsqueeze(0), size=body_logits.shape[-2:], mode="area"
            ).squeeze(0).clamp(0.0, 1.0)
            body_targets.append(body)

            instances = F.interpolate(
                masks.unsqueeze(1), size=boundary_logits.shape[-2:], mode="area"
            )
            instance_bands = []
            for instance in instances:
                area = instance.sum().detach().float().sqrt().item()
                radius = int(round(area * self.sabr_boundary_radius_fraction))
                radius = max(
                    self.sabr_boundary_radius_min,
                    min(self.sabr_boundary_radius_max, radius),
                )
                kernel = 2 * radius + 1
                dilation = F.max_pool2d(instance.unsqueeze(0), kernel, 1, radius)
                erosion = -F.max_pool2d(-instance.unsqueeze(0), kernel, 1, radius)
                instance_bands.append((dilation - erosion).squeeze(0).clamp(0.0, 1.0))
            boundary_targets.append(torch.stack(instance_bands, dim=0).amax(dim=0))

        return (
            torch.stack(boundary_targets, dim=0),
            torch.stack(body_targets, dim=0),
            torch.as_tensor(valid, device=boundary_logits.device, dtype=torch.bool),
        )

    def _sabr_map_loss(self, logits, target, valid):
        probability = logits.float().sigmoid()
        if not bool(valid.any()):
            return probability.sum() * 0.0
        l1 = (probability - target).abs().mean(dim=(1, 2, 3))
        intersection = (probability * target).sum(dim=(1, 2, 3))
        denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        per_sample = self.sabr_l1_weight * l1 + self.sabr_dice_weight * dice
        return per_sample[valid].mean()

    def _sabr_loss_components(self, boundary_logits, body_logits, targets):
        boundary_target, body_target, valid = self._sabr_targets(
            boundary_logits, body_logits, targets
        )
        boundary_loss = self._sabr_map_loss(boundary_logits, boundary_target, valid)
        body_loss = self._sabr_map_loss(body_logits, body_target, valid)
        return {
            "boundary": boundary_loss,
            "body": body_loss,
            "combined": (
                self.sabr_boundary_weight * boundary_loss
                + self.sabr_body_weight * body_loss
            ),
            "boundary_target": boundary_target,
            "body_target": body_target,
            "valid": valid,
        }

    def _mdqa_loss(self, query_embeddings, pixel_features, targets, indices):
        """Mask-DINO-style mask loss on final-layer matched detector queries.

        Only a query selected by the detector's ordinary Hungarian matching is
        allowed to see the SAM mask of its paired target. Samples without an
        accepted SAM mask contribute exactly zero.
        """
        query_embeddings = query_embeddings.float()
        pixel_features = pixel_features.float()
        if query_embeddings.ndim != 3 or pixel_features.ndim != 4:
            raise ValueError(
                "MDQA expects [B,Q,D] query embeddings and [B,D,H,W] pixels"
            )
        if query_embeddings.shape[0] != pixel_features.shape[0]:
            raise ValueError("MDQA query/pixel batch sizes do not match")
        if query_embeddings.shape[-1] != pixel_features.shape[1]:
            raise ValueError("MDQA query/pixel embedding dimensions do not match")

        focal_terms = []
        dice_terms = []
        scale = math.sqrt(float(query_embeddings.shape[-1]))
        height, width = pixel_features.shape[-2:]

        for batch_index, (source_indices, target_indices) in enumerate(indices):
            target = targets[batch_index]
            if "masks" not in target:
                raise RuntimeError("MDQA training requires transformed SAM masks")
            instance_masks = target["masks"]
            if source_indices.numel() == 0 or instance_masks.numel() == 0:
                continue

            accepted_source = []
            accepted_target = []
            for source_index, target_index in zip(
                source_indices.tolist(), target_indices.tolist()
            ):
                mask = instance_masks[target_index]
                if bool(mask.any()):
                    accepted_source.append(source_index)
                    accepted_target.append(target_index)
            if not accepted_source:
                continue

            source_tensor = torch.as_tensor(
                accepted_source,
                device=query_embeddings.device,
                dtype=torch.long,
            )
            target_tensor = torch.as_tensor(
                accepted_target,
                device=instance_masks.device,
                dtype=torch.long,
            )
            matched_queries = query_embeddings[batch_index, source_tensor]
            logits = torch.einsum(
                "nd,dhw->nhw", matched_queries, pixel_features[batch_index]
            ) / scale
            target_masks = instance_masks[target_tensor].float().to(logits.device)
            target_masks = F.interpolate(
                target_masks.unsqueeze(1),
                size=(height, width),
                mode="area",
            ).squeeze(1).clamp(0.0, 1.0)

            cross_entropy = F.binary_cross_entropy_with_logits(
                logits, target_masks, reduction="none"
            )
            probabilities = logits.sigmoid()
            probability_t = probabilities * target_masks + (
                1.0 - probabilities
            ) * (1.0 - target_masks)
            alpha_t = self.mdqa_focal_alpha * target_masks + (
                1.0 - self.mdqa_focal_alpha
            ) * (1.0 - target_masks)
            focal = alpha_t * (1.0 - probability_t).pow(
                self.mdqa_focal_gamma
            ) * cross_entropy
            focal_terms.append(focal.mean(dim=(1, 2)))

            intersection = (probabilities * target_masks).sum(dim=(1, 2))
            denominator = probabilities.sum(dim=(1, 2)) + target_masks.sum(
                dim=(1, 2)
            )
            dice_terms.append(
                1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
            )

        if not focal_terms:
            return (query_embeddings.sum() + pixel_features.sum()) * 0.0
        focal_loss = torch.cat(focal_terms).mean()
        dice_loss = torch.cat(dice_terms).mean()
        return self.mdqa_focal_weight * focal_loss + self.mdqa_dice_weight * dice_loss

    @staticmethod
    def _sbrd_response(feature):
        """Channel-agnostic spatial response, normalized independently per image."""
        response = feature.float().square().mean(dim=1, keepdim=True).sqrt()
        flat = response.flatten(1)
        minimum = flat.amin(dim=1).view(-1, 1, 1, 1)
        maximum = flat.amax(dim=1).view(-1, 1, 1, 1)
        return (response - minimum) / (maximum - minimum).clamp_min(1e-6)

    def _sbrd_boundary_weights(self, reference, targets):
        """Build equal-instance SAM boundary weights on the S8 feature grid."""
        sample_weights = []
        valid = []
        radius = self.sbrd_boundary_radius
        kernel_size = 2 * radius + 1
        for target in targets:
            if "masks" not in target:
                raise RuntimeError("SBRD training requires transformed SAM masks")
            masks = target["masks"]
            instance_weights = []
            for mask in masks:
                mask = mask.float().to(reference.device).unsqueeze(0).unsqueeze(0)
                mask = F.interpolate(
                    mask,
                    size=reference.shape[-2:],
                    mode="area",
                )
                dilated = F.max_pool2d(
                    mask,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=radius,
                )
                eroded = 1.0 - F.max_pool2d(
                    1.0 - mask,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=radius,
                )
                boundary = (dilated - eroded).clamp(0.0, 1.0)
                if self.sbrd_region_mode == "incoherent_boundary":
                    # Retain only boundary pixels whose SAM geometry changes
                    # under the exact S8->S16->S8 sampling operation.
                    low_resolution = F.interpolate(
                        mask,
                        size=(
                            max(1, mask.shape[-2] // 2),
                            max(1, mask.shape[-1] // 2),
                        ),
                        mode="area",
                    )
                    reconstructed = F.interpolate(
                        low_resolution,
                        size=mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    incoherence = (mask - reconstructed).abs().clamp(0.0, 1.0)
                    boundary = boundary * incoherence
                if self.sbrd_mask_shift_fraction:
                    shift_pixels = int(
                        round(boundary.shape[-1] * self.sbrd_mask_shift_fraction)
                    )
                    boundary = torch.roll(boundary, shifts=shift_pixels, dims=-1)
                mass = boundary.sum()
                if bool(mass.detach() > 0):
                    instance_weights.append(boundary / mass.clamp_min(1e-6))
            if instance_weights:
                sample_weights.append(torch.stack(instance_weights).mean(dim=0).squeeze(0))
                valid.append(True)
            else:
                sample_weights.append(reference.new_zeros((1, *reference.shape[-2:])))
                valid.append(False)
        return torch.stack(sample_weights), torch.as_tensor(
            valid, device=reference.device, dtype=torch.bool
        )

    def _sbrd_loss(self, stage_features, targets):
        """Preserve the S8 boundary response after the ordinary S8->S16 sampling."""
        s8, s16 = stage_features
        teacher = self._sbrd_response(s8).detach()
        student = F.interpolate(
            self._sbrd_response(s16),
            size=teacher.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        boundary_weights, valid = self._sbrd_boundary_weights(teacher, targets)
        if not bool(valid.any()):
            return student.sum() * 0.0
        per_sample = ((student - teacher).abs() * boundary_weights).sum(
            dim=(1, 2, 3)
        )
        return per_sample[valid].mean()

    def _qcsr_transfer_term(self, student_logits, teacher_logits, region):
        """Compare one query-conditioned student/teacher response pair.

        ``raw_huber`` preserves the original REP3 objective.  The
        ``shape_correlation`` variant removes the weighted mean and scale of
        each response independently, so a teacher that merely becomes more
        confident cannot make the transfer target grow without bound.
        """
        mass = region.sum().clamp_min(1e-6)
        teacher_logits = teacher_logits.detach()
        if self.qcsr_transfer_mode == "raw_huber":
            transfer_map = F.smooth_l1_loss(
                student_logits,
                teacher_logits,
                reduction="none",
                beta=0.5,
            )
            return (transfer_map * region).sum() / mass

        def normalize(value):
            mean = (value * region).sum() / mass
            centered = value - mean
            variance = (centered.square() * region).sum() / mass
            return centered / variance.clamp_min(self.qcsr_shape_eps).sqrt()

        student_shape = normalize(student_logits)
        teacher_shape = normalize(teacher_logits)
        correlation = (
            (student_shape * teacher_shape * region).sum() / mass
        ).clamp(-1.0, 1.0)
        # Bounded to [0, 1]: identical spatial shape -> 0, inverse shape -> 1.
        return 0.5 * (1.0 - correlation)

    def _qcsr_losses(
        self,
        query_embeddings,
        teacher_pixels,
        student_pixels,
        targets,
        indices,
    ):
        """Anchor a detached S8 teacher with SAM, then retain it at S16.

        SAM trains only the private query/S8 teacher heads.  The detector sees
        no full-mask reconstruction gradient.  Its only auxiliary gradient is
        the student-side cross-scale loss inside a soft, downsampling-sensitive
        boundary neighborhood.
        """
        query_embeddings = query_embeddings.float()
        teacher_pixels = teacher_pixels.float()
        student_pixels = F.interpolate(
            student_pixels.float(),
            size=teacher_pixels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if query_embeddings.shape[0] != teacher_pixels.shape[0]:
            raise ValueError("QCSR query/teacher batch sizes do not match")
        if teacher_pixels.shape != student_pixels.shape:
            raise ValueError("QCSR teacher/student projected shapes do not match")
        if query_embeddings.shape[-1] != teacher_pixels.shape[1]:
            raise ValueError("QCSR embedding dimensions do not match")

        teacher_focal_terms = []
        teacher_dice_terms = []
        transfer_terms = []
        scale = math.sqrt(float(query_embeddings.shape[-1]))
        height, width = teacher_pixels.shape[-2:]
        radius = self.qcsr_boundary_radius
        kernel_size = 2 * radius + 1

        for batch_index, (source_indices, target_indices) in enumerate(indices):
            target = targets[batch_index]
            if "masks" not in target:
                raise RuntimeError("QCSR training requires transformed SAM masks")
            masks = target["masks"]
            for source_index, target_index in zip(
                source_indices.tolist(), target_indices.tolist()
            ):
                mask = masks[target_index]
                if not bool(mask.any()):
                    continue
                mask = F.interpolate(
                    mask.float().to(teacher_pixels.device)[None, None],
                    size=(height, width),
                    mode="area",
                ).clamp(0.0, 1.0)
                query = query_embeddings[batch_index, source_index]
                teacher_logits = torch.einsum(
                    "d,dhw->hw", query, teacher_pixels[batch_index]
                ) / scale

                # Full SAM supervision is confined to private teacher heads.
                target_mask = mask[0, 0]
                cross_entropy = F.binary_cross_entropy_with_logits(
                    teacher_logits, target_mask, reduction="none"
                )
                teacher_probability = teacher_logits.sigmoid()
                probability_t = teacher_probability * target_mask + (
                    1.0 - teacher_probability
                ) * (1.0 - target_mask)
                alpha_t = self.mdqa_focal_alpha * target_mask + (
                    1.0 - self.mdqa_focal_alpha
                ) * (1.0 - target_mask)
                teacher_focal_terms.append(
                    (
                        alpha_t
                        * (1.0 - probability_t).pow(self.mdqa_focal_gamma)
                        * cross_entropy
                    ).mean()
                )
                intersection = (teacher_probability * target_mask).sum()
                denominator = teacher_probability.sum() + target_mask.sum()
                teacher_dice_terms.append(
                    1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
                )

                # The transfer target and query condition are detached.  This
                # loss can only train the student projection and S8->S16 path.
                student_logits = torch.einsum(
                    "d,dhw->hw",
                    query.detach(),
                    student_pixels[batch_index],
                ) / scale
                dilated = F.max_pool2d(
                    mask, kernel_size=kernel_size, stride=1, padding=radius
                )
                eroded = 1.0 - F.max_pool2d(
                    1.0 - mask,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=radius,
                )
                boundary = (dilated - eroded).clamp(0.0, 1.0)
                low = F.interpolate(
                    mask,
                    size=(max(1, height // 2), max(1, width // 2)),
                    mode="area",
                )
                reconstructed = F.interpolate(
                    low,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                incoherence = (mask - reconstructed).abs().clamp(0.0, 1.0)
                region = boundary * (
                    self.qcsr_incoherence_floor
                    + (1.0 - self.qcsr_incoherence_floor) * incoherence
                )
                mass = region.sum()
                if bool(mass.detach() > 0):
                    transfer_terms.append(
                        self._qcsr_transfer_term(
                            student_logits,
                            teacher_logits,
                            region[0, 0],
                        )
                    )

        zero = (query_embeddings.sum() + teacher_pixels.sum() + student_pixels.sum()) * 0.0
        if teacher_focal_terms:
            teacher_loss = torch.stack(teacher_focal_terms).mean() + torch.stack(
                teacher_dice_terms
            ).mean()
        else:
            teacher_loss = zero
        transfer_loss = torch.stack(transfer_terms).mean() if transfer_terms else zero
        return teacher_loss, transfer_loss

    def _pdbr_boundary_targets(self, logits_x, logits_y, targets):
        """Create soft LR/TB box-side targets on the pre-downsample grid."""
        target_x = torch.zeros_like(logits_x)
        target_y = torch.zeros_like(logits_y)
        height, width = logits_x.shape[-2:]
        yy = torch.arange(height, device=logits_x.device, dtype=logits_x.dtype).view(height, 1)
        xx = torch.arange(width, device=logits_x.device, dtype=logits_x.dtype).view(1, width)
        sigma = self.pdbr_boundary_sigma
        for batch_idx, target in enumerate(targets):
            for cx, cy, bw, bh in target["boxes"]:
                x1 = (cx - bw / 2) * width
                x2 = (cx + bw / 2) * width
                y1 = (cy - bh / 2) * height
                y2 = (cy + bh / 2) * height
                vertical_extent = ((yy >= y1 - sigma) & (yy <= y2 + sigma)).to(logits_x.dtype)
                horizontal_extent = ((xx >= x1 - sigma) & (xx <= x2 + sigma)).to(logits_x.dtype)
                lr_distance = torch.minimum((xx - x1).abs(), (xx - x2).abs())
                tb_distance = torch.minimum((yy - y1).abs(), (yy - y2).abs())
                lr = torch.exp(-0.5 * (lr_distance / sigma).square()) * vertical_extent
                tb = torch.exp(-0.5 * (tb_distance / sigma).square()) * horizontal_extent
                target_x[batch_idx, 0] = torch.maximum(target_x[batch_idx, 0], lr)
                target_y[batch_idx, 0] = torch.maximum(target_y[batch_idx, 0], tb)
        return target_x, target_y

    def _pdbr_boundary_loss(self, logits, target):
        focal_map = torchvision.ops.sigmoid_focal_loss(
            logits, target, alpha=0.25, gamma=2.0, reduction="none"
        )
        positive_mass = target.sum((1, 2, 3)).clamp(min=1)
        negative_mass = (1 - target).sum((1, 2, 3)).clamp(min=1)
        focal_pos = (focal_map * target).sum((1, 2, 3)) / positive_mass
        focal_neg = (focal_map * (1 - target)).sum((1, 2, 3)) / negative_mass
        focal = (focal_pos + focal_neg).mean()
        probability = logits.sigmoid()
        intersection = (probability * target).sum((1, 2, 3))
        denominator = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
        dice = (1 - (2 * intersection + 1) / (denominator + 1)).mean()
        return focal + self.pdbr_boundary_dice_weight * dice

    def _bpc_boundary_targets(self, logits, targets):
        """Build per-instance mask boundary bands and mark rejected masks invalid."""
        boundary_targets = []
        valid_samples = []
        radius = self.bpc_boundary_radius_pixels
        kernel_size = 2 * radius + 1
        for target in targets:
            if "masks" not in target:
                raise RuntimeError("BPC training requires transformed SAM masks")
            masks = target["masks"]
            if masks.shape[0] == 0:
                # A true negative image is valid zero supervision.
                boundary = logits.new_zeros((1, *masks.shape[-2:]))
                valid = True
            elif not bool(masks.any()):
                # A positive image with an all-zero loaded mask is a rejected
                # SAM pseudo label, not a background example.
                boundary = logits.new_zeros((1, *masks.shape[-2:]))
                valid = False
            else:
                masks_float = masks.float().to(logits.device).unsqueeze(1)
                dilated = F.max_pool2d(
                    masks_float,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=radius,
                )
                eroded = 1.0 - F.max_pool2d(
                    1.0 - masks_float,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=radius,
                )
                instance_bands = (dilated - eroded).clamp(0.0, 1.0)
                boundary = instance_bands.amax(dim=0)
                valid = True
            boundary = F.interpolate(
                boundary.unsqueeze(0),
                size=logits.shape[-2:],
                mode="area",
            ).squeeze(0)
            boundary_targets.append(boundary.clamp(0.0, 1.0))
            valid_samples.append(valid)
        boundary_target = torch.stack(boundary_targets)
        valid = torch.tensor(valid_samples, device=logits.device, dtype=logits.dtype)
        return boundary_target, valid

    def _bpc_boundary_loss(self, logits, target, valid):
        focal_map = torchvision.ops.sigmoid_focal_loss(
            logits, target, alpha=0.25, gamma=2.0, reduction="none"
        )
        positive_mass = target.sum((1, 2, 3)).clamp(min=1)
        negative_mass = (1 - target).sum((1, 2, 3)).clamp(min=1)
        focal_pos = (focal_map * target).sum((1, 2, 3)) / positive_mass
        focal_neg = (focal_map * (1 - target)).sum((1, 2, 3)) / negative_mass
        focal = focal_pos + focal_neg
        probability = logits.sigmoid()
        intersection = (probability * target).sum((1, 2, 3))
        denominator = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
        dice = 1 - (2 * intersection + 1) / (denominator + 1)
        per_sample = focal + self.bpc_boundary_dice_weight * dice
        return (per_sample * valid).sum() / valid.sum().clamp_min(1.0)

    def _qrl_region_schedule(self, epoch):
        epoch = int(epoch)
        if epoch < self.qrl_region_decay_start:
            return 1.0
        if epoch >= self.qrl_region_decay_end:
            return 0.0
        progress = (epoch - self.qrl_region_decay_start) / float(
            self.qrl_region_decay_end - self.qrl_region_decay_start
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _spar_schedule(self, epoch):
        if self.spar_decay_start < 0:
            return 1.0
        epoch = int(epoch)
        if epoch <= self.spar_decay_start:
            return 1.0
        if epoch >= self.spar_decay_end:
            return 0.0
        progress = (epoch - self.spar_decay_start) / float(
            self.spar_decay_end - self.spar_decay_start
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _sabr_schedule(self, epoch):
        if self.sabr_decay_start < 0:
            return 1.0
        epoch = int(epoch)
        if epoch <= self.sabr_decay_start:
            return 1.0
        if epoch >= self.sabr_decay_end:
            return 0.0
        progress = (epoch - self.sabr_decay_start) / float(
            self.sabr_decay_end - self.sabr_decay_start
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _sbox_schedule(self, epoch):
        epoch = int(epoch)
        if epoch <= self.sbox_decay_start:
            return 1.0
        if epoch >= self.sbox_decay_end:
            return 0.0
        progress = (epoch - self.sbox_decay_start) / float(
            self.sbox_decay_end - self.sbox_decay_start
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _sbox_targets(self, logits, targets):
        """Build left/top/right/bottom SAM extreme-point heatmaps at S8."""
        if logits.ndim != 4 or logits.shape[1] != 4:
            raise ValueError("S-BOX logits must have shape [B,4,H,W]")
        batch_targets = []
        neighborhoods = []
        valid_sides = []
        out_height, out_width = logits.shape[-2:]
        for target in targets:
            if "masks" not in target:
                raise RuntimeError("S-BOX training requires transformed SAM masks")
            if "sam_quality" not in target:
                raise RuntimeError("S-BOX training target is missing sam_quality")
            masks = target["masks"].float().to(logits.device)
            if masks.ndim != 3:
                raise ValueError("S-BOX instance masks must have shape [N,H,W]")
            height, width = masks.shape[-2:]
            heatmaps = logits.new_zeros((4, height, width))
            union = logits.new_zeros((1, height, width))
            side_valid = logits.new_zeros(4)
            quality = float(target["sam_quality"].reshape(-1)[0].detach())
            boxes = target["boxes"]
            if masks.shape[0] != boxes.shape[0]:
                raise ValueError("S-BOX masks and boxes must have the same count")

            for instance_index, instance in enumerate(masks):
                foreground = instance > 0.5
                coordinates = foreground.nonzero(as_tuple=False)
                if coordinates.numel() == 0 or quality <= 0:
                    continue
                ys, xs = coordinates[:, 0], coordinates[:, 1]
                min_y, max_y = int(ys.min()), int(ys.max())
                min_x, max_x = int(xs.min()), int(xs.max())
                band = self.sbox_extreme_band_pixels
                yy = torch.arange(height, device=logits.device)[:, None]
                xx = torch.arange(width, device=logits.device)[None, :]
                directions = (
                    foreground & (xx <= min_x + band),
                    foreground & (yy <= min_y + band),
                    foreground & (xx >= max_x - band),
                    foreground & (yy >= max_y - band),
                )
                for direction, region in enumerate(directions):
                    region = F.max_pool2d(
                        region.float()[None, None], 3, stride=1, padding=1
                    )[0, 0]
                    heatmaps[direction] = torch.maximum(
                        heatmaps[direction], region
                    )
                union[0] = torch.maximum(union[0], foreground.float())

                cx, cy, box_width, box_height = boxes[instance_index]
                box_sides = (
                    float((cx - box_width / 2) * width),
                    float((cy - box_height / 2) * height),
                    float((cx + box_width / 2) * width),
                    float((cy + box_height / 2) * height),
                )
                mask_sides = (float(min_x), float(min_y), float(max_x + 1), float(max_y + 1))
                strides = (width / out_width, height / out_height) * 2
                for direction, (mask_side, box_side, stride) in enumerate(
                    zip(mask_sides, box_sides, strides)
                ):
                    offset_cells = abs(mask_side - box_side) / max(stride, 1e-6)
                    if offset_cells <= self.sbox_max_side_offset_cells:
                        side_valid[direction] = max(
                            float(side_valid[direction]), quality
                        )

            heatmaps = F.interpolate(
                heatmaps.unsqueeze(0),
                size=(out_height, out_width),
                mode="area",
            ).squeeze(0).clamp(0.0, 1.0)
            neighborhood = F.interpolate(
                union.unsqueeze(0),
                size=(out_height, out_width),
                mode="area",
            ).squeeze(0)
            if self.sbox_neighborhood_radius > 0:
                radius = self.sbox_neighborhood_radius
                neighborhood = F.max_pool2d(
                    neighborhood.unsqueeze(0),
                    2 * radius + 1,
                    stride=1,
                    padding=radius,
                ).squeeze(0)
            neighborhood = torch.maximum(
                neighborhood.expand(4, -1, -1), heatmaps
            ).clamp(0.0, 1.0)
            batch_targets.append(heatmaps)
            neighborhoods.append(neighborhood)
            valid_sides.append(side_valid)

        return (
            torch.stack(batch_targets),
            torch.stack(neighborhoods),
            torch.stack(valid_sides),
        )

    def _sbox_loss_components(self, logits, targets):
        logits = logits.float()
        target, neighborhood, valid = self._sbox_targets(logits, targets)
        focal_map = torchvision.ops.sigmoid_focal_loss(
            logits,
            target,
            alpha=self.sbox_focal_alpha,
            gamma=self.sbox_focal_gamma,
            reduction="none",
        )
        positive_mass = target.sum((-2, -1)).clamp_min(1.0)
        negative_weight = neighborhood * (1.0 - target)
        negative_mass = negative_weight.sum((-2, -1)).clamp_min(1.0)
        focal_positive = (focal_map * target).sum((-2, -1)) / positive_mass
        focal_negative = (focal_map * negative_weight).sum((-2, -1)) / negative_mass
        focal = focal_positive + focal_negative

        probability = logits.sigmoid() * neighborhood
        intersection = (probability * target).sum((-2, -1))
        denominator = probability.sum((-2, -1)) + target.sum((-2, -1))
        dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        per_side = focal + self.sbox_dice_weight * dice
        combined = (per_side * valid).sum() / valid.sum().clamp_min(1.0)
        return {
            "combined": combined,
            "focal": (focal * valid).sum() / valid.sum().clamp_min(1.0),
            "dice": (dice * valid).sum() / valid.sum().clamp_min(1.0),
            "per_side": per_side,
            "target": target,
            "neighborhood": neighborhood,
            "valid": valid,
        }

    def _qrl_region_targets(self, logits, targets):
        """Adaptive interior/transition/context targets from transformed masks."""
        region_targets = []
        quality_weights = []
        for target in targets:
            if "masks" not in target:
                raise RuntimeError("QRL training requires transformed SAM masks")
            masks = target["masks"]
            if masks.shape[0] == 0 or not bool(masks.any()):
                regions = logits.new_zeros((3, *masks.shape[-2:]))
                quality = 0.0
            else:
                union = masks.float().to(logits.device).amax(dim=0, keepdim=True)
                height, width = union.shape[-2:]
                boxes = target["boxes"]
                short_side = min(
                    float((boxes[:, 2] * width).min().detach()),
                    float((boxes[:, 3] * height).min().detach()),
                )
                radius = int(round(self.qrl_region_radius_fraction * short_side))
                radius = max(
                    self.qrl_region_radius_min,
                    min(self.qrl_region_radius_max, radius),
                )
                near = F.max_pool2d(
                    union.unsqueeze(0),
                    kernel_size=2 * radius + 1,
                    stride=1,
                    padding=radius,
                ).squeeze(0)
                eroded = 1.0 - F.max_pool2d(
                    (1.0 - union).unsqueeze(0),
                    kernel_size=2 * radius + 1,
                    stride=1,
                    padding=radius,
                ).squeeze(0)
                outer_radius = 3 * radius
                outer = F.max_pool2d(
                    union.unsqueeze(0),
                    kernel_size=2 * outer_radius + 1,
                    stride=1,
                    padding=outer_radius,
                ).squeeze(0)
                interior = union
                transition = (near - eroded).clamp(0.0, 1.0)
                context = (outer - near).clamp(0.0, 1.0)
                regions = torch.cat((interior, transition, context), dim=0)
                if "sam_quality" not in target:
                    raise RuntimeError("QRL training target is missing sam_quality")
                quality = float(target["sam_quality"].reshape(-1)[0].detach())
            regions = F.interpolate(
                regions.unsqueeze(0), size=logits.shape[-2:], mode="area"
            ).squeeze(0)
            region_targets.append(regions.clamp(0.0, 1.0))
            quality_weights.append(quality)
        return torch.stack(region_targets), logits.new_tensor(quality_weights)

    def _qrl_region_loss(self, logits, target, quality):
        bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        positive_mass = target.sum((-2, -1)).clamp_min(1.0)
        negative_mass = (1.0 - target).sum((-2, -1)).clamp_min(1.0)
        bce_positive = (bce_map * target).sum((-2, -1)) / positive_mass
        bce_negative = (bce_map * (1.0 - target)).sum((-2, -1)) / negative_mass
        bce = 0.5 * (bce_positive + bce_negative)
        probability = logits.sigmoid()
        intersection = (probability * target).sum((-2, -1))
        denominator = probability.sum((-2, -1)) + target.sum((-2, -1))
        dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        per_sample = (bce + self.qrl_region_dice_weight * dice).mean(dim=1)
        return (per_sample * quality).sum() / quality.sum().clamp_min(1.0)

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {"loss_focal": loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_labels_mal(self, outputs, targets, indices, num_boxes, values=None):
        """Matchability-aware classification loss from DEIM.

        Positive targets use detached matched IoU raised by ``gamma``.  The
        method only changes training supervision and adds no inference path.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat(
                [target["boxes"][matched] for target, (_, matched) in zip(targets, indices)],
                dim=0,
            )
            ious, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat(
            [target["labels"][matched] for target, (_, matched) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = (target_score_o.unsqueeze(-1) * target).pow(self.gamma)
        pred_score = src_logits.sigmoid().detach()
        negative_weight = pred_score.pow(self.gamma)
        if self.mal_alpha is not None:
            negative_weight = float(self.mal_alpha) * negative_weight
        weight = negative_weight * (1 - target) + target

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_mal": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        """Compute Fine-Grained Localization (FGL) Loss
        and Decoupled Distillation Focal (DDF) Loss."""

        losses = {}
        if "pred_corners" in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.reg_max + 1))
            ref_points = outputs["ref_points"][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and "is_dn" in outputs:
                    self.fgl_targets_dn = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )
                if self.fgl_targets is None and "is_dn" not in outputs:
                    self.fgl_targets = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )

            target_corners, weight_right, weight_left = (
                self.fgl_targets_dn if "is_dn" in outputs else self.fgl_targets
            )

            ious = torch.diag(
                box_iou(
                    box_cxcywh_to_xyxy(outputs["pred_boxes"][idx]), box_cxcywh_to_xyxy(target_boxes)
                )[0]
            )
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
            if self.fgl_edge_weight_mode != "none":
                weight_targets = weight_targets * self._fgl_edge_weights(
                    target_boxes, weight_targets.dtype
                )

            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )

            if "teacher_corners" in outputs:
                pred_corners = outputs["pred_corners"].reshape(-1, (self.reg_max + 1))
                target_corners = outputs["teacher_corners"].reshape(-1, (self.reg_max + 1))
                if torch.equal(pred_corners, target_corners):
                    losses["loss_ddf"] = pred_corners.sum() * 0
                else:
                    weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(
                        weight_targets_local.dtype
                    )
                    weight_targets_local = (
                        weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
                    )

                    loss_match_local = (
                        weight_targets_local
                        * (T**2)
                        * (
                            nn.KLDivLoss(reduction="none")(
                                F.log_softmax(pred_corners / T, dim=1),
                                F.softmax(target_corners.detach() / T, dim=1),
                            )
                        ).sum(-1)
                    )
                    if self.ddf_edge_reliability_mode == "entropy":
                        edge_reliability = self._ddf_edge_reliability(
                            target_corners.reshape(-1, 4, self.reg_max + 1)
                        ).reshape(-1)
                        loss_match_local = loss_match_local * edge_reliability
                    if "is_dn" not in outputs:
                        batch_scale = (
                            8 / outputs["pred_boxes"].shape[0]
                        )  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (
                            (mask.sum() * batch_scale) ** 0.5,
                            ((~mask).sum() * batch_scale) ** 0.5,
                        )
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses["loss_ddf"] = (
                        loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg
                    ) / (self.num_pos + self.num_neg)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers."""
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "mal": self.loss_labels_mal,
            "local": self.loss_local,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        is_hbs_aux_pass = bool(kwargs.pop("_hbs_aux_pass", False))
        current_epoch = int(kwargs.get("epoch", 0))
        hbs_aux_outputs = outputs.get("hbs_aux_outputs")
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        matching_outputs = dict(outputs_without_aux)
        if "base_pred_logits" in outputs:
            # QCER/QDMF residuals are deliberately excluded from assignment.
            # Final detection losses still use the matched indices below.
            matching_outputs["pred_logits"] = outputs["base_pred_logits"]
        if (
            "base_pred_boxes" in outputs
            and outputs.get("qdmf_matcher_use_base_outputs", False)
        ):
            matching_outputs["pred_boxes"] = outputs["base_pred_boxes"]
        indices = self.matcher(matching_outputs, targets)["indices"]
        if (
            self.sqer_shape_weight > 0
            and self.training_epoch >= self.sqer_shape_start_epoch
            and not is_hbs_aux_pass
        ):
            shape_loss, shape_stats = sqer_shape_loss(
                outputs,
                targets,
                indices,
                supervision=self.sqer_supervision,
            )
            losses = {"loss_sqer_shape": self.sqer_shape_weight * shape_loss}
            outputs.update(shape_stats)
        else:
            losses = {}
        if "qdmf_gate" in outputs:
            gate = outputs["qdmf_gate"].detach().float().mean(dim=-1)
            scale_weight = outputs["qdmf_scale_weight"].detach().float()
            area = outputs["qdmf_area"].detach().float()
            matched = torch.zeros_like(gate, dtype=torch.bool)
            for batch_index, (source_indices, _) in enumerate(indices):
                matched[batch_index, source_indices.to(matched.device)] = True
            small = area < (32.0 * 32.0) / (512.0 * 640.0)
            medium = (~small) & (
                area < (96.0 * 96.0) / (512.0 * 640.0)
            )

            def masked_mean(value, mask):
                return (
                    value[mask].mean()
                    if mask.any()
                    else value.new_zeros(())
                )

            outputs["qdmf_gate_matched_mean"] = masked_mean(gate, matched)
            outputs["qdmf_gate_unmatched_mean"] = masked_mean(gate, ~matched)
            outputs["qdmf_gate_small_mean"] = masked_mean(gate, small)
            outputs["qdmf_gate_medium_mean"] = masked_mean(gate, medium)
            for name, mask in (("small", small), ("medium", medium)):
                for level, level_name in enumerate(("s8", "s16", "s32")):
                    outputs[f"qdmf_scale_{name}_{level_name}"] = masked_mean(
                        scale_weight[..., level], mask
                    )
        self._clear_cache()

        # Get the matching union set across all decoder layers.
        if "aux_outputs" in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            for i, aux_outputs in enumerate(outputs["aux_outputs"] + [outputs["pre_outputs"]]):
                indices_aux = self.matcher(aux_outputs, targets)["indices"]
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                indices_enc = self.matcher(aux_outputs, targets)["indices"]
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor(
                [num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device
            )
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert "aux_outputs" in outputs, ""

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        if self.stql_weight > 0 and not is_hbs_aux_pass:
            required = {"stql_query_embeddings", "stql_pixel_features"}
            missing = required.difference(outputs)
            if missing:
                raise RuntimeError(
                    f"STQL loss is enabled but outputs are missing {sorted(missing)}"
                )
            stql_raw, stql_stats = stql_shape_loss(
                outputs["stql_query_embeddings"],
                outputs["stql_pixel_features"],
                targets,
                indices,
                supervision=self.stql_supervision,
            )
            step = float(kwargs.get("step", 0))
            epoch_steps = max(float(kwargs.get("epoch_step", 1)), 1.0)
            progress = (float(current_epoch) + step / epoch_steps) / self.stql_total_epochs
            progress = max(0.0, min(progress, 1.0))
            if progress < 0.1:
                schedule = progress / 0.1
            elif progress < 0.5:
                schedule = 1.0
            elif progress < 0.7:
                schedule = (0.7 - progress) / 0.2
            else:
                schedule = 0.0
            losses["loss_stql"] = self.stql_weight * schedule * stql_raw
            outputs["stql_foreground_response"] = stql_stats["foreground_response"]
            outputs["stql_background_response"] = stql_stats["background_response"]
            outputs["stql_response_margin"] = stql_stats["response_margin"]
            outputs["stql_valid_instances"] = stql_stats["valid_instances"]
            outputs["stql_schedule"] = stql_raw.new_tensor(schedule)
        if self.qcer_objectness_weight > 0 and not is_hbs_aux_pass:
            required = {
                "qcer_objectness_logits",
                "qcer_valid_ir",
                "qcer_grid_centers",
                "qcer_level_ids",
            }
            missing = required.difference(outputs)
            if missing:
                raise RuntimeError(
                    "QCER objectness loss is enabled but outputs are missing "
                    f"{sorted(missing)}"
                )
            objectness_raw, objectness_stats = qcer_objectness_loss(
                outputs["qcer_objectness_logits"],
                outputs["qcer_valid_ir"],
                outputs["qcer_grid_centers"],
                outputs["qcer_level_ids"],
                targets,
            )
            losses["loss_qcer_objectness"] = (
                self.qcer_objectness_weight * objectness_raw
            )
            outputs["qcer_positive_tokens"] = objectness_stats["positive_tokens"]
            outputs["qcer_fallback_tokens"] = objectness_stats["fallback_tokens"]
            outputs["qcer_known_images"] = objectness_stats["known_images"]
        if 'sgc_group_loss' in outputs and not is_hbs_aux_pass:
            losses['loss_sgc_group'] = outputs['sgc_group_loss']
        if 'mote_ir_objectness_loss' in outputs and not is_hbs_aux_pass:
            losses['loss_mote_ir_objectness'] = outputs['mote_ir_objectness_loss']
        if 'sbra_relation_loss' in outputs and not is_hbs_aux_pass:
            losses['loss_sbra_relation'] = outputs['sbra_relation_loss']
        if 'mfam_region_loss' in outputs and not is_hbs_aux_pass:
            if isinstance(outputs['mfam_region_loss'], dict):
                for name, value in outputs['mfam_region_loss'].items():
                    losses['loss_mfam_' + name] = value
            else:
                losses['loss_mfam_region'] = outputs['mfam_region_loss']
        for loss in self.losses:
            indices_in = indices_go if loss in ["boxes", "local"] else indices
            num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
            meta = self.get_loss_meta_info(loss, outputs, targets, indices_in)
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        spar_schedule = self._spar_schedule(current_epoch)
        if self.spar_aux_weight > 0 and spar_schedule > 0:
            if "spar_fused_features" not in outputs:
                raise RuntimeError(
                    "spar_aux_weight is enabled but the model did not return "
                    "spar_fused_features"
                )
            losses["loss_spar"] = (
                self.spar_aux_weight
                * spar_schedule
                * self._spar_loss(outputs["spar_fused_features"], targets)
            )
        elif self.spar_aux_weight > 0:
            losses["loss_spar"] = outputs["pred_boxes"].sum() * 0.0

        sabr_schedule = self._sabr_schedule(current_epoch)
        if self.sabr_aux_weight > 0 and sabr_schedule > 0:
            required = {"sabr_boundary_logits", "sabr_body_logits"}
            missing = required.difference(outputs)
            if missing:
                raise RuntimeError(
                    "sabr_aux_weight is enabled but the model omitted "
                    + ", ".join(sorted(missing))
                )
            sabr = self._sabr_loss_components(
                outputs["sabr_boundary_logits"],
                outputs["sabr_body_logits"],
                targets,
            )
            losses["loss_sabr"] = self.sabr_aux_weight * sabr_schedule * sabr["combined"]
        elif self.sabr_aux_weight > 0:
            losses["loss_sabr"] = outputs["pred_boxes"].sum() * 0.0

        sbox_schedule = self._sbox_schedule(current_epoch)
        if self.sbox_aux_weight > 0 and sbox_schedule > 0:
            if "sbox_extreme_logits" not in outputs:
                raise RuntimeError(
                    "sbox_aux_weight is enabled but the model omitted "
                    "sbox_extreme_logits"
                )
            sbox = self._sbox_loss_components(
                outputs["sbox_extreme_logits"], targets
            )
            losses["loss_sbox"] = (
                self.sbox_aux_weight * sbox_schedule * sbox["combined"]
            )
        elif self.sbox_aux_weight > 0:
            losses["loss_sbox"] = outputs["pred_boxes"].sum() * 0.0

        if self.mdqa_aux_weight > 0 and not is_hbs_aux_pass:
            required = {"mdqa_query_embeddings", "mdqa_pixel_features"}
            missing = required.difference(outputs)
            if missing:
                raise RuntimeError(
                    "mdqa_aux_weight is enabled but the model omitted "
                    + ", ".join(sorted(missing))
                )
            losses["loss_mdqa"] = self.mdqa_aux_weight * self._mdqa_loss(
                outputs["mdqa_query_embeddings"],
                outputs["mdqa_pixel_features"],
                targets,
                indices,
            )

        if self.sqmi_aux_weight > 0 and not is_hbs_aux_pass:
            required = {"sqmi_query_embeddings", "sqmi_pixel_features"}
            missing = required.difference(outputs)
            if missing:
                raise RuntimeError(
                    "sqmi_aux_weight is enabled but the model omitted "
                    + ", ".join(sorted(missing))
                )
            # S-QMI1 intentionally uses the same focal+dice formulation as the
            # AUX control.  The only experimental variable is whether the
            # predicted mask is allowed to initialize detector reference boxes.
            losses["loss_sqmi"] = self.sqmi_aux_weight * self._mdqa_loss(
                outputs["sqmi_query_embeddings"],
                outputs["sqmi_pixel_features"],
                targets,
                indices,
            )

        if self.sbrd_aux_weight > 0:
            if "sbrd_stage_features" not in outputs:
                raise RuntimeError(
                    "sbrd_aux_weight is enabled but the model did not return "
                    "sbrd_stage_features"
                )
            losses["loss_sbrd"] = self.sbrd_aux_weight * self._sbrd_loss(
                outputs["sbrd_stage_features"], targets
            )

        if (
            self.qcsr_teacher_weight > 0 or self.qcsr_transfer_weight > 0
        ) and not is_hbs_aux_pass:
            required = {
                "qcsr_query_embeddings",
                "qcsr_teacher_pixels",
                "qcsr_student_pixels",
            }
            missing = required.difference(outputs)
            if missing:
                raise RuntimeError(
                    "QCSR is enabled but the model omitted "
                    + ", ".join(sorted(missing))
                )
            teacher_loss, transfer_loss = self._qcsr_losses(
                outputs["qcsr_query_embeddings"],
                outputs["qcsr_teacher_pixels"],
                outputs["qcsr_student_pixels"],
                targets,
                indices,
            )
            losses["loss_qcsr_teacher"] = self.qcsr_teacher_weight * teacher_loss
            losses["loss_qcsr_transfer"] = self.qcsr_transfer_weight * transfer_loss

        if self.srtod_reconstruction_weight > 0:
            if "srtod_reconstruction_loss" not in outputs:
                raise RuntimeError(
                    "srtod_reconstruction_weight is enabled but the model did not "
                    "return srtod_reconstruction_loss"
                )
            losses["loss_srtod_reconstruction"] = (
                self.srtod_reconstruction_weight
                * outputs["srtod_reconstruction_loss"]
            )

        if self.spatial_aux_weight > 0 and "spatial_importance_logits" in outputs:
            logits = outputs["spatial_importance_logits"]
            mask = torch.zeros_like(logits)
            height, width = logits.shape[-2:]
            for batch_idx, target in enumerate(targets):
                for cx, cy, bw, bh in target["boxes"]:
                    x1 = max(0, min(width - 1, int(torch.floor((cx - bw / 2) * width).item())))
                    y1 = max(0, min(height - 1, int(torch.floor((cy - bh / 2) * height).item())))
                    x2 = max(x1 + 1, min(width, int(torch.ceil((cx + bw / 2) * width).item())))
                    y2 = max(y1 + 1, min(height, int(torch.ceil((cy + bh / 2) * height).item())))
                    mask[batch_idx, 0, y1:y2, x1:x2] = 1.0

            focal_map = torchvision.ops.sigmoid_focal_loss(
                logits, mask, alpha=0.25, gamma=2.0, reduction="none"
            )
            positive = mask.sum((1, 2, 3)).clamp(min=1)
            negative = (1 - mask).sum((1, 2, 3)).clamp(min=1)
            focal_pos = (focal_map * mask).sum((1, 2, 3)) / positive
            focal_neg = (focal_map * (1 - mask)).sum((1, 2, 3)) / negative
            focal = (focal_pos + focal_neg).mean()

            probability = logits.sigmoid()
            intersection = (probability * mask).sum((1, 2, 3))
            denominator = probability.sum((1, 2, 3)) + mask.sum((1, 2, 3))
            dice = (1 - (2 * intersection + 1) / (denominator + 1)).mean()
            losses["loss_spatial_aux"] = self.spatial_aux_weight * (
                focal + self.spatial_aux_dice_weight * dice
            )

        if self.pdbr_boundary_aux_weight > 0:
            if "pdbr_boundary_logits" not in outputs:
                raise RuntimeError(
                    "pdbr_boundary_aux_weight is enabled but the model did not "
                    "return pdbr_boundary_logits"
                )
            logits_x, logits_y = outputs["pdbr_boundary_logits"]
            target_x, target_y = self._pdbr_boundary_targets(logits_x, logits_y, targets)
            boundary_loss = 0.5 * (
                self._pdbr_boundary_loss(logits_x, target_x)
                + self._pdbr_boundary_loss(logits_y, target_y)
            )
            losses["loss_pdbr_boundary_aux"] = (
                self.pdbr_boundary_aux_weight * boundary_loss
            )

        if self.tndp_detail_weight > 0:
            if "tndp_detail_prediction" not in outputs or "tndp_detail_target" not in outputs:
                raise RuntimeError(
                    "tndp_detail_weight is enabled but the model did not return "
                    "TNDP prediction and target"
                )
            prediction = outputs["tndp_detail_prediction"].float()
            detail_target = outputs["tndp_detail_target"].detach().float()
            if prediction.shape != detail_target.shape:
                raise RuntimeError(
                    "TNDP prediction/target shape mismatch: "
                    f"{tuple(prediction.shape)} vs {tuple(detail_target.shape)}"
                )
            gates = []
            for batch_index, target in enumerate(targets):
                if "masks" not in target:
                    raise RuntimeError(
                        "TNDP training requires transformed SAM masks in every target"
                    )
                masks = target["masks"]
                if masks.numel() == 0:
                    union = prediction.new_zeros((1, *masks.shape[-2:]))
                else:
                    union = masks.float().amax(dim=0, keepdim=True).to(prediction.device)
                gate = F.interpolate(
                    union.unsqueeze(0),
                    size=prediction.shape[-2:],
                    mode="area",
                ).squeeze(0)
                if self.tndp_neighborhood_radius > 0:
                    radius = self.tndp_neighborhood_radius
                    gate = F.max_pool2d(
                        gate.unsqueeze(0),
                        kernel_size=2 * radius + 1,
                        stride=1,
                        padding=radius,
                    ).squeeze(0)
                gates.append(gate.clamp_(0.0, 1.0))
            gate = torch.stack(gates, dim=0)
            regression = F.smooth_l1_loss(
                prediction, detail_target, reduction="none", beta=0.25
            )
            denominator = (gate.sum() * prediction.shape[1]).clamp_min(1.0)
            losses["loss_tndp_detail"] = self.tndp_detail_weight * (
                regression.mul(gate).sum() / denominator
            )

        if self.bpc_boundary_aux_weight > 0:
            if "bpc_boundary_logits" not in outputs:
                raise RuntimeError(
                    "bpc_boundary_aux_weight is enabled but the model did not return "
                    "bpc_boundary_logits"
                )
            logits = outputs["bpc_boundary_logits"].float()
            boundary_target, valid = self._bpc_boundary_targets(logits, targets)
            losses["loss_bpc_boundary"] = self.bpc_boundary_aux_weight * (
                self._bpc_boundary_loss(logits, boundary_target, valid)
            )

        qrl_schedule = self._qrl_region_schedule(current_epoch)
        if (
            self.qrl_region_aux_weight > 0
            and qrl_schedule > 0
            and not is_hbs_aux_pass
        ):
            if "qrl_region_logits" not in outputs:
                raise RuntimeError(
                    "qrl_region_aux_weight is enabled but the model did not return "
                    "qrl_region_logits"
                )
            logits = outputs["qrl_region_logits"].float()
            region_target, quality = self._qrl_region_targets(logits, targets)
            losses["loss_qrl_region"] = (
                self.qrl_region_aux_weight
                * qrl_schedule
                * self._qrl_region_loss(logits, region_target, quality)
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self.losses:
                    indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[i]
                    num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_in, num_boxes_in, **meta
                    )

                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer.
        if "pre_outputs" in outputs:
            aux_outputs = outputs["pre_outputs"]
            for loss in self.losses:
                indices_in = indices_go if loss in ["boxes", "local"] else cached_indices[-1]
                num_boxes_in = num_boxes_go if loss in ["boxes", "local"] else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {
                    k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                }
                l_dict = {k + "_pre": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if "enc_aux_outputs" in outputs:
            assert "enc_meta" in outputs, ""
            class_agnostic = outputs["enc_meta"]["class_agnostic"]
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t["labels"] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs["enc_aux_outputs"]):
                for loss in self.losses:
                    indices_in = indices_go if loss == "boxes" else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if loss == "boxes" else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(
                        loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses. For dfine
        if "dn_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]
            dn_num_boxes = dn_num_boxes if dn_num_boxes > 0 else 1

            for i, aux_outputs in enumerate(outputs["dn_outputs"]):
                aux_outputs["is_dn"] = True
                aux_outputs["up"], aux_outputs["reg_scale"] = outputs["up"], outputs["reg_scale"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer.
            if "dn_pre_outputs" in outputs:
                aux_outputs = outputs["dn_pre_outputs"]
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta
                    )
                    l_dict = {
                        k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict
                    }
                    l_dict = {k + "_dn_pre": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if self.hbs_aux_weight > 0 and not is_hbs_aux_pass:
            if hbs_aux_outputs is None:
                raise RuntimeError(
                    "hbs_aux_weight is enabled but the model did not return "
                    "hbs_aux_outputs; HBS must be active during training"
                )
            hbs_losses = self.forward(
                hbs_aux_outputs,
                targets,
                _hbs_aux_pass=True,
            )
            losses.update(
                {
                    f"{name}_hbs": self.hbs_aux_weight * value
                    for name, value in hbs_losses.items()
                }
            )

        if self.sdtec_presence_weight > 0:
            if "sdtec_presence_logits" not in outputs:
                raise RuntimeError(
                    "sdtec_presence_weight is enabled but the model omitted "
                    "sdtec_presence_logits"
                )
            if any("infrared_presence" not in target for target in targets):
                raise RuntimeError(
                    "SDTEC presence supervision requires infrared_presence in every target"
                )
            presence_target = torch.cat(
                [target["infrared_presence"] for target in targets]
            ).to(outputs["sdtec_presence_logits"])
            dropout_mask = outputs.get(
                "sdtec_dropout_mask",
                torch.zeros_like(presence_target, dtype=torch.bool),
            ).bool()
            # Real Anti-UAV thermal frames are positive in about 99% of the
            # training set, so excluding synthetic modality dropout would make
            # this head learn a nearly constant one.  Dropped samples are the
            # controlled negative examples for availability/reliability.
            availability_target = presence_target * (~dropout_mask).to(
                presence_target.dtype
            )
            presence_loss = F.binary_cross_entropy_with_logits(
                outputs["sdtec_presence_logits"], availability_target
            )
            losses["loss_sdtec_presence"] = (
                self.sdtec_presence_weight * presence_loss
            )

        if self.sdtec_diversity_weight > 0:
            if "sdtec_tokens" not in outputs:
                raise RuntimeError(
                    "sdtec_diversity_weight is enabled but the model omitted sdtec_tokens"
                )
            tokens = F.normalize(outputs["sdtec_tokens"], dim=-1)
            num_tokens = tokens.shape[1]
            if num_tokens > 1:
                similarity = torch.matmul(tokens, tokens.transpose(1, 2))
                identity = torch.eye(
                    num_tokens, device=similarity.device, dtype=similarity.dtype
                ).unsqueeze(0)
                diversity = ((similarity - identity) ** 2).sum()
                diversity = diversity / (
                    similarity.shape[0] * num_tokens * (num_tokens - 1)
                )
            else:
                diversity = tokens.sum() * 0.0
            losses["loss_sdtec_diversity"] = (
                self.sdtec_diversity_weight * diversity
            )

        if self.sdtec_fallback_weight > 0:
            required = {"sdtec_gate_by_layer", "sdtec_dropout_mask"}
            missing = required.difference(outputs)
            if missing:
                raise RuntimeError(
                    "sdtec_fallback_weight is enabled but the model omitted "
                    + ", ".join(sorted(missing))
                )
            dropout_mask = outputs["sdtec_dropout_mask"].bool()
            gates = outputs["sdtec_gate_by_layer"]
            if dropout_mask.any():
                fallback = gates[:, dropout_mask].square().mean()
            else:
                fallback = gates.sum() * 0.0
            losses["loss_sdtec_fallback"] = (
                self.sdtec_fallback_weight * fallback
            )

        if self.sdtec_pair_rank_weight > 0:
            required = {
                "sdtec_mismatch_logits",
                "sdtec_pair_valid_mask",
            }
            missing = required.difference(outputs)
            if missing:
                raise RuntimeError(
                    "M2 paired ranking is enabled but the model omitted "
                    + ", ".join(sorted(missing))
                )
            src_batch, src_query = self._get_src_permutation_idx(indices)
            target_classes = torch.cat(
                [target["labels"][target_index] for target, (_, target_index) in zip(targets, indices)]
            )
            logit_device = outputs["pred_logits"].device
            src_batch = src_batch.to(logit_device)
            src_query = src_query.to(logit_device)
            target_classes = target_classes.to(logit_device)
            pair_valid = outputs["sdtec_pair_valid_mask"].bool()[src_batch]
            if pair_valid.any():
                src_batch = src_batch[pair_valid]
                src_query = src_query[pair_valid]
                target_classes = target_classes[pair_valid]
                correct_positive = outputs["pred_logits"][
                    src_batch, src_query, target_classes
                ]
                mismatch_positive = outputs["sdtec_mismatch_logits"][
                    src_batch, src_query, target_classes
                ]
                correct_error = F.softplus(-correct_positive)
                mismatch_error = F.softplus(-mismatch_positive)
                pair_rank = F.relu(
                    correct_error
                    - mismatch_error
                    + self.sdtec_pair_rank_margin
                ).mean()
            else:
                pair_rank = outputs["pred_logits"].sum() * 0.0
            losses["loss_sdtec_pair_rank"] = (
                self.sdtec_pair_rank_weight * pair_rank
            )

        if self.sdtec_mismatch_suppression_weight > 0:
            required = {"sdtec_mismatch_delta", "sdtec_pair_valid_mask"}
            missing = required.difference(outputs)
            if missing:
                raise RuntimeError(
                    "M2 mismatch suppression is enabled but the model omitted "
                    + ", ".join(sorted(missing))
                )
            pair_valid = outputs["sdtec_pair_valid_mask"].bool()
            if pair_valid.any():
                mismatch_penalty = outputs["sdtec_mismatch_delta"][
                    pair_valid
                ].square().mean()
            else:
                mismatch_penalty = outputs["pred_logits"].sum() * 0.0
            losses["loss_sdtec_mismatch"] = (
                self.sdtec_mismatch_suppression_weight * mismatch_penalty
            )

        if self.sdtec_delta_l2_weight > 0:
            if "sdtec_logit_delta" not in outputs:
                raise RuntimeError(
                    "M2 delta regularization is enabled but sdtec_logit_delta is missing"
                )
            losses["loss_sdtec_delta_l2"] = (
                self.sdtec_delta_l2_weight
                * outputs["sdtec_logit_delta"].square().mean()
            )

        if self.ma1_alignment_weight > 0 or self.ma1_gate_weight > 0:
            required = {"ma1_aligned_centres", "ma1_gate_logits"}
            missing = required.difference(outputs)
            if missing:
                raise RuntimeError(
                    "M-B2 explicit alignment supervision is enabled but the model omitted "
                    + ", ".join(sorted(missing))
                )
            if any("infrared_boxes" not in target for target in targets):
                raise RuntimeError(
                    "M-B2 alignment supervision requires infrared_boxes in every target"
                )
            aligned_centres = outputs["ma1_aligned_centres"]
            gate_logits = outputs["ma1_gate_logits"]
            content_mask = outputs.get(
                "ma1_thermal_content_mask",
                torch.ones(aligned_centres.shape[0], device=aligned_centres.device),
            ).bool()
            selected_batch = []
            selected_query = []
            selected_gate_target = []
            positive_batch = []
            positive_query = []
            positive_centres = []
            for batch_index, (target, (source_index, target_index)) in enumerate(
                zip(targets, indices)
            ):
                # Anti-UAV is a single-target task.  Restrict direct geometric
                # supervision to unambiguous 1-RGB/0-or-1-IR samples instead of
                # inventing cross-modal identities for rare multi-box frames.
                if len(target["boxes"]) != 1 or len(source_index) == 0:
                    continue
                match = torch.where(target_index == 0)[0]
                if len(match) != 1:
                    continue
                query_index = int(source_index[match[0]])
                infrared_boxes = target["infrared_boxes"]
                has_unambiguous_ir = len(infrared_boxes) == 1
                pair_is_valid = bool(has_unambiguous_ir and content_mask[batch_index])
                selected_batch.append(batch_index)
                selected_query.append(query_index)
                selected_gate_target.append(float(pair_is_valid))
                if pair_is_valid:
                    positive_batch.append(batch_index)
                    positive_query.append(query_index)
                    positive_centres.append(
                        torch.as_tensor(infrared_boxes[0, :2]).to(aligned_centres)
                    )
            if positive_batch:
                predicted = aligned_centres[
                    torch.as_tensor(positive_batch, device=aligned_centres.device),
                    torch.as_tensor(positive_query, device=aligned_centres.device),
                ]
                target_centres = torch.stack(positive_centres)
                alignment_loss = F.smooth_l1_loss(
                    predicted, target_centres, beta=0.02
                )
            else:
                alignment_loss = aligned_centres.sum() * 0.0
            losses["loss_ma1_alignment"] = (
                self.ma1_alignment_weight * alignment_loss
            )

            normal_gate_logits = []
            normal_gate_targets = []
            if selected_batch:
                normal_gate_logits.append(
                    gate_logits[
                        torch.as_tensor(selected_batch, device=gate_logits.device),
                        torch.as_tensor(selected_query, device=gate_logits.device),
                    ]
                )
                normal_gate_targets.append(
                    gate_logits.new_tensor(selected_gate_target)
                )
            if "ma1_mismatch_gate_logits" in outputs and positive_batch:
                mismatch_valid = outputs["ma1_pair_valid_mask"].bool()
                mismatch_batch = [
                    batch_index
                    for batch_index in positive_batch
                    if bool(mismatch_valid[batch_index])
                ]
                mismatch_query = [
                    query_index
                    for batch_index, query_index in zip(
                        positive_batch, positive_query
                    )
                    if bool(mismatch_valid[batch_index])
                ]
                if mismatch_batch:
                    normal_gate_logits.append(
                        outputs["ma1_mismatch_gate_logits"][
                            torch.as_tensor(mismatch_batch, device=gate_logits.device),
                            torch.as_tensor(mismatch_query, device=gate_logits.device),
                        ]
                    )
                    normal_gate_targets.append(
                        gate_logits.new_zeros(len(mismatch_batch))
                    )
            if normal_gate_logits:
                gate_loss = F.binary_cross_entropy_with_logits(
                    torch.cat(normal_gate_logits), torch.cat(normal_gate_targets)
                )
            else:
                gate_loss = gate_logits.sum() * 0.0
            losses["loss_ma1_gate"] = self.ma1_gate_weight * gate_loss

        if self.ma1_mismatch_weight > 0:
            required = {"ma1_mismatch_delta", "ma1_pair_valid_mask"}
            missing = required.difference(outputs)
            if missing:
                raise RuntimeError(
                    "M-B2 mismatch suppression is enabled but the model omitted "
                    + ", ".join(sorted(missing))
                )
            valid = outputs["ma1_pair_valid_mask"].bool()
            if valid.any():
                # AMP residuals are initially around 1e-5; squaring in fp16
                # would underflow to an artificial exact zero.
                mismatch_loss = (
                    outputs["ma1_mismatch_delta"][valid].float().square().mean()
                )
            else:
                mismatch_loss = outputs["pred_logits"].sum() * 0.0
            losses["loss_ma1_mismatch"] = (
                self.ma1_mismatch_weight * mismatch_loss
            )

        # For debugging Objects365 pre-train.
        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs["pred_boxes"][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == "iou":
            iou, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
            )
            iou = torch.diag(iou)
        elif self.boxes_weight_format == "giou":
            iou = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
                )
            )
        else:
            raise AttributeError()

        if loss in ("boxes",):
            meta = {"boxes_weight": iou}
        elif loss in ("vfl", "mal"):
            meta = {"values": iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices"""
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    )
                )

        return dn_match_indices

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)

    def unimodal_distribution_focal_loss(
        self, pred, label, weight_right, weight_left, weight=None, reduction="sum", avg_factor=None
    ):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(
            -1
        ) + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs["aux_outputs"]) + 1 if "aux_outputs" in outputs else 1
        step = 0.5 / (num_layers - 1)
        opt_list = [0.5 + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
