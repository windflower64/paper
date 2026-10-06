"""Add IR P32 evidence at IR-selected candidates; no RGB/IR alignment."""
import torch
from torch import nn
import torch.nn.functional as F
from .target_evidence_thermal_fusion import TargetEvidenceThermalFusion


class ThermalP32Fusion(TargetEvidenceThermalFusion):
    def __init__(self, original, channels):
        super().__init__(visible_channels=original.visible_channels,
            thermal_s8_channels=original.object_token_s8.in_channels,
            thermal_s16_channels=original.object_token_s16.in_channels,
            embed_dim=original.object_token_s8.out_channels,
            num_candidates=original.num_candidates,max_rms_ratio=original.max_rms_ratio,
            final_residual_scale=original.final_residual_scale,
            objectness_loss_weight=original.objectness_loss_weight,layout_fix=original.layout_fix)
        self.load_state_dict(original.state_dict(),strict=True)
        self.object_token_p32=nn.Conv2d(channels,original.object_token_s8.out_channels,1,bias=False)
        nn.init.zeros_(self.object_token_p32.weight)
        self.p32=None

    def _extra_scale_tokens(self, probability, x8, y8):
        if self.p32 is None: raise RuntimeError('IR P32 not captured')
        feature=self.object_token_p32(self.p32)
        h,w=feature.shape[-2:]; h8,w8=probability.shape[-2:]
        x=((x8.float()+.5)*w/w8).long().clamp(0,w-1)
        y=((y8.float()+.5)*h/h8).long().clamp(0,h-1)
        flat=feature.flatten(2).transpose(1,2)
        objects=flat.gather(1,(y*w+x)[...,None].expand(-1,-1,flat.shape[-1]))
        weights=(1-F.interpolate(probability,size=(h,w),mode='bilinear',align_corners=False)).flatten(2)
        background=torch.bmm(weights,flat)/weights.sum(-1,keepdim=True).clamp_min(1.)
        return objects,background
