"""S-MFAM2: box support and conditional SAM shape, a project hypothesis.

Three soft partitions sum to one: object body, within-support context,
outside-support background. No SAM/GT is consumed by the inference path.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .sam_mask_aggregation import SAMMaskAggregation, conv, region_loss


class SupportShapeAggregation(SAMMaskAggregation):
    def __init__(self,low_channels=256,high_channels=512,width=32):
        super().__init__(low_channels,high_channels,width)
        self.mask_head=nn.Sequential(conv(2*width,width),nn.Conv2d(width,2,1))
        self.low_split=conv(3*width,width)
        self.high_split=conv(3*width,width)

    def partition(self,logits):
        support,shape=logits.sigmoid().split(1,dim=1)
        if self.intervention=='support_constant':support=torch.full_like(support,.5)
        elif self.intervention=='shape_constant':shape=torch.full_like(shape,.5)
        elif self.intervention=='constant':
            support=torch.full_like(support,.5);shape=torch.full_like(shape,.5)
        elif self.intervention!='learned':raise ValueError(self.intervention)
        return support*shape,support*(1-shape),1-support

    def forward(self,low,high):
        if self.intervention=='disabled':return high,None
        shallow=self.low_proj(low)
        deep=F.interpolate(self.high_proj(high),size=low.shape[-2:],mode='bilinear',align_corners=False)
        logits=self.mask_head(torch.cat([shallow,deep],1))
        regions=self.partition(logits)
        low_content=self.low_split(torch.cat([weight*shallow for weight in regions],1))
        high_content=self.high_split(torch.cat([weight*deep for weight in regions],1))
        # Keep MFAM1's unconditional residual fixed for an interpretable test.
        fused=self.aggregate(torch.cat([high_content,low_content],1))+deep+shallow
        delta=self.output_proj(F.adaptive_avg_pool2d(fused,high.shape[-2:]))
        return high+self.residual_scale*delta,logits


def support_shape_loss(logits,targets,source):
    if source not in ('sam','box'):raise ValueError(source)
    logits=logits.float()
    support=region_loss(logits[:,:1],targets,'box')
    total=logits[:,1:].sum()*0
    weight_sum=logits.new_zeros(())
    for prediction,target in zip(logits[:,1:],targets):
        if len(target['boxes'])==0:continue
        quality=float(target['sam_quality'].item())
        if quality<=0:continue
        masks=target['masks'].to(prediction)
        height,width=masks.shape[-2:]
        boxmap=prediction.new_zeros(height,width)
        for cx,cy,bw,bh in target['boxes'].detach().tolist():
            x0=max(0,int((cx-bw/2)*width));x1=min(width,math.ceil((cx+bw/2)*width))
            y0=max(0,int((cy-bh/2)*height));y1=min(height,math.ceil((cy+bh/2)*height))
            boxmap[y0:y1,x0:x1]=1
        region=F.interpolate(boxmap[None,None],size=prediction.shape[-2:],mode='area')[0]
        body=boxmap if source=='box' else masks.amax(0)*boxmap
        body=F.interpolate(body[None,None],size=prediction.shape[-2:],mode='area')[0]
        # Conditional occupancy, not unconditional shape divided a second time.
        truth=(body/region.clamp_min(1e-8)).clamp(0,1)
        bce=F.binary_cross_entropy_with_logits(prediction,truth,reduction='none')
        bce=(bce*region).sum()/region.sum().clamp_min(1e-8)
        prob=prediction.sigmoid()
        dice=1-(2*(prob*body).sum()+1)/((prob*region).sum()+body.sum()+1)
        total=total+quality*(bce+dice)
        weight_sum=weight_sum+quality
    # Fixed total loss budget shared by SAM and box-control models.
    return {'support':.5*support,'shape':.5*total/weight_sum.clamp_min(1)}
