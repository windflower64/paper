"""Position-preserving residual carrier with separately supervised relations."""
import torch
from torch import nn
from torch.nn import functional as F


class SAMBoundaryRelation(nn.Module):
    def __init__(self, channels=256, width=64):
        super().__init__()
        self.width=int(width)
        self.project=nn.Conv2d(channels,width,1,bias=False)
        self.relation=nn.Sequential(nn.Conv2d(channels,width,1),nn.ReLU(),
                                    nn.AvgPool2d(2),nn.Conv2d(width,16,1))
        self.output=nn.Conv2d(4*width,channels,1,bias=False)
        nn.init.zeros_(self.output.weight)
        self._aux_input=None

    def forward(self,x,standard):
        if x.shape[-2]%2 or x.shape[-1]%2:
            raise ValueError('S-BRA1 requires even S8 spatial dimensions')
        b,_,h,w=standard.shape
        if x.shape[-2:]!=(h*2,w*2):raise ValueError('Expected S8/S16 size ratio 2')
        self._aux_input=x.detach() if self.training else None
        logits=self.relation(x).permute(0,2,3,1).reshape(b,h,w,4,4)
        with torch.autocast(device_type=x.device.type,enabled=False):
            affinity=logits.float().softmax(-1)
        z=F.pixel_unshuffle(self.project(x),2).reshape(b,self.width,4,h,w).permute(0,3,4,2,1)
        mixed=.5*z+.5*torch.matmul(affinity.to(z.dtype),z)
        carrier=mixed.permute(0,4,3,1,2).reshape(b,4*self.width,h,w)
        return standard+self.output(carrier)

    def relation_loss(self,target,valid):
        if self._aux_input is None:raise RuntimeError('Training forward required before supervision')
        b,h,w=valid.shape
        logits=self.relation(self._aux_input).permute(0,2,3,1).reshape(b,h,w,4,4).float()
        target=target.detach().float()
        target=target/target.sum(-1,keepdim=True).clamp_min(1e-8)
        error=-(target*logits.log_softmax(-1)).sum(-1).mean(-1)
        return (error*valid).sum()/valid.sum().clamp_min(1)

    def supervision(self,targets,source):
        from ...zoo.dfine.sam_group_contrast import raster
        if source not in ('sam','box'):raise ValueError(source)
        if self._aux_input is None:raise RuntimeError('Missing training features')
        h,w=self._aux_input.shape[-2:]
        relations=[]; supports=[]
        for t in targets:
            masks=t['masks'].float()
            mh,mw=masks.shape[-2:]
            boxes=t['boxes']
            valid=len(boxes)==1 and float(t['sam_quality'])>0 and masks.numel()>0
            field=masks.amax(0) if valid and source=='sam' else raster(boxes,mh,mw)
            p=F.interpolate(field[None,None],size=(h,w),mode='area')
            p=F.pixel_unshuffle(p,2)[0].permute(1,2,0)
            relations.append(p[...,None]*p[...,None,:]+(1-p[...,None])*(1-p[...,None,:]))
            # Identical box-defined support in all arms, independent of mask shape.
            local=raster(boxes,mh,mw,2.)
            support=F.interpolate(local[None,None],size=(h//2,w//2),mode='area')[0,0]>.1
            supports.append(support & valid)
        return self.relation_loss(torch.stack(relations),torch.stack(supports))
