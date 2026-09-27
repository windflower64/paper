"""Experimental native-decoder S4 readout. No production model changes."""
import torch
from torch import nn
from torch.nn import functional as F
from src.zoo.dfine.sam_group_contrast import raster, choose


class QueryReadout(nn.Module):
    def __init__(self, source_channels, query_channels, width=32):
        super().__init__()
        self.project=nn.Conv2d(source_channels,width,1,bias=False)
        self.offsets=nn.Linear(query_channels,24)
        self.weights=nn.Linear(query_channels,12)
        self.output=nn.Linear(width,query_channels,bias=False)
        nn.init.zeros_(self.offsets.weight)
        # Four points each for interior, transition, and local context.
        directions=torch.tensor([[-1.,0.],[0.,-1.],[1.,0.],[0.,1.]])
        initial=torch.cat([directions*r for r in (.15,.5,.85)])
        with torch.no_grad():self.offsets.bias.copy_(torch.atanh(initial).flatten())
        nn.init.zeros_(self.weights.weight);nn.init.zeros_(self.weights.bias)
        self.locations=None

    def forward(self, feature, query, references):
        boxes=references[:,:,0,:].detach()
        self.references=boxes
        offsets=self.offsets(query).reshape(*query.shape[:2],12,2).tanh()
        self.locations=boxes[:,:,None,:2]+offsets*boxes[:,:,None,2:]
        projected=self.project(feature)
        sampled=F.grid_sample(projected,self.locations*2-1,mode='bilinear',
                              padding_mode='zeros',align_corners=False)
        weights=self.weights(query).softmax(-1)
        tokens=(sampled.permute(0,2,3,1)*weights[...,None]).sum(2)
        return self.output(tokens)


def role_points(target, source):
    if source not in ('sam','box'):raise ValueError(source)
    if len(target['boxes'])!=1 or float(target['sam_quality'])<=0:return None
    mask=target['masks'].float().amax(0)
    h,w=mask.shape
    if source=='box':mask=raster(target['boxes'],h,w)
    # One native-image pixel transition, not an S8/S16 occupancy target.
    dilated=F.max_pool2d(mask[None,None],3,1,1)[0,0]
    eroded=-F.max_pool2d(-mask[None,None],3,1,1)[0,0]
    neighborhood=raster(target['boxes'],h,w,expansion=2.)
    fields=[eroded>.5,(dilated-eroded)>.5,(dilated<.5)&(neighborhood>.5)]
    result=[]
    for field in fields:
        indices=field.flatten().nonzero().flatten()
        if len(indices)<4:return None
        # Deterministic, bounded target set; no model-dependent selection.
        indices=choose(indices,64)
        result.append(torch.stack(((indices%w+.5)/w,(indices//w+.5)/h),-1))
    return result


def geometry_loss(locations,targets,indices,source):
    loss=locations.float().sum()*0.;count=0
    for batch,(queries,objects) in enumerate(indices):
        target=targets[batch]
        both={arm:role_points(target,arm) for arm in ('sam','box')}
        if any(v is None for v in both.values()):continue
        for query,obj in zip(queries,objects):
            if int(obj)!=0:continue
            norm=target['boxes'][0,2:].float().clamp_min(1e-4)
            points=locations[batch,query].float().reshape(3,4,2)
            for role,teacher in zip(points,both[source]):
                distances=torch.cdist(role/norm,teacher.float()/norm).square()
                loss=loss+.5*(distances.min(0).values.mean()+distances.min(1).values.mean())/3
            count+=1
    return loss/max(count,1),count


class CrossAttentionReadout(nn.Module):
    """Add high-resolution values before the native gateway/FFN and box heads."""
    def __init__(self,original,reader,context):
        super().__init__();self.original=original;self.reader=reader;self.context=context
        self.gain=.1  # Engineering probe only; not a selected training hyperparameter.
        self.intervention='normal'

    def forward(self,query,references,value,spatial_shapes):
        standard=self.original(query,references,value,spatial_shapes)
        if self.gain==0:return standard
        feature=self.context['s4']
        if self.intervention=='zero':feature=torch.zeros_like(feature)
        elif self.intervention=='shift':feature=torch.roll(feature,feature.shape[-1]//2,-1)
        elif self.intervention!='normal':raise ValueError(self.intervention)
        return standard+self.gain*self.reader(feature,query,references)
