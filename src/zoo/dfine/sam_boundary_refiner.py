"""SBR1: spatial, four-sided box refinement; teacher masks never enter forward."""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import box_convert, generalized_box_iou
from .sam_query_evidence_reader import _QueryRegionBlock


class SAMBoundaryRefiner(nn.Module):
    def __init__(self, dim=64, query_dim=128, topk=64, roi_size=32, radius=0.45):
        super().__init__()
        if not 0 < radius < 0.5:
            raise ValueError('Independent side residuals require radius < 0.5')
        self.dim, self.topk, self.roi_size, self.radius = dim, topk, roi_size, radius
        self.position = nn.Linear(4, dim)
        self.query_projection = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, dim))
        self.roles = nn.Parameter(torch.empty(4, dim))
        nn.init.normal_(self.roles, std=0.02)
        self.blocks = nn.ModuleList([_QueryRegionBlock(dim, heads=4) for _ in range(2)])
        self.head = nn.Sequential(nn.LayerNorm(2*dim+8), nn.Linear(2*dim+8,128), nn.GELU(),
                                  nn.Linear(128,64), nn.GELU(), nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, memory_map, queries, base_logits, boxes, image_hw):
        b, total, _ = boxes.shape
        k, r = min(self.topk,total), self.roi_size
        indices = base_logits.detach().float().sigmoid().amax(-1).topk(k,dim=1).indices
        chosen = boxes.detach().float().gather(1,indices[...,None].expand(-1,-1,4))
        cx,cy,w,h = chosen.unbind(-1)
        rw = (2*w).clamp_min(32/image_hw[1]); rh = (2*h).clamp_min(32/image_hw[0])
        lo = torch.stack(((cx-rw/2).clamp(0,1),(cy-rh/2).clamp(0,1)),dim=-1)
        hi = torch.stack(((cx+rw/2).clamp(0,1),(cy+rh/2).clamp(0,1)),dim=-1)
        valid = (w>1e-6)&(h>1e-6)&((hi-lo)>1e-6).all(-1)
        axis = (torch.arange(r,device=boxes.device,dtype=torch.float32)+0.5)/r
        yy,xx = torch.meshgrid(axis,axis,indexing='ij')
        unit = torch.stack((xx,yy),-1)
        grid = 2*(lo[:,:,None,None,:]+(hi-lo)[:,:,None,None,:]*unit)-1
        sampled = F.grid_sample(memory_map,grid.reshape(b,k*r,r,2).to(memory_map.dtype),align_corners=False)
        visual = sampled.reshape(b,self.dim,k,r*r).permute(0,2,3,1).reshape(b*k,r*r,self.dim)
        uv = (2*unit-1).reshape(r*r,2)
        pos = torch.cat((uv,uv.square()),-1)
        memory = visual + self.position(pos)[None]
        selected_q = queries.detach().gather(1,indices[...,None].expand(-1,-1,queries.shape[-1]))
        q = self.query_projection(selected_q).reshape(b*k,1,self.dim)+self.roles[None]
        for i,block in enumerate(self.blocks):
            q,weights = block(q,memory,return_attention=i==len(self.blocks)-1)
        attention = weights.float().mean(1)
        pooled = attention @ visual.float()
        # Spatial moments must remain FP32 even under autocast.
        with torch.autocast(device_type=boxes.device.type, enabled=False):
            mean = attention @ uv
            variance = (attention @ uv.square()-mean.square()).clamp_min(0)
            original_xy = box_convert(chosen,'cxcywh','xyxy').reshape(b*k,4)
            extent = (hi-lo).reshape(b*k,2).clamp_min(1e-6)
            local_box = 2*(original_xy-lo.reshape(b*k,2).repeat(1,2))/extent.repeat(1,2)-1
        feature = torch.cat((pooled,q.float(),mean,variance,local_box[:,None].expand(-1,4,-1)),-1)
        raw = self.head(feature).squeeze(-1).reshape(b,k,4).float()
        normalized_delta = raw.tanh()*valid[...,None]
        scale = torch.stack((w,h,w,h),-1)*self.radius
        selected_refined = box_convert(box_convert(chosen,'cxcywh','xyxy')+scale*normalized_delta,'xyxy','cxcywh')
        # Preserve bitwise identity at initialization, including xyxy conversion rounding.
        residual = selected_refined-chosen
        zero = normalized_delta.eq(0).all(-1,keepdim=True)
        residual = torch.where(zero,residual-residual.detach(),residual)
        refined = boxes.float()+torch.zeros_like(boxes.float()).scatter(1,indices[...,None].expand(-1,-1,4),residual)
        return {'boxes':refined,'indices':indices,'attention':attention.reshape(b,k,4,r*r),
                'grid':grid,'raw':raw,'normalized_delta':normalized_delta,'valid':valid,'scale':scale}


def boundary_teacher(mask,grid,ignore=None):
    """Direction-weighted SAM contour, not a target box or a forward input."""
    r = grid.shape[-2]
    occ = F.grid_sample(mask.float()[None,None],grid.float()[None],align_corners=False)
    edge = (F.max_pool2d(occ,3,1,1)+F.max_pool2d(-occ,3,1,1)).clamp_min(0)[0,0]
    if ignore is not None:
        edge = edge*(~ignore)
    mass = edge.sum()
    border = edge[0].sum()+edge[-1].sum()+edge[1:-1,0].sum()+edge[1:-1,-1].sum()
    valid = bool(mass>=1) and bool(border/mass.clamp_min(1e-8)<=0.05)
    axis = 2*(torch.arange(r,device=edge.device,dtype=torch.float32)+0.5)/r-1
    v,u = torch.meshgrid(axis,axis,indexing='ij')
    directional = torch.stack((-u,-v,u,v),0).mul(4).exp()
    teacher = edge[None]*directional+1e-4
    if ignore is not None: teacher = teacher*(~ignore)[None]
    teacher = teacher.flatten(1)
    teacher = teacher/teacher.sum(-1,keepdim=True).clamp_min(1e-8)
    return teacher,valid


def refinement_loss(result,base_boxes,targets,matches,shape_weight=0.05):
    device = base_boxes.device
    positive = torch.zeros_like(result['indices'],dtype=torch.bool)
    pred,gt,base,teacher_losses = [],[],[],[]
    required_outside = 0; teacher_attempts=0; valid_teacher=0; selected_matched=0; matched=0
    for bi,(src,dst) in enumerate(matches):
        target = targets[bi]
        for qi,ti in zip(src.tolist(),dst.tolist()):
            matched+=1
            found = (result['indices'][bi]==qi).nonzero().flatten()
            if not len(found): continue
            j=int(found[0]); positive[bi,j]=True; selected_matched+=1
            pred.append(result['boxes'][bi,qi]); gt.append(target['boxes'][ti].float())
            base.append(base_boxes[bi,qi].detach().float())
            needed = (box_convert(gt[-1][None],'cxcywh','xyxy')[0]-box_convert(base[-1][None],'cxcywh','xyxy')[0]).abs()
            required_outside+=int((needed>result['scale'][bi,j]).any())
            masks=target.get('masks'); quality=target.get('sam_quality')
            validity=target.get('sam_instance_valid')
            if masks is None or len(masks)!=len(target['boxes']) or quality is None: continue
            if float(torch.as_tensor(quality).flatten()[0])<=0: continue
            if validity is not None and not bool(validity[ti]): continue
            teacher_attempts+=1
            # Single-instance pilot; reject rather than mislabel unknown other instances.
            if len(target['boxes'])!=1: raise ValueError('SBR1 pilot teacher currently requires one instance')
            teacher,ok=boundary_teacher(masks[ti],result['grid'][bi,j])
            if not ok: continue
            valid_teacher+=1
            att=result['attention'][bi,j].float()
            teacher_losses.append((teacher*(teacher.clamp_min(1e-8).log()-att.clamp_min(1e-8).log())).sum(-1).mean())
    zero=result['raw'].sum()*0
    if pred:
        pp,gg,bb=torch.stack(pred),torch.stack(gt),torch.stack(base)
        l1=F.l1_loss(pp,gg,reduction='none').sum(-1).mean()
        giou=(1-torch.diag(generalized_box_iou(box_convert(pp,'cxcywh','xyxy'),box_convert(gg,'cxcywh','xyxy')))).mean()
    else: l1=giou=zero
    shape=torch.stack(teacher_losses).mean() if teacher_losses else zero
    negative=result['normalized_delta'][~positive]
    stable=negative.abs().mean() if negative.numel() else zero
    loss=5*l1+2*giou+shape_weight*shape+0.1*stable
    return loss,{'l1':float(l1.detach()),'giou':float(giou.detach()),'shape_kl':float(shape.detach()),
                 'negative_stability':float(stable.detach()),'matched':matched,'selected_matched':selected_matched,
                 'teacher_attempts':teacher_attempts,'teacher_valid':valid_teacher,'outside_radius':required_outside,
                 'saturation_fraction':float((result['normalized_delta'].abs()>0.95).float().mean())}
