"""SBR3: inside/outside evidence at 17 alternative positions per box side.

No teacher enters forward. The predicted occupancy participates directly in
position scoring; masks and synthetic GT boxes are used only by loss().
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import box_convert,generalized_box_iou


class SAMSideSearch(nn.Module):
    def __init__(self,reader,bins=17,strip_points=16,radius=.45):
        super().__init__()
        assert bins%2==1 and 0<radius<.5
        self.bins,self.strip_points,self.radius,self.topk=bins,strip_points,radius,64
        self.s4_projection=copy.deepcopy(reader.s4_projection).requires_grad_(True)
        self.s8_projection=copy.deepcopy(reader.s8_projection).requires_grad_(True)
        self.fusion=copy.deepcopy(reader.fusion).requires_grad_(True)
        self.shape_head=nn.Sequential(nn.Conv2d(64,64,3,padding=1,bias=False),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,1,1))
        self.score=nn.Sequential(nn.LayerNorm(200),nn.Linear(200,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.score[-1].weight); nn.init.zeros_(self.score[-1].bias)
        self.register_buffer('offsets',torch.linspace(-radius,radius,bins))

    def encode(self,s4,s8):
        fine=self.s4_projection(s4.detach()); coarse=self.s8_projection(s8.detach())
        return self.fusion(torch.cat((fine,F.interpolate(coarse,size=fine.shape[-2:],mode='bilinear',align_corners=False)),1))

    def search(self,memory,shape_logits,boxes,hw):
        boxes=boxes.detach().float(); b,k,_=boxes.shape
        xy=box_convert(boxes,'cxcywh','xyxy'); w,h=boxes[...,2],boxes[...,3]
        valid=(boxes[...,2:]>1e-6).all(-1)
        axis=(torch.arange(self.strip_points,device=boxes.device).float()+.5)/self.strip_points
        strips=[]
        for side in range(4):
            horizontal=side in (1,3)
            scale=h if horizontal else w
            pos=xy[...,side,None]+scale[...,None]*self.offsets
            tangent=(xy[...,0,None]+w[...,None]*axis) if horizontal else (xy[...,1,None]+h[...,None]*axis)
            eps=torch.minimum((scale*.04).clamp_min(1/hw[0 if horizontal else 1]),scale*.1)
            inward=1. if side in (0,1) else -1.
            coordinates=[]
            for probe in (inward,0.,-inward):
                normal=(pos+probe*eps[...,None])[...,None].expand(-1,-1,-1,self.strip_points)
                tang=tangent[:,:,None,:].expand(-1,-1,self.bins,-1)
                coordinates.append(torch.stack((tang,normal) if horizontal else (normal,tang),-1))
            strips.append(torch.stack(coordinates,dim=3))
        grid=torch.stack(strips,dim=2) # B,K,4,N,3,P,2
        fields=torch.cat((memory,shape_logits.sigmoid()),1)
        sampled=F.grid_sample(fields,(2*grid-1).reshape(b,k*4*self.bins*3,self.strip_points,2),align_corners=False)
        sampled=sampled.reshape(b,65,k,4,self.bins,3,self.strip_points).mean(-1).permute(0,2,3,4,5,1)
        inside,center,outside=sampled.unbind(-2)
        side_code=torch.eye(4,device=boxes.device).reshape(1,1,4,1,4).expand(b,k,-1,self.bins,-1)
        offset=self.offsets.reshape(1,1,1,self.bins,1).expand(b,k,4,-1,-1)
        feature=torch.cat((center[...,:64],inside[...,:64],inside[...,:64]-outside[...,:64],
                           inside[...,64:],center[...,64:],outside[...,64:],offset,side_code),-1)
        scores=self.score_candidates(feature, grid)
        probability=scores.softmax(-1)
        delta=((probability-1/self.bins)*self.offsets).sum(-1)*valid[...,None]
        scale=torch.stack((w,h,w,h),-1)
        changed=box_convert(xy+delta*scale,'xyxy','cxcywh')
        residual=changed-boxes
        zero=delta.eq(0).all(-1,keepdim=True)
        residual=torch.where(zero,residual-residual.detach(),residual)
        return {'boxes':boxes+residual,'scores':scores,'delta':delta,'scale':scale,'valid':valid}

    def score_candidates(self, feature, grid):
        return self.score(feature).squeeze(-1)

    def forward(self,s4,s8,base_logits,boxes,image_hw):
        memory=self.encode(s4,s8); shape_logits=self.shape_head(memory)
        indices=base_logits.detach().sigmoid().amax(-1).topk(min(self.topk,boxes.shape[1]),dim=1).indices
        chosen=boxes.gather(1,indices[...,None].expand(-1,-1,4)).detach()
        selected=self.search(memory,shape_logits,chosen,image_hw)
        refined=boxes.float()+torch.zeros_like(boxes.float()).scatter(1,indices[...,None].expand(-1,-1,4),selected['boxes']-chosen)
        result={'boxes':refined,'indices':indices,'selected':selected,'original':chosen,'hw':tuple(image_hw)}
        if self.training: result.update(memory=memory,shape_logits=shape_logits)
        return result

    def position_loss(self,scores,original,gt):
        needed=(box_convert(gt,'cxcywh','xyxy')-box_convert(original,'cxcywh','xyxy'))/original[:,[2,3,2,3]].clamp_min(1e-6)
        valid=(needed.abs()<=self.radius)
        coordinate=((needed/self.radius+1)*.5*(self.bins-1)).clamp(0,self.bins-1)
        lower=coordinate.floor().long(); upper=(lower+1).clamp_max(self.bins-1); weight=coordinate-lower.float()
        logp=scores.log_softmax(-1)
        ce=-(1-weight)*logp.gather(-1,lower[...,None]).squeeze(-1)-weight*logp.gather(-1,upper[...,None]).squeeze(-1)
        return ce[valid].mean() if valid.any() else scores.sum()*0,valid

    def geometric_loss(self,pred,gt):
        return 5*(pred-gt).abs().sum(-1).mean()+2*(1-torch.diag(generalized_box_iou(box_convert(pred,'cxcywh','xyxy'),box_convert(gt,'cxcywh','xyxy')))).mean()

    def loss(self,result,base_boxes,targets,matches,shape_weight=1.):
        zero=result['selected']['scores'].sum()*0
        positive=torch.zeros_like(result['indices'],dtype=torch.bool)
        preds=[]; originals=[]; gts=[]; scores=[]; shape_terms=[]; synthetic=[]; synthetic_gt=[]; synthetic_valid=[]
        for b,t in enumerate(targets):
            if len(t['boxes'])>1: raise ValueError('SBR3 pilot requires one instance; multi-instance teachers need explicit handling')
            src,dst=matches[b]
            for q,g in zip(src.tolist(),dst.tolist()):
                found=(result['indices'][b]==q).nonzero().flatten()
                if not len(found): continue
                j=int(found[0]); positive[b,j]=True
                if not result['selected']['valid'][b,j]: continue
                preds.append(result['boxes'][b,q]); originals.append(base_boxes[b,q].detach()); gts.append(t['boxes'][g].float()); scores.append(result['selected']['scores'][b,j])
            if len(t['boxes']):
                gt=t['boxes'][0].float().detach(); corners=box_convert(gt[None],'cxcywh','xyxy')[0]
                noise=(torch.rand(3,4,device=gt.device)*2-1)*.2; noise[0]=0
                synthetic.append(box_convert(corners[None]+noise*gt[[2,3,2,3]],'xyxy','cxcywh'))
                synthetic_gt.append(gt[None].expand(3,-1)); synthetic_valid.append(True)
                masks=t.get('masks'); quality=t.get('sam_quality'); valid=t.get('sam_instance_valid')
                if masks is not None and len(masks)==1 and quality is not None and float(torch.as_tensor(quality).flatten()[0])>0 and (valid is None or bool(valid[0])):
                    c=gt[:2]; size=(gt[2:]*2).clamp_min(gt.new_tensor([32/result['hw'][1],32/result['hw'][0]]))
                    axis=(torch.arange(32,device=gt.device).float()+.5)/32
                    v,u=torch.meshgrid(axis,axis,indexing='ij')
                    grid=2*(c-size/2+torch.stack((u,v),-1)*size)-1
                    teacher=F.grid_sample(masks[0].float()[None,None],grid[None],align_corners=False)[0,0]
                    prediction=F.grid_sample(result['shape_logits'][b:b+1],grid[None],align_corners=False)[0,0]
                    if teacher.sum()>=.25:
                        p=prediction.sigmoid(); bce=F.binary_cross_entropy_with_logits(prediction,teacher)
                        dice=1-(2*(p*teacher).sum()+1)/(p.sum()+teacher.sum()+1)
                        shape_terms.append(bce+dice)
            else:
                synthetic.append(base_boxes.new_tensor([[.5,.5,.2,.2]]).expand(3,-1)); synthetic_gt.append(synthetic[-1]); synthetic_valid.append(False)
        if preds:
            pred,old,gt,score=map(torch.stack,(preds,originals,gts,scores))
            geometric=self.geometric_loss(pred,gt); position,inside=self.position_loss(score,old,gt)
            corrected=box_convert(pred,'cxcywh','xyxy'); original=box_convert(old,'cxcywh','xyxy'); desired=box_convert(gt,'cxcywh','xyxy')
            pixel=pred.new_tensor([result['hw'][1],result['hw'][0]]*2)
            error_before=float(((desired-original)*pixel).abs().mean().detach()); error_after=float(((desired-corrected)*pixel).abs().mean().detach()); out_of_range=float((~inside).float().mean())
        else: geometric=position=zero; error_before=error_after=out_of_range=0.
        augmented=self.search(result['memory'],result['shape_logits'],torch.stack(synthetic),result['hw'])
        use=torch.tensor(synthetic_valid,device=base_boxes.device,dtype=torch.bool)
        if use.any():
            gt_aug=torch.stack(synthetic_gt)[use].reshape(-1,4); old_aug=torch.stack(synthetic)[use].reshape(-1,4)
            synthetic_position,_=self.position_loss(augmented['scores'][use].reshape(-1,4,self.bins),old_aug,gt_aug)
            synthetic_loss=self.geometric_loss(augmented['boxes'][use].reshape(-1,4),gt_aug)+synthetic_position
        else: synthetic_loss=zero
        neg=result['selected']['scores'][~positive].reshape(-1,self.bins)
        stability=F.cross_entropy(neg,torch.full((len(neg),),self.bins//2,device=neg.device,dtype=torch.long)) if neg.numel() else zero
        shape=torch.stack(shape_terms).mean() if shape_terms else zero
        total=geometric+position+.5*synthetic_loss+.1*stability+shape_weight*shape
        return total,{'geometry':float(geometric.detach()),'position_ce':float(position.detach()),'synthetic_loss':float(synthetic_loss.detach()),
                      'shape_loss':float(shape.detach()),'stability_ce':float(stability.detach()),'positive_queries':len(preds),
                      'shape_supervised_images':len(shape_terms),'outside_search_side_fraction':out_of_range,
                      'positive_side_error_before_px':error_before,'positive_side_error_after_px':error_after}
