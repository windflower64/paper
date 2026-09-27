"""Feasible local supervision in the same coordinate system as predicted offsets.

Does not change matching, sampling limits, detector losses or inference.
"""
import torch
from tools.sam_query_readout import role_points


def reachable_geometry_loss(locations,references,targets,indices,source):
    if source not in ('sam','box'):raise ValueError(source)
    loss=locations.float().sum()*0.;count=0
    for batch,(queries,objects) in enumerate(indices):
        both={a:role_points(targets[batch],a) for a in ('sam','box')}
        if any(v is None for v in both.values()):continue
        for query,obj in zip(queries,objects):
            if int(obj)!=0:continue
            box=references[batch,query].detach().float()
            center=box[:2];scale=box[2:].clamp_min(1e-6)
            selected={}
            for arm,roles in both.items():
                selected[arm]=[]
                for points in roles:
                    offsets=(points.float()-center)/scale
                    selected[arm].append(offsets[(offsets.abs()<1).all(-1)])
            # Identical eligibility for both arms; other detector losses never skip.
            if any(len(p)<4 for roles in selected.values() for p in roles):continue
            points=((locations[batch,query].float()-center)/scale).reshape(3,4,2)
            for role,teacher in zip(points,selected[source]):
                distances=torch.cdist(role,teacher).square()
                loss=loss+.5*(distances.min(0).values.mean()+distances.min(1).values.mean())/3
            count+=1
    # Per physical batch, so sparse eligibility does not boost the surviving images.
    return loss/max(1,len(targets)),count
