"""Isolated training adapter for report137; production DFINE remains unchanged."""
import json
from pathlib import Path
import torch
from torch import nn
from src.zoo.dfine.dfine import DFINE
from tools.sam_query_readout import QueryReadout,CrossAttentionReadout
from tools.query_reachable_guidance import reachable_geometry_loss


class QueryGuidedDFINE(DFINE):
    def forward(self,x,targets=None):
        wrapper=self.decoder.decoder.layers[0].cross_attn
        wrapper.context={}
        handle=self.backbone.stages[0].register_forward_hook(
            lambda m,args,result:wrapper.context.update(s4=result))
        try:
            output=super().forward(x,targets)
            if self.training and targets is not None:
                dn=output['dn_meta']['dn_num_split'][0] if output.get('dn_meta') else 0
                output['query_read_locations']=wrapper.reader.locations[:,dn:]
                output['query_read_references']=wrapper.reader.references[:,dn:]
            return output
        finally:
            handle.remove();wrapper.context.clear()
            wrapper.reader.locations=None;wrapper.reader.references=None


def install(model):
    model.__class__=QueryGuidedDFINE
    layer=model.decoder.decoder.layers[0]
    reader=QueryReadout(64,layer.cross_attn.embed_dim)
    reader.references=None
    layer.cross_attn=CrossAttentionReadout(layer.cross_attn,reader,{})
    return model


class GuidanceCriterion(nn.Module):
    def __init__(self,base,arm,telemetry=None):
        super().__init__();self.base=base;self.arm=arm;self.telemetry=telemetry

    def forward(self,outputs,targets,**kwargs):
        losses=self.base(outputs,targets,**kwargs)
        if 'query_read_locations' not in outputs:return losses
        indices=self.base.matcher(outputs,targets)['indices']
        # Always compute both targets for equal pipeline and explicit telemetry.
        aux={};counts={}
        for arm in ('sam','box'):
            aux[arm],counts[arm]=reachable_geometry_loss(outputs['query_read_locations'],
                outputs['query_read_references'],targets,indices,arm)
        assert counts['sam']==counts['box']
        if self.arm!='none':losses['loss_query_geometry']=aux[self.arm]
        if self.telemetry:
            row={'epoch':kwargs.get('epoch'), 'step':kwargs.get('step'), 'batch':len(targets),
                 'eligible':counts['sam'],'sam_geometry':float(aux['sam'].detach()),
                 'box_geometry':float(aux['box'].detach()),'applied':self.arm}
            with Path(self.telemetry).open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        return losses
