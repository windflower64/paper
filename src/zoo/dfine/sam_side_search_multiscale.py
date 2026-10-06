"""Read S16 at the same candidate strips; add zero-initialized context scores."""
import torch
from torch import nn
import torch.nn.functional as F
from .sam_side_search import SAMSideSearch


class SAMSideSearchContext(SAMSideSearch):
    def __init__(self, reader, s16_channels):
        super().__init__(reader)
        self.context_projection=nn.Sequential(nn.Conv2d(s16_channels,32,1,bias=False),nn.GroupNorm(8,32),nn.SiLU())
        self.context_score=nn.Sequential(nn.LayerNorm(296),nn.Linear(296,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.context_score[-1].weight)
        nn.init.zeros_(self.context_score[-1].bias)
        self.context=None

    def forward(self,s4,s8,s16,logits,boxes,hw):
        self.context=self.context_projection(s16.detach())
        return super().forward(s4,s8,logits,boxes,hw)

    def score_candidates(self,feature,grid):
        if self.context is None: raise RuntimeError('Missing S16 context')
        b,k,s,n,r,p,_=grid.shape
        sampled=F.grid_sample(self.context,(2*grid-1).reshape(b,k*s*n*r,p,2),align_corners=False)
        sampled=sampled.reshape(b,32,k,s,n,r,p).mean(-1).permute(0,2,3,4,5,1)
        inside,center,outside=sampled.unbind(-2)
        joined=torch.cat((feature,center,inside,inside-outside),-1)
        return super().score_candidates(feature,grid)+self.context_score(joined).squeeze(-1)
