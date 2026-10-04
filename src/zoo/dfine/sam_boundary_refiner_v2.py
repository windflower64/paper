"""SBR2: let localization learn its own S4/S8 feature encoder.

Copied pretrained local projections start with identical feature values to
SQER2, but their gradients and parameters are independent of that classifier.
No SAM masks enter this forward path.
"""
import copy
import torch
import torch.nn.functional as F
from .sam_boundary_refiner import SAMBoundaryRefiner


class SAMBoundaryRefinerV2(SAMBoundaryRefiner):
    def __init__(self,reader,initial_refiner_state):
        super().__init__()
        self.load_state_dict(initial_refiner_state,strict=True)
        self.s4_projection=copy.deepcopy(reader.s4_projection).requires_grad_(True)
        self.s8_projection=copy.deepcopy(reader.s8_projection).requires_grad_(True)
        self.fusion=copy.deepcopy(reader.fusion).requires_grad_(True)

    def encode(self,s4,s8):
        fine=self.s4_projection(s4.detach())
        coarse=self.s8_projection(s8.detach())
        coarse=F.interpolate(coarse,size=fine.shape[-2:],mode='bilinear',align_corners=False)
        return self.fusion(torch.cat((fine,coarse),dim=1))

    def forward(self,s4,s8,queries,base_logits,boxes,image_hw):
        memory=self.encode(s4,s8)
        result=super().forward(memory,queries,base_logits,boxes,image_hw)
        result['image_hw']=tuple(image_hw)
        return result
