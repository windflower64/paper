"""Geometry and gradient ownership checks for external teacher feature KD."""
import copy
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.shape_feature_interface import FeatureDetector,common_weights,rectangle_occupancy,roi_grid,schedule


class TinyBackbone(nn.Module):
    return_idx = [2,3]
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,512,1)
    def forward(self,x):
        z = self.conv(x)
        return [z,z]


class TinyDetector(nn.Module):
    sgc_enabled = False
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.encoder = nn.Identity(); self.encoder.in_channels = [512,512]
        self.head = nn.Conv2d(512,5,1)
    def set_training_epoch(self,epoch):
        pass
    def forward(self,x,targets=None):
        z = self.head(self.backbone(x)[0]).mean((2,3))
        return {'pred_logits':z[:,:1], 'pred_boxes':z[:,1:]}


def target():
    mask = torch.zeros(1,16,16); mask[:,5:11,6:10] = 1
    return dict(boxes=torch.tensor([[.5,.5,.5,.5]]),sam_quality=torch.ones(1),
        masks=mask,kd_roi=torch.tensor([.125,.125,.875,.875]),
        kd_extent=torch.tensor([.375,.3125,.625,.6875]),kd_teacher=torch.randn(256,32,32))


def test_selection_is_normalized_and_shape_distinct():
    weights,shape,filled = common_weights(target())
    assert all(torch.allclose(w.sum(),torch.tensor(1.),atol=1e-6) for w in weights.values())
    assert bool((filled>=0).all()) and bool((filled<=1).all())
    # Replace rectangular mask with internal hole, same extent.
    t=target(); t['masks'][:,7:9,7:9]=0
    weights2,_,filled2=common_weights(t)
    assert torch.equal(filled,filled2)
    assert not torch.equal(weights2['sam'],weights['sam'])
    assert torch.equal(weights2['filled'],weights['filled'])


def test_flip_transforms_both_sampling_geometry_and_selection():
    t=target(); original=common_weights(t)
    t['masks']=t['masks'].flip(-1)
    for key in ('kd_roi','kd_extent'):
        x0,x1=t[key][0].item(),t[key][2].item()
        t[key][0],t[key][2]=1-x1,1-x0
    flipped=common_weights(t)
    for name in ('uniform','filled','sam'):
        assert torch.allclose(original[0][name].flip(-1),flipped[0][name],atol=1e-6)


def test_teacher_loss_gradient_and_ema_hook_ownership():
    torch.manual_seed(0)
    d=TinyDetector(); model=FeatureDetector(d,'sam',{'mean':[0.]*256,'std':[1.]*256})
    x=torch.rand(1,3,16,16); t=target()
    output=model(x,[t]); output['feature_kd_loss'].backward()
    assert model.detector.backbone.conv.weight.grad.abs().max()>0
    assert model.kd_projection.weight.grad.abs().max()>0
    assert model.detector.head.weight.grad is None
    assert t['kd_teacher'].grad is None
    assert model.captured is None
    ema=copy.deepcopy(model).eval()
    hook=list(ema.detector.backbone._forward_hooks.values())[0]
    assert hook.__self__ is ema
    model.eval()
    with torch.no_grad():
        a=d(x); b=model(x); c=ema(x)
    assert torch.equal(a['pred_boxes'],b['pred_boxes'])
    assert torch.equal(a['pred_boxes'],c['pred_boxes'])
    assert model.captured is None and ema.captured is None
    model.train(); model.set_training_epoch(10)
    output=model(x,[t])
    assert float(output['feature_kd_loss']) == 0


def test_schedule_and_rectangle_area():
    assert [schedule(i) for i in range(20)]==[1.]*6+[.8,.6,.4,.2]+[0.]*10
    coverage=rectangle_occupancy(torch.tensor([0.,0.,1.,1.]),torch.tensor([.2,.3,.6,.9]))
    assert torch.allclose(coverage.mean(),torch.tensor(.24),atol=1e-6)


if __name__ == '__main__':
    for test in (test_selection_is_normalized_and_shape_distinct,
                 test_flip_transforms_both_sampling_geometry_and_selection,
                 test_teacher_loss_gradient_and_ema_hook_ownership,
                 test_schedule_and_rectangle_area):
        test()
        print('PASS',test.__name__,flush=True)
