import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.sam_query_readout import QueryReadout,role_points,geometry_loss


class ReadoutTests(unittest.TestCase):
    def test_zero_feature_exact(self):
        reader=QueryReadout(8,16)
        out=reader(torch.zeros(2,8,16,16),torch.randn(2,5,16),torch.full((2,5,1,4),.3))
        self.assertEqual(float(out.abs().max()),0)

    def test_reading_gradients(self):
        reader=QueryReadout(8,16)
        feature=torch.randn(2,8,16,16,requires_grad=True)
        out=reader(feature,torch.randn(2,5,16),torch.full((2,5,1,4),.3))
        out.square().mean().backward()
        for p in (feature,reader.offsets.weight,reader.project.weight):
            self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.norm()),0)

    def test_shape_target_and_gradient_differ(self):
        y,x=torch.meshgrid(torch.arange(64),torch.arange(64),indexing='ij')
        mask=((x-32)**2+(y-32)**2<100)[None]
        target={'masks':mask,'boxes':torch.tensor([[.5,.5,.35,.35]]),'sam_quality':torch.tensor(1.)}
        self.assertFalse(torch.equal(role_points(target,'sam')[1],role_points(target,'box')[1]))
        loc=(torch.rand(1,1,12,2)*.4+.3).requires_grad_()
        gradients=[]
        for arm in ('sam','box'):
            loss,n=geometry_loss(loc,[target],[(torch.tensor([0]),torch.tensor([0]))],arm)
            self.assertEqual(n,1);gradients.append(torch.autograd.grad(loss,loc)[0])
        self.assertGreater(float((gradients[0]-gradients[1]).abs().max()),0)

if __name__=='__main__':unittest.main()
