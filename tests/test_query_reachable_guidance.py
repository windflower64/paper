import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.query_reachable_guidance import reachable_geometry_loss


class ReachableTests(unittest.TestCase):
    def test_common_skip_and_finite_difference(self):
        y,x=torch.meshgrid(torch.arange(64),torch.arange(64),indexing='ij')
        t={'masks':(((x-32)**2+(y-32)**2)<100)[None],
           'boxes':torch.tensor([[.5,.5,.35,.35]]),'sam_quality':torch.tensor(1.)}
        locations=(torch.rand(1,1,12,2)*.4+.3).requires_grad_()
        indices=[(torch.tensor([0]),torch.tensor([0]))]
        gradients=[]
        for arm in ('sam','box'):
            loss,count=reachable_geometry_loss(locations,torch.tensor([[[.5,.5,.35,.35]]]),[t],indices,arm)
            self.assertEqual(count,1);self.assertTrue(0<=loss<=8)
            gradients.append(torch.autograd.grad(loss,locations)[0])
            skipped,n=reachable_geometry_loss(locations,torch.tensor([[[.05,.05,.01,.01]]]),[t],indices,arm)
            self.assertEqual(n,0);self.assertEqual(float(skipped),0)
        self.assertGreater(float((gradients[0]-gradients[1]).norm()),0)

if __name__=='__main__':unittest.main()
