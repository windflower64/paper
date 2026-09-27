import unittest
import torch
from src.nn.backbone.sam_boundary_relation import SAMBoundaryRelation

class TestSBRA1(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.module=SAMBoundaryRelation(8,4)
        self.x=torch.randn(2,8,8,10,requires_grad=True)
        self.base=torch.randn(2,8,4,5)

    def test_identity(self):
        torch.testing.assert_close(self.module(self.x,self.base),self.base,rtol=0,atol=0)

    def test_detection_reaches_relation_after_projection_update(self):
        opt=torch.optim.SGD(self.module.parameters(),lr=.1)
        for step in range(2):
            opt.zero_grad(); self.module(self.x,self.base).square().mean().backward()
            self.assertGreater(self.module.output.weight.grad.abs().sum().item(),0)
            if step==1:self.assertGreater(self.module.relation[-1].weight.grad.abs().sum().item(),0)
            opt.step()

    def test_aux_detaches_feature(self):
        self.module(self.x,self.base)
        loss=self.module.relation_loss(torch.rand(2,4,5,4,4),torch.ones(2,4,5,dtype=torch.bool))
        loss.backward()
        self.assertIsNone(self.x.grad)
        self.assertGreater(self.module.relation[-1].weight.grad.abs().sum().item(),0)
        self.assertIsNone(self.module.project.weight.grad)

    def test_invalid_support_zero(self):
        self.module(self.x,self.base)
        loss=self.module.relation_loss(torch.ones(2,4,5,4,4),torch.zeros(2,4,5,dtype=torch.bool))
        self.assertEqual(loss.item(),0)

    def test_supervision_sources_and_inference(self):
        self.module(self.x,self.base)
        target=dict(masks=torch.ones(1,64,80),boxes=torch.tensor([[.5,.5,.5,.5]]),sam_quality=torch.tensor(1.))
        for source in ('sam','box'):
            self.assertTrue(torch.isfinite(self.module.supervision([target,target],source)))
        self.module.eval()
        self.module(self.x,self.base)
        self.assertIsNone(self.module._aux_input)

if __name__=='__main__':unittest.main()
