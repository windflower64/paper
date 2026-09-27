import unittest
import torch
from src.zoo.dfine.sam_support_shape import SupportShapeAggregation, support_shape_loss


class TestSupportShape(unittest.TestCase):
    def test_partition_and_detection_gradients(self):
        torch.manual_seed(3)
        m=SupportShapeAggregation(8,16,8).eval()
        low,high=torch.randn(2,8,8,10),torch.randn(2,16,4,5)
        out,logits=m(low,high)
        self.assertEqual(logits.shape,(2,2,8,10))
        out.square().mean().backward()
        for channel in (0,1):
            self.assertGreater(m.mask_head[-1].weight.grad[channel].abs().sum().item(),0)
        weights=m.partition(logits)
        torch.testing.assert_close(sum(weights),torch.ones_like(weights[0]))
        for mode in ('support_constant','shape_constant'):
            m.intervention=mode
            changed,_=m(low,high)
            self.assertGreater((out-changed).abs().max().item(),0)
        m.intervention='disabled'
        torch.testing.assert_close(m(low,high)[0],high,rtol=0,atol=0)

    def test_shape_only_supervised_inside_support(self):
        logits=torch.zeros(1,2,8,8,requires_grad=True)
        mask=torch.zeros(1,64,64);mask[:,24:40,24:40]=1
        target={'boxes':torch.tensor([[.5,.5,.5,.5]]),'masks':mask,'sam_quality':torch.tensor([1.])}
        losses=support_shape_loss(logits,[target],'sam')
        sum(losses.values()).backward()
        self.assertEqual(logits.grad[0,1,:2].abs().sum().item(),0)
        self.assertGreater(logits.grad[0,1,2:6,2:6].abs().sum().item(),0)
        self.assertTrue(torch.isfinite(sum(losses.values())))

    def test_empty_and_rejected(self):
        logits=torch.zeros(2,2,8,8,requires_grad=True)
        empty={'boxes':torch.empty(0,4),'masks':torch.empty(0,64,64),'sam_quality':torch.tensor([0.])}
        rejected={'boxes':torch.tensor([[.5,.5,.5,.5]]),'masks':torch.zeros(1,64,64),'sam_quality':torch.tensor([0.])}
        sum(support_shape_loss(logits,[empty,rejected],'sam').values()).backward()
        self.assertGreater(logits.grad[0,0].abs().sum().item(),0)
        self.assertEqual(logits.grad[0,1].abs().sum().item(),0)
        self.assertEqual(logits.grad[1].abs().sum().item(),0)


if __name__=='__main__':unittest.main()
