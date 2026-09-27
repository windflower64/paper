import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.sgc_scale_interface import scale_losses
from src.zoo.dfine.sam_group_contrast import group_loss


def target(size=8, accepted=True):
    mask = torch.zeros(1, 64, 64, dtype=torch.uint8)
    mask[:, 24:24+size, 24:24+size] = 1
    center = (24+size/2)/64
    return {'masks':mask, 'boxes':torch.tensor([[center,center,size/64,size/64]]),
            'sam_quality':torch.tensor([float(accepted)])}


class ScaleInterfaceTest(unittest.TestCase):
    def features(self, batch=1):
        return (torch.randn(batch,4,16,16,requires_grad=True),
                torch.randn(batch,8,8,8,requires_grad=True))

    def test_existing_s8_exact_preserved(self):
        s4,s8=self.features()
        losses,routes=scale_losses(s4,s8,[target(16)],'sam')
        self.assertEqual(routes,['S8'])
        torch.testing.assert_close(losses['total'],group_loss(s8,[target(16)],'sam'),rtol=0,atol=0)
        self.assertEqual(float(losses['rescue']),0)

    def test_rescue_and_common_routes(self):
        s4,s8=self.features()
        for arm in ['sam','box']:
            losses,routes=scale_losses(s4,s8,[target()],arm)
            self.assertEqual(routes,['S4_rescue'])
            self.assertEqual(float(losses['s8']),0)
            grad=torch.autograd.grad(losses['rescue'],s4,retain_graph=True)[0]
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.norm()),0)

    def test_rejected_target_zero(self):
        s4,s8=self.features()
        losses,routes=scale_losses(s4,s8,[target(accepted=False)],'sam')
        self.assertEqual(routes,['skip'])
        losses['total'].backward()
        self.assertEqual(float(s4.grad.abs().sum()),0)
        self.assertEqual(float(s8.grad.abs().sum()),0)

    def test_mixed_batch_preserves_old_normalization(self):
        s4,s8=self.features(2)
        targets=[target(16),target()]
        losses,routes=scale_losses(s4,s8,targets,'sam')
        self.assertEqual(routes,['S8','S4_rescue'])
        torch.testing.assert_close(losses['s8'],group_loss(s8,targets,'sam'),rtol=0,atol=0)
        solo,_=scale_losses(s4[1:],s8[1:],targets[1:],'sam')
        torch.testing.assert_close(losses['rescue'],solo['rescue']/2)

    def test_does_not_modify_features(self):
        s4,s8=self.features()
        before=[s4.detach().clone(),s8.detach().clone()]
        scale_losses(s4,s8,[target()],'sam')
        torch.testing.assert_close(s4,before[0],rtol=0,atol=0)
        torch.testing.assert_close(s8,before[1],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
