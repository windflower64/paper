import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.pilot_s_bci1 import protection,perturb,target_size,texture_perturb
from PIL import Image


class ProtectionTest(unittest.TestCase):
    def test_texture_does_not_add_foreground_halo(self):
        x=torch.full((3,32,32),.7); mask=torch.zeros(32,32); mask[12:20,12:20]=1
        x[:,mask.bool()]=.1
        y=texture_perturb(x,protection(mask),.3)
        torch.testing.assert_close(x,y,rtol=0,atol=2e-7)
        self.assertTrue(torch.equal(x[:,mask.bool()],y[:,mask.bool()]))

    def test_texture_background_changed(self):
        x=torch.rand(3,32,32); mask=torch.zeros(32,32); mask[10:15,10:15]=1
        y=texture_perturb(x,protection(mask),.3)
        self.assertGreater((x-y).abs().sum().item(),0.)
        self.assertTrue(torch.equal(x[:,mask.bool()],y[:,mask.bool()]))

    def test_width_height_protocol(self):
        image=Image.new('RGB',(640,512))
        self.assertEqual(target_size(image,2,'cpu').tolist(),[[640,512],[640,512]])

    def test_foreground_identity(self):
        mask=torch.zeros(32,32); mask[10:15,10:15]=1
        alpha=protection(mask); x=torch.rand(3,32,32)
        y=perturb(x,alpha,.3,[1,-1])
        self.assertTrue(torch.equal(y[:,mask.bool()],x[:,mask.bool()]))
        self.assertGreater((y-x).abs().sum().item(),0.)
        self.assertTrue(((alpha>=0)&(alpha<=1)).all())

    def test_all_foreground_identity(self):
        x=torch.rand(3,32,32); alpha=protection(torch.ones(32,32))
        self.assertTrue(torch.equal(x,perturb(x,alpha,.3,[-1,1])))

    def test_empty_is_global(self):
        x=torch.rand(3,32,32); alpha=protection(torch.zeros(32,32))
        self.assertEqual(alpha.sum().item(),0.)
        y=perturb(x,alpha,.2,[1,-1]); self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(((y>=0)&(y<=1)).all())

    def test_zero_strength(self):
        x=torch.rand(3,32,32); alpha=protection(torch.zeros(32,32))
        torch.testing.assert_close(x,perturb(x,alpha,0,[1,1]),rtol=0,atol=0)


if __name__=='__main__': unittest.main()
