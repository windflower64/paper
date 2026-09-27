import copy
import unittest
import torch
from test_sgc_scale_interface import target
from src.core import YAMLConfig
from src.zoo.dfine.sam_group_contrast import group_loss
from src.zoo.dfine.sam_scale_supervision import rescue_loss, scale_losses
from tools.sgc_scale_interface import scale_losses as prototype


class ScaleTrainingTest(unittest.TestCase):
    def test_prototype_exact(self):
        s4 = torch.randn(3, 4, 16, 16, requires_grad=True)
        s8 = torch.randn(3, 8, 8, 8, requires_grad=True)
        targets = [target(16), target(), target(accepted=False)]
        for arm in ('sam', 'box'):
            old, routes = prototype(s4, s8, targets, arm)
            new, new_routes = scale_losses(s4, s8, targets, arm)
            self.assertEqual(routes, new_routes)
            for key in old:
                torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
                g1 = torch.autograd.grad(old[key], (s4,s8), allow_unused=True, retain_graph=True)
                g2 = torch.autograd.grad(new[key], (s4,s8), allow_unused=True, retain_graph=True)
                for x,y in zip(g1,g2):
                    if x is None: self.assertIsNone(y)
                    else: torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_box_rescue_keeps_sam_s8(self):
        s4 = torch.randn(2,4,16,16,requires_grad=True)
        s8 = torch.randn(2,8,8,8,requires_grad=True)
        targets = [target(16),target()]
        extra,_ = rescue_loss(s4,s8,targets,'box')
        total = group_loss(s8,targets,'sam') + extra
        actual = torch.autograd.grad(total,s8,retain_graph=True)[0]
        expected = torch.autograd.grad(group_loss(s8,targets,'sam'),s8)[0]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_configs_only_arm_and_output_differ(self):
        configs=[]
        for arm in ('none','box','sam'):
            cfg=YAMLConfig(f'experiments/phase_s/sgc_scale_{arm}_b16a2_20e.yml')
            c=copy.deepcopy(cfg.yaml_cfg)
            self.assertEqual(c['DFINE'].pop('sgc_s4_rescue'),arm)
            self.assertEqual(c['DFINE']['sgc_supervision'],'sam')
            self.assertEqual(c['DFINE']['rgbt_sd2_final_residual_scale'],.5)
            self.assertEqual(c['train_dataloader']['total_batch_size'],16)
            self.assertEqual(c['gradient_accumulation_steps'],2)
            self.assertEqual(c['epochs'],20)
            c.pop('output_dir');c.pop('__include__',None)
            configs.append(c)
        self.assertEqual(configs[0],configs[1]);self.assertEqual(configs[1],configs[2])


if __name__=='__main__': unittest.main()
