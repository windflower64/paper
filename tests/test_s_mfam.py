import unittest
import torch
from src.zoo.dfine.sam_mask_aggregation import SAMMaskAggregation, region_loss


class TestMFAM(unittest.TestCase):
    def test_protocol_has_no_validation_teacher(self):
        from pathlib import Path
        from src.core import YAMLConfig
        root = Path(__file__).resolve().parents[1]
        cfg = YAMLConfig(str(root/'experiments/phase_s/s_mfam1_c_sam_b8a4_20e_testdev_local.yml')).yaml_cfg
        self.assertIsNone(cfg['val_dataloader']['dataset']['sam_mask_root'])
        self.assertEqual(cfg['HGNetv2']['return_idx'], [1,2,3])
        self.assertEqual(cfg['train_dataloader']['total_batch_size'], 8)
        self.assertEqual(cfg['gradient_accumulation_steps'], 4)
        self.assertEqual(cfg['epochs'], 20)

    def test_shape_gradient_and_intervention(self):
        torch.manual_seed(17)
        module = SAMMaskAggregation(8, 16, 8).eval()
        low = torch.randn(2, 8, 8, 10, requires_grad=True)
        high = torch.randn(2, 16, 4, 5, requires_grad=True)
        out, logits = module(low, high)
        self.assertEqual(out.shape, high.shape)
        self.assertEqual(logits.shape, (2, 1, 8, 10))
        out.square().mean().backward()
        self.assertGreater(module.mask_head[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(low.grad.abs().sum().item(), 0)
        module.intervention = 'constant'
        constant, _ = module(low, high)
        self.assertGreater((out - constant).abs().max().item(), 0)
        module.intervention = 'disabled'
        disabled, _ = module(low, high)
        torch.testing.assert_close(disabled, high, rtol=0, atol=0)

    def test_rejected_masks_are_not_background(self):
        logits = torch.randn(2, 1, 8, 10, requires_grad=True)
        rejected = {'boxes': torch.tensor([[.5, .5, .2, .2]]),
                    'masks': torch.zeros(1, 64, 80), 'sam_quality': torch.tensor([0.])}
        empty = {'boxes': torch.empty(0, 4), 'masks': torch.empty(0, 64, 80),
                 'sam_quality': torch.tensor([0.])}
        loss = region_loss(logits, [rejected, empty], 'sam')
        loss.backward()
        self.assertEqual(logits.grad[0].abs().sum().item(), 0)
        self.assertGreater(logits.grad[1].abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_box_control_uses_same_quality_selection(self):
        logits = torch.zeros(1, 1, 8, 10, requires_grad=True)
        target = {'boxes': torch.tensor([[.5, .5, .3, .3]]),
                  'masks': torch.ones(1, 64, 80), 'sam_quality': torch.tensor([0.])}
        self.assertEqual(region_loss(logits, [target], 'box').item(), 0)
        target['sam_quality'] = torch.tensor([1.])
        self.assertGreater(region_loss(logits, [target], 'box').item(), 0)


if __name__ == '__main__':
    unittest.main()
