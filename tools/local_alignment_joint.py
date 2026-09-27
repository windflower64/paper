"""RGB-referenced coarse alignment and local cross-modal attention before encoder."""
import torch
from torch import nn
from torch.nn import functional as F
from mbudet_interface import sample_flow, supervision


class LocalAlignmentLevel(nn.Module):
    def __init__(self, channels, dim=128, heads=4):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.coarse_enabled = True
        self.rgb_projection = nn.Sequential(nn.Conv2d(channels, dim, 1), nn.GroupNorm(8, dim))
        self.ir_projection = nn.Sequential(nn.Conv2d(channels, dim, 1), nn.GroupNorm(8, dim))
        self.coarse = nn.Sequential(nn.Conv2d(dim*2, dim, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(dim, 2, 3, padding=1))
        self.refine = nn.Conv2d(dim*2, 18, 3, padding=1)
        self.query = nn.Conv2d(dim, dim, 1)
        self.key = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(heads, 9))
        self.gate = nn.Sequential(nn.Conv2d(dim*4, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.complement = nn.Sequential(nn.Conv2d(dim*3, dim*2, 1), nn.GELU(),
            nn.Conv2d(dim*2, dim*2, 3, padding=1, groups=dim*2), nn.GELU(), nn.Conv2d(dim*2, dim, 1))
        self.output = nn.Conv2d(dim, channels, 1, bias=False)
        for m in (self.coarse[-1], self.refine, self.output):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        self.register_buffer('anchors', torch.tensor([(x*2., y*2.) for y in (-1, 0, 1)
                                                      for x in (-1, 0, 1)]))

    def forward(self, rgb, ir):
        r, t = self.rgb_projection(rgb), self.ir_projection(ir)
        pair = torch.cat((r, t), 1)
        flow = self.coarse(pair) if self.coarse_enabled else r.new_zeros((r.shape[0],2,*r.shape[-2:]))
        b, c, h, w = r.shape
        local = self.refine(pair).reshape(b, 9, 2, h, w).tanh()
        q = self.query(r).reshape(b, self.heads, c//self.heads, h, w)
        k, v = self.key(t), self.value(t)
        scores, values = [], []
        for i in range(9):
            shift = flow + self.anchors[i][None, :, None, None] + local[:, i]
            ki = sample_flow(k, shift).reshape_as(q)
            vi = sample_flow(v, shift).reshape_as(q)
            scores.append((q.float()*ki.float()).sum(2)/(c//self.heads)**.5 + self.position_bias[:, i][None, :, None, None])
            values.append(vi)
        attention = torch.stack(scores, 2).softmax(2).to(r.dtype)
        selected = (torch.stack(values, 2)*attention.unsqueeze(3)).sum(2).reshape_as(r)
        gate = self.gate(torch.cat((r, selected, r-selected, r*selected), 1)).sigmoid()
        increment = self.output(self.complement(torch.cat((selected, selected-r, selected*r), 1))*gate)
        return rgb+increment, flow, increment


class JointDetector(nn.Module):
    def __init__(self, detector, arm='fusion'):
        super().__init__()
        self.detector, self.arm = detector, arm
        detector.sd2_conditioner = None
        detector.thermal_encoder = nn.Identity()  # no separate IR encoder after early fusion
        detector.rgbt_freeze_thermal_stream = False
        detector.rgbt_train_sdtec_only = False
        detector.sqmi_train_only = False
        detector.requires_grad_(False)
        for name in ('backbone', 'encoder', 'decoder'):
            for key, parameter in getattr(detector, name).named_parameters():
                if parameter.is_floating_point() and key not in ('up', 'reg_scale'):
                    parameter.requires_grad_(True)
        if arm == 'fusion':
            for parameter in detector.thermal_backbone.parameters():
                if parameter.is_floating_point():
                    parameter.requires_grad_(True)
        channels = detector.encoder.in_channels
        self.alignment = nn.ModuleList([LocalAlignmentLevel(c) for c in channels]) if arm == 'fusion' else nn.ModuleList()
        self.epoch = -1
        self.telemetry = {}

    def train(self, mode=True):
        super().train(mode)
        for m in self.detector.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        return self

    def set_training_epoch(self, epoch):
        self.epoch = int(epoch)
        self.detector.set_training_epoch(epoch)

    def forward(self, samples, targets=None):
        d = self.detector
        rgb = d.backbone(samples[:, :3])
        flows, ratios = [], []
        if self.arm == 'fusion':
            ir = d.thermal_backbone(samples[:, 3:])
            fused = []
            for r, t, module in zip(rgb, ir, self.alignment):
                f, flow, increment = module(r, t)
                fused.append(f); flows.append(flow)
                ratios.append(float(increment.detach().float().square().mean().sqrt()/r.detach().float().square().mean().sqrt().clamp_min(1e-6)))
            rgb = fused
        outputs = d.decoder(d.encoder(rgb), targets)
        self.telemetry = {'residual_rms_ratios': ratios}
        if self.training and targets is not None and flows:
            losses, errors, counts = zip(*(supervision(f, targets) for f in flows))
            outputs['alignment_aux_loss'] = .1*torch.stack(losses).mean()
            self.telemetry.update(field_loss=float(torch.stack(losses).mean().detach()),
                                  field_error_cells=[float(e) for e in errors])
        return outputs
