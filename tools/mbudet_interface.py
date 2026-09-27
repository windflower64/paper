"""MBUDet-inspired feature alignment in RGB coordinates (not original reproduction)."""
import torch
from torch import nn
from torch.nn import functional as F


def sample_flow(source, flow):
    """Backward sampling: at RGB output location, read IR at location + flow.

    flow is in FEATURE PIXELS, channel order x/y. Pixel-center grid with
    align_corners=False; a positive x reads to the right in the source.
    """
    b, _, h, w = source.shape
    y, x = torch.meshgrid(torch.arange(h, device=source.device, dtype=torch.float32),
                         torch.arange(w, device=source.device, dtype=torch.float32), indexing='ij')
    grid = torch.stack(((x + .5)*2/w-1, (y + .5)*2/h-1), -1)[None]
    delta = flow.float().permute(0, 2, 3, 1)*flow.new_tensor([2/w, 2/h]).float()
    # FP32 sampler gives meaningful subpixel motion under AMP.
    with torch.autocast(source.device.type, enabled=False):
        sampled = F.grid_sample(source.float(), grid+delta, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
    return sampled.to(source.dtype)


def supervision(flow, targets, expansion=1.5):
    """One physical target per original frame; no Mosaic or cross-target matching.

    Only two-sided, unambiguous targets supervise flow. No IR box is used
    in the detection forward or at inference. Neighborhood includes at least
    one feature cell; offset magnitude is retained, not clipped to a window.
    """
    b, _, h, w = flow.shape
    mask = flow.new_zeros((b, 1, h, w), dtype=torch.float32)
    truth = flow.new_zeros((b, 2, h, w), dtype=torch.float32)
    for i, target in enumerate(targets):
        rgb, ir = target['boxes'], target['infrared_boxes']
        if len(rgb) != 1 or len(ir) != 1:
            continue
        r, t = rgb[0].float(), ir[0].float()
        assert bool(torch.isfinite(r).all() and torch.isfinite(t).all())
        assert bool(((r[:2] >= 0) & (r[:2] <= 1)).all())
        center = r[:2]*r.new_tensor([w, h])
        extent = torch.maximum(r[2:]*r.new_tensor([w, h])*expansion, r.new_tensor([2., 2.]))
        left, top = torch.floor(center-extent/2).long().tolist()
        right, bottom = torch.ceil(center+extent/2).long().tolist()
        left, top, right, bottom = max(0, left), max(0, top), min(w, right), min(h, bottom)
        mask[i, :, top:bottom, left:right] = 1
        truth[i] = ((t[:2]-r[:2])*r.new_tensor([w, h]))[:, None, None]
    squared = (flow.float()-truth).square()*mask
    loss = squared.sum()/(2*mask.sum()).clamp_min(1)
    # Pixel distance is for monitoring, not an additional loss.
    error = (((flow.float()-truth).square().sum(1, keepdim=True)).sqrt()*mask).sum()/mask.sum().clamp_min(1)
    return loss, error.detach(), mask.sum().detach()


class AlignmentLevel(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.offset = nn.Sequential(nn.Conv2d(2*channels, 32, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 2, 3, padding=1))
        nn.init.zeros_(self.offset[-1].weight); nn.init.zeros_(self.offset[-1].bias)
        # Reference spatial attention uses channel max/mean from both modalities.
        self.attention = nn.Conv2d(4, 1, 3, padding=1)
        self.projection = nn.Conv2d(2*channels, channels, 1, bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, rgb, ir, aligned):
        flow = self.offset(torch.cat((rgb, ir), 1))
        read = sample_flow(ir, flow) if aligned else ir
        statistics = torch.cat((rgb.mean(1, True), rgb.amax(1, True),
                                read.mean(1, True), read.amax(1, True)), 1)
        # IR remains the only source of the residual; RGB conditions its weight.
        # Avoid creating a standalone RGB-only correction through concat(rgb,ir).
        weight = self.attention(statistics).sigmoid()
        increment = self.projection(torch.cat((read, read*weight), 1))
        return rgb+increment, flow, increment


class AlignedDetector(nn.Module):
    def __init__(self, detector, aligned=True):
        super().__init__()
        self.detector = detector
        self.aligned = bool(aligned)
        self.bypass = False
        detector.sd2_conditioner = None
        detector.requires_grad_(False)
        self.alignment = nn.ModuleList([AlignmentLevel(detector.encoder.hidden_dim) for _ in range(2)])
        if not aligned:
            for level in self.alignment:
                level.offset.requires_grad_(False)
        self.epoch = -1
        self.telemetry = {}

    def train(self, mode=True):
        super().train(mode)
        self.detector.eval()
        # Keep auxiliary detection outputs and DN supervision for the adapter.
        self.detector.decoder.train(mode)
        for m in self.detector.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        return self

    def set_training_epoch(self, epoch):
        self.epoch = int(epoch)
        self.detector.set_training_epoch(epoch)

    def forward(self, samples, targets=None):
        d = self.detector
        assert samples.shape[1] == 6
        with torch.no_grad():
            rgb = d.encoder(d.backbone(samples[:, :3]))
            ir = d.thermal_encoder(d.thermal_backbone(samples[:, 3:]))
        fused, flows, ratios = [], [], []
        for r, t, level in zip(rgb, ir, self.alignment):
            value, flow, increment = level(r, t, self.aligned)
            fused.append(r if self.bypass else value)
            flows.append(flow)
            ratios.append(increment.detach().float().square().mean().sqrt()/r.float().square().mean().sqrt().clamp_min(1e-6))
        outputs = d.decoder(fused, targets)
        if self.training and targets is not None:
            losses, errors, counts = zip(*(supervision(flow, targets) for flow in flows))
            outputs['alignment_aux_loss'] = .1*torch.stack(losses).mean() if self.aligned else flows[0].sum()*0
            self.telemetry = dict(field_loss=float(torch.stack(losses).mean().detach()),
                field_error_cells=[float(e) for e in errors], supervised_cells=[float(c) for c in counts],
                residual_rms_ratios=[float(r) for r in ratios])
        return outputs


class AlignmentCriterion(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.base = criterion

    def forward(self, outputs, targets, **kwargs):
        values = self.base(outputs, targets, **kwargs)
        if 'alignment_aux_loss' in outputs:
            values['loss_alignment_field'] = outputs['alignment_aux_loss']
        return values
