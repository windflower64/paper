"""Actual replacement detector gradient check, no optimizer update."""
import torch
import train_sbr3_replace as task
from preflight_sqer1 import move_targets, seed_all

seed_all(0)
torch.set_num_threads(4)
cfg, source, refiner, _ = task.setup(8)
model = source.model
model.sgc_enabled = True
model.sgc_multiscale = True
model.sgc_s4_rescue = 'none'
model.sgc_decay_start = model.sgc_decay_end = -1
model.backbone.requires_grad_(True)
model.train()
model.thermal_backbone.eval()
model.thermal_encoder.eval()
samples, targets = next(iter(cfg.train_dataloader))
targets = move_targets(targets, torch.device('cuda'))
out = model(samples.cuda(), targets)
loss = out['sgc_group_loss']
assert torch.isfinite(loss) and loss.item()>0
loss.backward()
grads = {str(i): sum(float(p.grad.norm()) for p in stage.parameters() if p.grad is not None)
         for i,stage in enumerate(model.backbone.stages)}
assert grads['0']>0 and grads['1']>0
assert not any(p.grad is not None for p in model.thermal_backbone.parameters())
report = {'status':'PASS','batch_size':8,'loss':float(loss), 'stage_gradient_norm_sums':grads,
          'coverage':model.sgc_multiscale_stats,'sqer_parameters':sum(p.numel() for p in model.sqer.parameters()),
          'note':'Real forward and SGC backward only; no training or optimizer update'}
task.engine.write_json(task.engine.ROOT/'reports/SGC_MULTISCALE_20261004/gradient_preflight.json',report)
print(report,flush=True)
