"""CPU contracts: S8 compatibility, normalization, active gradients, empty mask."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.zoo.dfine.sam_group_contrast import group_loss
from src.zoo.dfine.sam_multiscale_group import multiscale_group_loss

torch.manual_seed(0)
mask = torch.zeros(1,64,64)
mask[:,16:48,16:48] = 1
t = {'boxes': torch.tensor([[.5,.5,.5,.5]]), 'masks': mask, 'sam_quality': torch.tensor(1.)}
f = torch.randn(1,8,8,8,requires_grad=True)
old = group_loss(f,[t],'sam')
new, stats = multiscale_group_loss({8:f},[t],{8:1.})
assert torch.allclose(old,new)
copies = {s:f.detach().clone().requires_grad_() for s in (4,8,16)}
multi, stats = multiscale_group_loss(copies,[t],{4:1.,8:1.,16:1.})
assert torch.allclose(new,multi), 'Duplicating scales must not triple loss'
multi.backward()
assert all(x.grad is not None and x.grad.abs().sum()>0 for x in copies.values())
empty = dict(t, masks=torch.zeros_like(mask))
zero, stats = multiscale_group_loss({8:f},[empty],{8:1.})
assert zero.item()==0 and stats['active_images']==0
print('PASS: S8 parity, scale normalization, gradients, invalid-mask skip')
