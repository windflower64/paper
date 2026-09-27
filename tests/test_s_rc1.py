import importlib.util
from pathlib import Path
import torch

path=Path(__file__).resolve().parents[1]/'src/zoo/dfine/sam_mask_aggregation.py'
spec=importlib.util.spec_from_file_location('rc1_module',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def sample():
    masks=torch.zeros(1,64,80);masks[:,20:44,28:52]=1
    return dict(masks=masks,boxes=torch.tensor([[.5,.5,.3,.375]]),sam_quality=torch.tensor(1.))


def test_sources_finite_gradients():
    for source in ['none','box','sam','edge']:
        logits=torch.zeros(1,1,8,10,requires_grad=True)
        loss=module.region_loss(logits,[sample()],source)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(logits.grad).all()
        assert (logits.grad.abs().sum()==0) if source=='none' else (logits.grad.abs().sum()>0)


def test_rejected_positive_is_not_background():
    target=sample();target['sam_quality']=torch.tensor(0.)
    logits=torch.zeros(1,1,8,10,requires_grad=True)
    assert module.region_loss(logits,[target],'edge').item()==0


def test_region_edge_are_different_targets():
    grads=[]
    for source in ['sam','edge']:
        logits=torch.zeros(1,1,8,10,requires_grad=True)
        module.region_loss(logits,[sample()],source).backward()
        grads.append(logits.grad.clone())
    assert not torch.allclose(*grads)


def test_inference_shape_and_detector_gradient():
    torch.manual_seed(0)
    net=module.SAMMaskAggregation(low_channels=8,high_channels=16,width=4)
    low=torch.randn(2,8,16,20);high=torch.randn(2,16,8,10)
    output,logits=net(low,high)
    assert output.shape==high.shape and logits.shape==(2,1,16,20)
    output.square().mean().backward()
    assert net.mask_head[-1].weight.grad.abs().sum()>0
