#!/usr/bin/env python3
"""One-batch finite forward/loss/backward test for an experiment config."""
import argparse
import sys
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    a=p.parse_args(); sys.path.insert(0,str(a.repo))
    from src.core import YAMLConfig
    c=YAMLConfig(str(a.config)); c.yaml_cfg["HGNetv2"]["pretrained"]=False
    model,criterion=c.model.cuda().train(),c.criterion.cuda().train()
    state=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    weights=state["model"]
    own=model.state_dict(); compatible={k:v for k,v in weights.items() if k in own and own[k].shape==v.shape}
    info=model.load_state_dict(compatible,strict=False)
    samples,targets=next(iter(c.train_dataloader)); samples=samples.cuda()
    targets=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in t.items()} for t in targets]
    with torch.autocast("cuda",dtype=torch.float16): outputs=model(samples,targets=targets)
    with torch.autocast("cuda",enabled=False):
        losses=criterion(outputs,targets,epoch=0,step=0,global_step=0,epoch_step=1); total=sum(losses.values())
    total.backward()
    grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    print({"config":a.config.name,"batch":tuple(samples.shape),"loss":float(total.detach()),
           "losses":{k:float(v.detach()) for k,v in losses.items() if not torch.isfinite(v).all() or k in ("loss_vfl","loss_bbox","loss_giou","loss_spatial_aux")},
           "outputs_finite":bool(torch.isfinite(outputs["pred_boxes"]).all() and torch.isfinite(outputs["pred_logits"]).all()),
           "grad_tensors":len(grads),"grads_finite":bool(all(torch.isfinite(g).all() for g in grads)),
           "missing":info.missing_keys,"unexpected":info.unexpected_keys,
           "peak_mib":round(torch.cuda.max_memory_allocated()/1024**2,2)})


if __name__=="__main__": main()
