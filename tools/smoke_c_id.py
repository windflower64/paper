#!/usr/bin/env python3
from pathlib import Path
import sys

import torch

root = Path("/root/autodl-tmp/rgbt_experiments")
repo = root / "D-FINE"
sys.path[:0] = [str(repo), str(root / "diagnostics")]

from src.core import YAMLConfig
from src.misc.channel_pruning import apply_channel_pruning
import diagnose_downsampling as helper


def target(boxes):
    b = torch.tensor(boxes, dtype=torch.float32, device="cuda").reshape(-1, 4)
    b[:, 0] = (b[:, 0] + b[:, 2] / 2) / 640
    b[:, 1] = (b[:, 1] + b[:, 3] / 2) / 512
    b[:, 2] /= 640
    b[:, 3] /= 512
    return {"boxes": b, "labels": torch.zeros(len(b), dtype=torch.long, device="cuda")}


def main():
    state = torch.load(
        root / "outputs/dfine_n_visible_640x512_full_seed0/best_stg1.pth",
        map_location="cpu", weights_only=False,
    )["ema"]["module"]
    records = helper.choose(
        helper.load_coco(root / "data/antiuav6k_common", "val"), 2, 20260810
    )
    items = [helper.image_tensor(root / "data/antiuav6k_common", "val", r) for r in records]
    images = torch.stack([x[0] for x in items]).cuda()
    targets = [target(x[1]) for x in items]

    for name in ("task", "magnitude", "random"):
        config = repo / f"experiments/phase_c/c_id_s16_r25_{name}.yml"
        cfg = YAMLConfig(str(config))
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        model = cfg.model
        model.load_state_dict(state, strict=True)
        manifest = apply_channel_pruning(model, cfg.yaml_cfg["channel_prune"])
        model.cuda().train()
        criterion = cfg.criterion.cuda()
        output = model(images, targets=targets)
        losses = criterion(output, targets, epoch=0, step=0, global_step=0, epoch_step=1)
        loss = sum(losses.values())
        loss.backward()
        rootconv = model.backbone.stages[2].blocks[0].aggregation[1].conv
        projection = model.encoder.input_proj[0].conv
        print(
            name, "root", rootconv.out_channels, "projection", projection.in_channels,
            "loss", float(loss.detach()), "params", sum(p.numel() for p in model.parameters()),
            "removed_head", manifest["removed_channels"][:8], flush=True,
        )
        del model, criterion, output, losses, loss
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
