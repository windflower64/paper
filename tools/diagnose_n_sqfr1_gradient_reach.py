#!/usr/bin/env python3
"""追踪N-SQFR1首步中选中查询、匹配查询和框梯度的交集。"""

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig


def main():
    torch.manual_seed(20260904)
    torch.cuda.manual_seed_all(20260904)
    config = YAMLConfig(
        str(ROOT / "experiments/phase_n/n_sqfr1_c_gq1_b8a4_20e_testdev_local.yml")
    )
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = config.model
    checkpoint = torch.load(
        ROOT.parent / "weights/m_sd2_joint_coco_thermal_identity_init.pth",
        map_location="cpu",
        weights_only=False,
    )
    source = checkpoint.get("ema", {}).get("module")
    if not isinstance(source, dict):
        source = checkpoint["model"]
    own = model.state_dict()
    model.load_state_dict(
        {
            key: value
            for key, value in source.items()
            if key in own and own[key].shape == value.shape
        },
        strict=False,
    )
    device = torch.device("cuda")
    model = model.to(device).train()
    criterion = config.criterion.to(device).train()
    samples, targets = next(iter(config.train_dataloader))
    samples = samples.to(device)
    targets = [
        {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
        for target in targets
    ]

    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(samples, targets=targets)
    outputs["pred_boxes"].retain_grad()
    matches = criterion.matcher(outputs, targets)["indices"]
    selected = model.decoder.sqfr_refiner.last_selected_indices
    with torch.autocast("cuda", enabled=False):
        losses = criterion(
            outputs,
            targets,
            epoch=0,
            step=0,
            global_step=0,
            epoch_step=len(config.train_dataloader),
        )
        total = sum(losses.values())
    total.backward()

    rows = []
    gradient_mask = outputs["pred_boxes"].grad.abs().sum(dim=-1) > 0
    for batch_index, (matched_queries, _targets) in enumerate(matches):
        selected_set = set(selected[batch_index].detach().cpu().tolist())
        matched_set = set(matched_queries.detach().cpu().tolist())
        gradient_set = set(
            gradient_mask[batch_index].nonzero().flatten().detach().cpu().tolist()
        )
        rows.append(
            {
                "batch": batch_index,
                "selected": len(selected_set),
                "matched": sorted(matched_set),
                "selected_matched": sorted(selected_set & matched_set),
                "gradient_queries": sorted(gradient_set),
                "selected_gradient": sorted(selected_set & gradient_set),
            }
        )
    final = model.decoder.sqfr_refiner.delta_head[-1].weight.grad
    print(
        {
            "rows": rows,
            "selected_matched_total": sum(len(row["selected_matched"]) for row in rows),
            "selected_gradient_total": sum(len(row["selected_gradient"]) for row in rows),
            "delta_head_gradient_max": (
                0.0 if final is None else float(final.detach().abs().max())
            ),
        }
    )


if __name__ == "__main__":
    main()
