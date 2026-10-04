"""Multi-object SAM supervision smoke test for the external RGB-T dataset."""

import torch

from src.zoo.dfine.sam_group_contrast import selection
from src.zoo.dfine.sam_query_evidence_reader import sqer_shape_loss


def test_two_instances_reach_group_and_query_supervision():
    size = 128
    boxes = torch.tensor([[0.30, 0.50, 0.20, 0.20], [0.70, 0.50, 0.20, 0.20]])
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    masks = torch.stack([
        ((xx.float() / size - cx).abs() <= 0.10)
        & ((yy.float() / size - 0.50).abs() <= 0.10)
        for cx in (0.30, 0.70)
    ]).to(torch.uint8)
    target = {"boxes": boxes, "masks": masks, "sam_quality": torch.tensor([1.0])}
    positives, negatives = selection(target, (32, 32), "sam")
    assert len(positives) >= 2 and len(negatives) >= 4

    roi_size = 16
    axes = []
    for center_x in (0.30, 0.70):
        x = torch.linspace(center_x - 0.20, center_x + 0.20, roi_size) * 2 - 1
        y = torch.linspace(0.30, 0.70, roi_size) * 2 - 1
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        axes.append(torch.stack([grid_x, grid_y], dim=-1))
    outputs = {
        "sqer_attention": torch.full((1, 2, 3, roi_size * roi_size), 1 / (roi_size * roi_size)),
        "sqer_roi_grid": torch.stack(axes).unsqueeze(0),
        "sqer_query_indices": torch.tensor([[3, 7]]),
    }
    loss, diagnostics = sqer_shape_loss(
        outputs, [target], [(torch.tensor([3, 7]), torch.tensor([0, 1]))]
    )
    assert torch.isfinite(loss)
    assert diagnostics["sqer_supervised_queries"].item() == 2
