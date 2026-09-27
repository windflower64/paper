#!/usr/bin/env python3
import math
import sys

import torch


def main(repo):
    sys.path.insert(0, repo)
    from src.zoo.dfine.dfine_criterion import DFINECriterion

    base = DFINECriterion(
        matcher=None,
        weight_dict={},
        losses=[],
        reg_max=32,
    )
    shape = DFINECriterion(
        matcher=None,
        weight_dict={},
        losses=[],
        reg_max=32,
        fgl_edge_weight_mode="shape",
    )
    reverse = DFINECriterion(
        matcher=None,
        weight_dict={},
        losses=[],
        reg_max=32,
        fgl_edge_weight_mode="shape_reverse",
    )
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.1], [0.5, 0.5, 0.2, 0.2]])
    weights = shape._fgl_edge_weights(boxes, torch.float32).reshape(-1, 4)
    reverse_weights = reverse._fgl_edge_weights(boxes, torch.float32).reshape(-1, 4)
    assert torch.equal(weights.mean(-1), torch.ones(2))
    assert torch.equal(reverse_weights.mean(-1), torch.ones(2))
    assert weights[0, 1] > weights[0, 0]
    assert torch.equal(weights[0, [0, 2]], reverse_weights[0, [1, 3]])
    assert torch.equal(weights[0, [1, 3]], reverse_weights[0, [0, 2]])
    assert torch.equal(weights[1], torch.ones(4))

    logits = torch.zeros(2, 4, 33)
    logits[0, 0, 3] = 10
    logits[0, 1, :2] = 5
    logits[0, 2, :4] = 3
    logits[0, 3] = 0
    reliability = base._ddf_edge_reliability(logits)
    assert torch.allclose(reliability.mean(-1), torch.ones(2), atol=1e-6)
    assert reliability[0, 0] > reliability[0, 1] > reliability[0, 2] > reliability[0, 3]
    assert torch.allclose(reliability[1], torch.ones(4), atol=1e-6)
    assert not reliability.requires_grad

    print({
        "shape_wide": weights[0].tolist(),
        "shape_square": weights[1].tolist(),
        "reverse_wide": reverse_weights[0].tolist(),
        "entropy_reliability": reliability[0].tolist(),
        "mean_one": reliability.mean(-1).tolist(),
        "log_bins": math.log(33),
    })


if __name__ == "__main__":
    main(sys.argv[1])
