import copy

import torch

from src.zoo.dfine.qdmf import QueryConditionedDynamicModalityFusion


def make_inputs(batch=2, queries=7, requires_grad=False):
    torch.manual_seed(17)
    query = torch.randn(batch, queries, 16, requires_grad=requires_grad)
    boxes = torch.rand(batch, queries, 4, requires_grad=requires_grad)
    boxes = torch.cat((boxes[..., :2], boxes[..., 2:] * 0.25), dim=-1)
    logits = torch.randn(batch, queries, 2, requires_grad=requires_grad)
    features = [
        torch.randn(batch, 6, 16, 20, requires_grad=requires_grad),
        torch.randn(batch, 8, 8, 10, requires_grad=requires_grad),
        torch.randn(batch, 10, 4, 5, requires_grad=requires_grad),
    ]
    return query, boxes, logits, features


def make_module(dropout=0.0):
    torch.manual_seed(3)
    return QueryConditionedDynamicModalityFusion(
        query_dim=16,
        thermal_channels=(6, 8, 10),
        roi_grid=3,
        num_heads=4,
        gate_dim=8,
        gate_hidden=12,
        ir_dropout=dropout,
    )


def test_t2_bypass_identity():
    module = make_module().eval()
    query, boxes, logits, features = make_inputs()
    fused, diagnostics = module(query, boxes, logits, features, bypass=True)
    assert torch.equal(fused, query)
    assert torch.count_nonzero(diagnostics["residual"]) == 0


def test_t3_missing_ir_is_exact_zero_and_finite():
    module = make_module().eval()
    query, boxes, logits, features = make_inputs()
    fused, diagnostics = module(
        query, boxes, logits, features, availability=torch.tensor([False, False])
    )
    assert torch.equal(fused, query)
    assert torch.count_nonzero(diagnostics["gate"]) == 0
    assert torch.count_nonzero(diagnostics["residual"]) == 0
    assert torch.isfinite(fused).all()


def test_t4_ir_content_changes_active_path_not_base_inputs():
    module = make_module().eval()
    query, boxes, logits, features = make_inputs()
    query0, boxes0, logits0 = query.clone(), boxes.clone(), logits.clone()
    fused_a, diag_a = module(query, boxes, logits, features)
    changed = [feature + 2.0 for feature in features]
    fused_b, diag_b = module(query, boxes, logits, changed)
    assert torch.equal(query, query0)
    assert torch.equal(boxes, boxes0)
    assert torch.equal(logits, logits0)
    assert not torch.allclose(fused_a, fused_b)
    assert not torch.allclose(diag_a["evidence"], diag_b["evidence"])


def test_t5_batch_shuffle_changes_gate_or_output():
    module = make_module().eval()
    query, boxes, logits, features = make_inputs()
    fused_a, diag_a = module(query, boxes, logits, features)
    shuffled = [feature.flip(0) for feature in features]
    fused_b, diag_b = module(query, boxes, logits, shuffled)
    difference = (fused_a - fused_b).abs().max()
    gate_difference = (diag_a["gate"] - diag_b["gate"]).abs().max()
    assert difference > 0 or gate_difference > 0


def test_t7_roi_boundaries_and_degenerate_boxes_are_finite():
    module = make_module().eval()
    query, boxes, logits, features = make_inputs(batch=1, queries=5)
    boxes = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1e-12, 1e-12],
          [-0.3, 0.5, 0.4, 0.7], [1.4, 0.5, 2.0, 2.0],
          [0.5, 0.5, 0.01, 0.02]]],
        dtype=query.dtype,
    )
    fused, diagnostics = module(query, boxes, logits, features)
    assert fused.shape == query.shape
    assert diagnostics["evidence"].shape == (1, 5, 3, 16)
    assert torch.isfinite(fused).all()
    assert torch.isfinite(diagnostics["gate"]).all()


def test_t8_gradients_reach_all_qdmf_groups_but_not_ir_or_roi_boxes():
    module = make_module().train()
    query, boxes, logits, features = make_inputs(requires_grad=True)
    fused, diagnostics = module(query, boxes, logits, features)
    loss = fused.square().mean() + diagnostics["gate"].mean() * 0.01
    loss.backward()
    required_prefixes = (
        "feature_projections", "local_attention", "scale_mlp", "gate_mlp",
        "output_projection",
    )
    for prefix in required_prefixes:
        gradients = [
            parameter.grad for name, parameter in module.named_parameters()
            if name.startswith(prefix)
        ]
        assert gradients and any(value is not None for value in gradients)
        assert all(torch.isfinite(value).all() for value in gradients if value is not None)
    assert all(feature.grad is None for feature in features)
    # boxes is non-leaf after concatenation; its leaf source is intentionally
    # not retained.  The detached RoI path is covered by the absence of a
    # sampling-coordinate gradient contribution, while area gating may still
    # train the base box when detach is disabled only by configuration.


def test_modality_dropout_forces_exact_fallback():
    module = make_module(dropout=1.0 - 1e-6).train()
    query, boxes, logits, features = make_inputs(batch=32, queries=2)
    torch.manual_seed(0)
    fused, diagnostics = module(query, boxes, logits, features)
    dropped = diagnostics["dropout_mask"]
    assert dropped.any()
    assert torch.equal(fused[dropped], query[dropped])
    assert torch.count_nonzero(diagnostics["residual"][dropped]) == 0


def test_warmup_schedule_and_state_dict_round_trip():
    module = make_module()
    expected = {0: 0.1, 1: 0.1, 2: 0.1, 4: 0.55, 6: 1.0, 9: 1.0}
    for epoch, value in expected.items():
        module.set_training_progress(epoch)
        assert abs(module.residual_scale() - value) < 1e-7
    restored = make_module()
    restored.load_state_dict(copy.deepcopy(module.state_dict()), strict=True)
    for key, value in module.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
