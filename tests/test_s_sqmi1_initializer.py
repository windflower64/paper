import pathlib
import sys

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.sam_query_mask_init import SAMQueryMaskInitializer


def _small_transformer(**kwargs):
    return DFINETransformer(
        num_classes=1,
        hidden_dim=32,
        num_queries=10,
        feat_channels=[16, 32, 64],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=2,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        num_denoising=0,
        reg_max=8,
        **kwargs,
    )


def test_masks_to_boxes_uses_normalized_inclusive_extrema():
    masks = torch.zeros(1, 1, 4, 5, dtype=torch.bool)
    masks[:, :, 1:3, 2:5] = True
    boxes, valid = SAMQueryMaskInitializer.masks_to_boxes(masks)
    expected = torch.tensor([[[0.7, 0.5, 0.6, 0.5]]])
    torch.testing.assert_close(boxes, expected)
    assert valid.item()


def test_empty_mask_falls_back_exactly_to_detector_boxes():
    module = SAMQueryMaskInitializer(
        hidden_dim=8, source_channels=4, mask_dim=4, topk=2
    )
    with torch.no_grad():
        for parameter in module.query_proj.parameters():
            parameter.zero_()
    queries = torch.randn(1, 3, 8)
    source = torch.randn(1, 4, 5, 6)
    base = torch.rand(1, 3, 4).mul(0.8).add(0.1)
    refined, _, _, diagnostics = module(queries, source, base)
    torch.testing.assert_close(refined, base, rtol=0.0, atol=0.0)
    assert not diagnostics["valid_mask"].any()


def test_aux_control_disables_box_intervention_exactly():
    module = SAMQueryMaskInitializer(
        hidden_dim=8,
        source_channels=4,
        mask_dim=4,
        topk=2,
        apply_initialization=False,
    )
    queries = torch.randn(2, 3, 8)
    source = torch.randn(2, 4, 5, 6)
    base = torch.rand(2, 3, 4).mul(0.8).add(0.1)
    refined, query_embeddings, pixel_features, diagnostics = module(
        queries, source, base
    )
    torch.testing.assert_close(refined, base, rtol=0.0, atol=0.0)
    assert query_embeddings.shape == (2, 3, 4)
    assert pixel_features.shape == (2, 4, 5, 6)
    assert diagnostics["mix"].count_nonzero() == 0


def test_only_configured_topk_boxes_can_change():
    module = SAMQueryMaskInitializer(
        hidden_dim=8,
        source_channels=1,
        mask_dim=1,
        topk=2,
        max_mix=1.0,
        max_box_delta=1.0,
        gate_bias=10.0,
    )
    with torch.no_grad():
        for parameter in module.query_proj.parameters():
            parameter.zero_()
        module.query_proj[-1].bias.fill_(10.0)
        module.pixel_proj[0].weight.fill_(1.0)
        module.pixel_proj[1].weight.fill_(1.0)
        module.pixel_proj[1].bias.zero_()
        module.gate[-1].weight.zero_()
        module.gate[-1].bias.fill_(10.0)

    queries = torch.zeros(1, 3, 8)
    source = -torch.ones(1, 1, 6, 6)
    source[:, :, 1:4, 2:5] = 1.0
    base = torch.tensor(
        [[[0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]]]
    )
    refined, _, _, diagnostics = module(queries, source, base)
    assert diagnostics["valid_mask"].all()
    assert not torch.equal(refined[:, :2], base[:, :2])
    torch.testing.assert_close(refined[:, 2:], base[:, 2:], rtol=0.0, atol=0.0)
    refined.sum().backward()
    assert module.gate[-1].weight.grad is not None
    assert module.gate[-1].weight.grad.abs().sum() > 0


def test_mask_supervision_protects_detector_query_but_reaches_rgb_source():
    module = SAMQueryMaskInitializer(
        hidden_dim=8, source_channels=4, mask_dim=4, topk=2
    )
    queries = torch.randn(1, 3, 8, requires_grad=True)
    source = torch.randn(1, 4, 5, 6, requires_grad=True)
    base = torch.rand(1, 3, 4).mul(0.8).add(0.1)
    _, query_embeddings, pixel_features, _ = module(queries, source, base)
    loss = query_embeddings.square().mean() + pixel_features.square().mean()
    loss.backward()
    assert queries.grad is None
    assert source.grad is not None and source.grad.abs().sum() > 0
    assert module.query_proj[-1].weight.grad is not None
    assert module.pixel_proj[0].weight.grad is not None


def test_disabled_transformer_keeps_shared_initialization_and_output_exact():
    torch.manual_seed(20260916)
    baseline = _small_transformer()
    torch.manual_seed(20260916)
    disabled = _small_transformer(sqmi_enabled=False)
    assert baseline.state_dict().keys() == disabled.state_dict().keys()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, disabled.state_dict()[key], rtol=0.0, atol=0.0)

    feats = [
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 32, 4, 4),
        torch.randn(1, 64, 2, 2),
    ]
    baseline.eval()
    disabled.eval()
    with torch.no_grad():
        baseline_output = baseline(feats)
        disabled_output = disabled(feats)
    torch.testing.assert_close(
        baseline_output["pred_logits"], disabled_output["pred_logits"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        baseline_output["pred_boxes"], disabled_output["pred_boxes"], rtol=0.0, atol=0.0
    )


def test_enabled_transformer_runs_inference_without_external_sam():
    model = _small_transformer(
        sqmi_enabled=True,
        sqmi_source_channels=16,
        sqmi_mask_dim=8,
        sqmi_topk=4,
    )
    feats = [
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 32, 4, 4),
        torch.randn(1, 64, 2, 2),
    ]
    model.eval()
    with torch.no_grad():
        output = model(feats, sqmi_source=feats[0])
    assert output["pred_boxes"].shape == (1, 10, 4)
    assert model.last_sqmi_diagnostics is not None
    assert model.last_sqmi_diagnostics["mask_logits"].shape == (1, 4, 8, 8)
