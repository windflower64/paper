import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/zoo/dfine/sam_query_evidence_reader.py"
)
SPEC = importlib.util.spec_from_file_location("sqer_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
SAMQueryEvidenceReader = MODULE.SAMQueryEvidenceReader
sqer_shape_loss = MODULE.sqer_shape_loss
_region_distributions = MODULE._region_distributions


def _fixture():
    torch.manual_seed(12)
    reader = SAMQueryEvidenceReader(
        s4_channels=4,
        s8_channels=8,
        query_dim=12,
        num_classes=1,
        topk=3,
        roi_size=8,
        roi_expand=2.0,
        min_roi_width_px=4,
        min_roi_height_px=4,
        image_height=32,
        image_width=48,
    )
    s4 = torch.randn(1, 4, 8, 12)
    s8 = torch.randn(1, 8, 4, 6)
    queries = torch.randn(1, 7, 12)
    logits = torch.tensor([[[3.0], [2.0], [1.0], [0.0], [-1.0], [-2.0], [-3.0]]])
    boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.5]] * 7])
    return reader, s4, s8, queries, logits, boxes


def test_sqer_zero_initialized_and_bypass_are_exact_identity():
    reader, s4, s8, queries, logits, boxes = _fixture()
    result = reader(s4, s8, queries, logits, boxes, return_aux=True)
    assert torch.equal(result["delta_logits"], torch.zeros_like(logits))
    assert result["query_indices"].tolist() == [[0, 1, 2]]
    bypass = reader(s4, s8, queries, logits, boxes, bypass=True)
    assert torch.equal(bypass["delta_logits"], torch.zeros_like(logits))
    assert bypass["query_indices"] is None
    assert torch.isfinite(result["attention"]).all()


def test_sqer_sam_and_box_supervise_same_query_with_distinct_gradients():
    reader, s4, s8, queries, logits, boxes = _fixture()
    result = reader(s4, s8, queries, logits, boxes, return_aux=True)
    target_box = torch.tensor([[0.5, 0.5, 0.25, 0.3]])
    yy, xx = torch.meshgrid(
        torch.arange(32), torch.arange(48), indexing="ij"
    )
    irregular_mask = (
        ((xx - 24.0) / 5.0).square()
        + ((yy - 16.0) / 4.0).square()
        <= 1.0
    ).float()
    targets = [{
        "boxes": target_box,
        "masks": irregular_mask.unsqueeze(0),
        "sam_quality": torch.tensor([1.0]),
    }]
    source_query = result["query_indices"][0, 0].reshape(1)
    matched = [(source_query, torch.tensor([0]))]
    outputs = {
        "sqer_attention": result["attention"],
        "sqer_roi_grid": result["roi_grid"],
        "sqer_query_indices": result["query_indices"],
    }
    sam_loss, sam_stats = sqer_shape_loss(outputs, targets, matched, "sam")
    box_loss, box_stats = sqer_shape_loss(outputs, targets, matched, "box")
    assert torch.isfinite(sam_loss) and torch.isfinite(box_loss)
    assert sam_stats["sqer_supervised_queries"].item() == 1
    assert box_stats["sqer_supervised_queries"].item() == 1
    assert sam_stats["sqer_teacher_map_difference"].item() > 0
    assert sam_stats["sqer_matched_topk_coverage"].item() == 1
    assert torch.equal(
        outputs["sqer_query_indices"], result["query_indices"]
    )
    sam_grad = torch.autograd.grad(
        sam_loss, reader.query_blocks[1].attention.in_proj_weight,
        retain_graph=True,
    )[0]
    box_grad = torch.autograd.grad(
        box_loss, reader.query_blocks[1].attention.in_proj_weight,
    )[0]
    assert torch.isfinite(sam_grad).all() and torch.isfinite(box_grad).all()
    assert (sam_grad - box_grad).abs().max() > 0


def test_sqer_region_mass_rejection_is_finite_for_degenerate_regions():
    full = torch.ones(8, 8)
    distributions, valid, masses = _region_distributions(full)
    assert not bool(valid[0])
    assert torch.isfinite(distributions).all()
    assert torch.isfinite(masses).all()


def test_sqer_boundary_and_sub16_pixel_rois_remain_finite():
    reader, s4, s8, queries, logits, boxes = _fixture()
    boxes[0, 0] = torch.tensor([0.002, 0.002, 0.003, 0.003])
    boxes[0, 1] = torch.tensor([0.998, 0.998, 0.003, 0.003])
    boxes[0, 2] = torch.tensor([0.50, 0.50, 0.020, 0.025])
    result = reader(s4, s8, queries, logits, boxes, return_aux=True)
    assert result["roi_grid"].shape == (1, 3, 8, 8, 2)
    assert torch.isfinite(result["roi_grid"]).all()
    assert torch.isfinite(result["attention"]).all()
    assert torch.isfinite(result["delta_logits"]).all()


def test_sqer_rejected_sam_target_is_skipped_without_nan():
    reader, s4, s8, queries, logits, boxes = _fixture()
    result = reader(s4, s8, queries, logits, boxes, return_aux=True)
    outputs = {
        "sqer_attention": result["attention"],
        "sqer_roi_grid": result["roi_grid"],
        "sqer_query_indices": result["query_indices"],
    }
    targets = [{
        "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.3]]),
        "masks": torch.ones(1, 32, 48),
        "sam_quality": torch.tensor([0.0]),
    }]
    matched = [(result["query_indices"][0, :1], torch.tensor([0]))]
    loss, stats = sqer_shape_loss(outputs, targets, matched, "sam")
    assert loss.item() == 0
    assert stats["sqer_skip_teacher_rejected"].item() == 1
    assert torch.isfinite(loss)


def test_sqer_shape_supervision_accepts_horizontally_flipped_sample():
    reader, s4, s8, queries, logits, boxes = _fixture()
    result = reader(
        s4.flip(-1), s8.flip(-1), queries, logits, boxes, return_aux=True
    )
    yy, xx = torch.meshgrid(
        torch.arange(32), torch.arange(48), indexing="ij"
    )
    original_mask = (
        ((xx - 19.0) / 5.0).square()
        + ((yy - 16.0) / 4.0).square()
        <= 1.0
    ).float()
    flipped_box = torch.tensor([[0.60, 0.50, 0.25, 0.30]])
    targets = [{
        "boxes": flipped_box,
        "masks": original_mask.flip(-1).unsqueeze(0),
        "sam_quality": torch.tensor([1.0]),
    }]
    matched = [(result["query_indices"][0, :1], torch.tensor([0]))]
    outputs = {
        "sqer_attention": result["attention"],
        "sqer_roi_grid": result["roi_grid"],
        "sqer_query_indices": result["query_indices"],
    }
    loss, stats = sqer_shape_loss(outputs, targets, matched, "sam")
    assert torch.isfinite(loss)
    assert stats["sqer_supervised_queries"].item() == 1


def test_sqer2_constant_local_evidence_cannot_change_detector_score():
    reader = SAMQueryEvidenceReader(
        s4_channels=4,
        s8_channels=8,
        query_dim=12,
        num_classes=1,
        topk=2,
        roi_size=8,
        min_roi_width_px=4,
        min_roi_height_px=4,
        image_height=32,
        image_width=48,
        evidence_only=True,
    )
    with torch.no_grad():
        reader.quality_head[-1].weight.normal_()
        reader.quality_head[-1].bias.normal_()
    s4 = torch.ones(1, 4, 32, 48)
    s8 = torch.ones(1, 8, 16, 24)
    queries = torch.randn(1, 4, 12)
    logits = torch.tensor([[[3.0], [2.0], [1.0], [0.0]]])
    boxes = torch.tensor([[[0.5, 0.5, 0.10, 0.10]] * 4])
    result = reader(s4, s8, queries, logits, boxes, return_aux=True)
    assert result["evidence"].abs().max() < 1e-6
    assert result["delta_logits"].abs().max() < 1e-6


def test_sqer2_detection_score_gradient_reaches_query_and_rgb_features():
    reader, s4, s8, queries, logits, boxes = _fixture()
    reader.evidence_only = True
    reader.quality_head = torch.nn.Sequential(
        torch.nn.LayerNorm(128),
        torch.nn.Linear(128, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 1),
    )
    torch.nn.init.normal_(reader.quality_head[-1].weight, std=0.05)
    torch.nn.init.zeros_(reader.quality_head[-1].bias)
    s4.requires_grad_()
    queries.requires_grad_()
    result = reader(s4, s8, queries, logits, boxes, return_aux=True)
    result["delta_logits"].sum().backward()
    assert s4.grad is not None and torch.isfinite(s4.grad).all()
    assert s4.grad.abs().sum() > 0
    assert queries.grad is not None and torch.isfinite(queries.grad).all()
    assert queries.grad.abs().sum() > 0
