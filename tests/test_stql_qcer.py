import importlib.util
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "src/zoo/dfine/stql_qcer.py"
SPEC = importlib.util.spec_from_file_location("stql_qcer_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
QueryConditionedEvidenceReader = MODULE.QueryConditionedEvidenceReader
SharedQueryProjection = MODULE.SharedQueryProjection
STQLPixelProjection = MODULE.STQLPixelProjection
qcer_objectness_loss = MODULE.qcer_objectness_loss
stql_shape_loss = MODULE.stql_shape_loss


def test_qcer_zero_initialization_and_missing_ir():
    torch.manual_seed(0)
    query_projection = SharedQueryProjection(16, 8)
    reader = QueryConditionedEvidenceReader([12, 12], 8, 1, topk=5)
    queries = query_projection(torch.randn(2, 7, 16))
    thermal = [torch.randn(2, 12, 4, 5), torch.randn(2, 12, 2, 3)]

    initial = reader(queries, thermal)
    assert torch.equal(initial["delta_logits"], torch.zeros_like(initial["delta_logits"]))

    with torch.no_grad():
        reader.output.weight.fill_(0.25)
    available = reader(
        queries,
        thermal,
        availability=torch.tensor([True, False]),
    )
    assert available["delta_logits"][0].abs().sum() > 0
    assert torch.equal(
        available["delta_logits"][1],
        torch.zeros_like(available["delta_logits"][1]),
    )
    bypass = reader(queries, thermal, bypass=True)
    assert torch.equal(bypass["delta_logits"], torch.zeros_like(bypass["delta_logits"]))


def test_stql_loss_is_finite_and_reaches_shared_query_and_s8():
    torch.manual_seed(1)
    query_projection = SharedQueryProjection(16, 8)
    pixel_projection = STQLPixelProjection(10, 8)
    raw_queries = torch.randn(1, 5, 16, requires_grad=True)
    raw_s8 = torch.randn(1, 10, 8, 10, requires_grad=True)
    query_embeddings = query_projection(raw_queries)
    pixel_features = pixel_projection(raw_s8)
    mask = torch.zeros(1, 64, 80)
    mask[:, 24:40, 30:50] = 1
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
            "masks": mask,
            "sam_quality": torch.tensor([1.0]),
        }
    ]
    indices = [(torch.tensor([2]), torch.tensor([0]))]
    loss, stats = stql_shape_loss(
        query_embeddings,
        pixel_features,
        targets,
        indices,
        supervision="sam",
    )
    assert torch.isfinite(loss)
    assert stats["valid_instances"].item() == 1
    loss.backward()
    assert raw_queries.grad is not None and raw_queries.grad.abs().sum() > 0
    assert raw_s8.grad is not None and raw_s8.grad.abs().sum() > 0
    assert query_projection.proj.weight.grad is not None
    assert pixel_projection.proj.weight.grad is not None


def test_qcer_objectness_tiny_box_fallback_is_counted():
    logits = torch.zeros(1, 8, requires_grad=True)
    valid = torch.ones(1, 8, dtype=torch.bool)
    centers = torch.tensor(
        [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
            [0.125, 0.125],
            [0.375, 0.125],
            [0.125, 0.375],
            [0.375, 0.375],
        ]
    )
    levels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    targets = [
        {
            "infrared_boxes": torch.tensor([[0.51, 0.51, 0.01, 0.01]]),
            "infrared_label_known": torch.tensor([True]),
        }
    ]
    loss, stats = qcer_objectness_loss(logits, valid, centers, levels, targets)
    assert torch.isfinite(loss)
    assert stats["fallback_tokens"].item() == 2
    assert stats["positive_tokens"].item() == 2
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_qcer_uniform_attention_still_uses_objectness_topk():
    torch.manual_seed(2)
    reader = QueryConditionedEvidenceReader(
        [6], 4, 1, topk=3, uniform_attention=True
    )
    queries = torch.nn.functional.normalize(torch.randn(1, 2, 4), dim=-1)
    result = reader(queries, [torch.randn(1, 6, 2, 3)])
    attention = result["attention"]
    assert attention.shape == (1, 2, 3)
    assert torch.allclose(attention, torch.full_like(attention, 1 / 3))
    assert result["selected_indices"].shape == (1, 3)


def test_qcer_upstream_gradients_after_output_unfreezes():
    torch.manual_seed(3)
    query_projection = SharedQueryProjection(9, 6)
    reader = QueryConditionedEvidenceReader([7], 6, 1, topk=4)
    with torch.no_grad():
        reader.output.weight.normal_(std=0.02)
    raw_queries = torch.randn(2, 5, 9, requires_grad=True)
    thermal = torch.randn(2, 7, 3, 4, requires_grad=True)
    result = reader(query_projection(raw_queries), [thermal])
    loss = result["delta_logits"].square().mean()
    loss.backward()
    assert raw_queries.grad is not None and raw_queries.grad.abs().sum() > 0
    assert thermal.grad is not None and thermal.grad.abs().sum() > 0
    assert query_projection.proj.weight.grad is not None
    assert reader.thermal_proj[0].weight.grad is not None
    assert reader.key_proj.weight.grad is not None
    assert reader.value_proj.weight.grad is not None
    assert reader.gate[0].weight.grad is not None


def test_qcer_all_padding_and_single_candidate_are_finite():
    torch.manual_seed(4)
    reader = QueryConditionedEvidenceReader([5], 4, 1, topk=8)
    with torch.no_grad():
        reader.output.weight.fill_(0.1)
    queries = torch.nn.functional.normalize(torch.randn(2, 3, 4), dim=-1)
    thermal = [torch.randn(2, 5, 1, 1)]
    result = reader(
        queries,
        thermal,
        valid_ir=torch.tensor([[False], [True]]),
    )
    assert all(torch.isfinite(value).all() for value in (
        result["delta_logits"], result["attention"], result["attention_entropy"]
    ))
    assert torch.equal(result["attention"][0], torch.zeros_like(result["attention"][0]))
    assert torch.equal(
        result["attention_entropy"], torch.zeros_like(result["attention_entropy"])
    )


def test_empty_unaccepted_and_multiple_instance_targets_are_finite():
    torch.manual_seed(5)
    queries = torch.nn.functional.normalize(torch.randn(2, 6, 4), dim=-1)
    pixels = torch.nn.functional.normalize(torch.randn(2, 4, 8, 10), dim=1)
    masks = torch.zeros(2, 64, 80)
    masks[0, 12:28, 10:30] = 1
    masks[1, 36:54, 48:70] = 1
    targets = [
        {
            "boxes": torch.zeros((0, 4)),
            "masks": torch.zeros((0, 64, 80)),
            "sam_quality": torch.zeros(0),
        },
        {
            "boxes": torch.tensor(
                [[0.25, 0.31, 0.25, 0.25], [0.74, 0.70, 0.28, 0.28]]
            ),
            "masks": masks,
            # The second instance is intentionally not accepted.
            "sam_quality": torch.tensor([1.0, 0.0]),
        },
    ]
    indices = [
        (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)),
        (torch.tensor([1, 4]), torch.tensor([0, 1])),
    ]
    loss, stats = stql_shape_loss(queries, pixels, targets, indices, supervision="sam")
    assert torch.isfinite(loss)
    assert stats["valid_instances"].item() == 1

    objectness_logits = torch.zeros(2, 5, requires_grad=True)
    valid_ir = torch.ones(2, 5, dtype=torch.bool)
    centers = torch.tensor(
        [[0.1, 0.1], [0.3, 0.3], [0.5, 0.5], [0.7, 0.7], [0.9, 0.9]]
    )
    level_ids = torch.zeros(5, dtype=torch.long)
    ir_targets = [
        {
            "infrared_boxes": torch.zeros((0, 4)),
            "infrared_label_known": torch.tensor([True]),
        },
        {
            "infrared_boxes": torch.tensor(
                [[0.3, 0.3, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]]
            ),
            "infrared_label_known": torch.tensor([True]),
        },
    ]
    objectness_loss, objectness_stats = qcer_objectness_loss(
        objectness_logits, valid_ir, centers, level_ids, ir_targets
    )
    assert torch.isfinite(objectness_loss)
    assert objectness_stats["known_images"].item() == 2
    assert objectness_stats["positive_tokens"].item() == 2
    objectness_loss.backward()
    assert torch.isfinite(objectness_logits.grad).all()


if __name__ == "__main__":
    tests = [
        test_qcer_zero_initialization_and_missing_ir,
        test_stql_loss_is_finite_and_reaches_shared_query_and_s8,
        test_qcer_objectness_tiny_box_fallback_is_counted,
        test_qcer_uniform_attention_still_uses_objectness_topk,
        test_qcer_upstream_gradients_after_output_unfreezes,
        test_qcer_all_padding_and_single_candidate_are_finite,
        test_empty_unaccepted_and_multiple_instance_targets_are_finite,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
