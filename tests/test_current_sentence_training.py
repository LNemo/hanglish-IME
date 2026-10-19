from dataclasses import replace

import pytest
import torch

from scripts.train_current_sentence_ranker import (
    RECIPE,
    WeightedExample,
    build_weighted_example,
    weighted_ranking_loss,
)
from scripts.train_union_sentence_ranker import UnionExample, ranking_loss


def row(**changes):
    value = {
        "id": "train-1", "input": "a b", "context": "문맥",
        "candidates": ["정답", "오답", "다른 오답", "허용 정답"],
        "parent_candidates": ["정답", "오답", "다른 오답", "허용 정답"],
        "positive_indices": [0, 3], "expected": ["정답", "허용 정답"],
        "teacher_scores": [1., 5., 4., 0.], "sentence_eligible": True,
        "groups": [],
    }
    value.update(changes)
    return value


def test_teacher_wrong_natural_group_gets_weight_three_before_candidate_cap():
    example = build_weighted_example(row(), "natural_train", limit=2)

    assert example.loss_weight == 3.
    assert example.teacher_correct is False
    assert example.example.distill_teacher is False
    assert example.example.candidates == ("정답", "허용 정답")


def test_teacher_correct_natural_and_retention_groups_keep_unit_weight():
    correct = build_weighted_example(
        row(teacher_scores=[6., 5., 4., 0.]), "natural_train"
    )
    retention = build_weighted_example(row(), "retention_train")

    assert correct.loss_weight == retention.loss_weight == 1.
    assert correct.teacher_correct is True and correct.example.distill_teacher is True
    assert retention.example.distill_teacher is True


def test_only_training_roles_are_accepted_and_unrankable_rows_are_skipped():
    with pytest.raises(ValueError, match="training roles"):
        build_weighted_example(row(), "natural_dev")
    assert build_weighted_example(row(positive_indices=[]), "natural_train") is None
    assert build_weighted_example(
        row(sentence_eligible=False, teacher_scores=[]), "natural_train"
    ) is None


def test_weighted_loss_is_normalized_by_total_group_weight():
    first = UnionExample("", ("a", "b"), (0,), (0., 0.), "first", False)
    second = replace(first, case_id="second")
    examples = [
        WeightedExample(first, 3., False, "natural_train"),
        WeightedExample(second, 1., True, "retention_train"),
    ]
    scores = torch.tensor([0., 1., 2., 0.], requires_grad=True)
    expected = (3 * ranking_loss(scores[:2], [first])
                + ranking_loss(scores[2:], [second])) / 4

    actual = weighted_ranking_loss(scores, examples)

    assert torch.allclose(actual, expected)
    actual.backward()
    assert scores.grad is not None and bool(torch.isfinite(scores.grad).all())
    with pytest.raises(ValueError, match="every logit"):
        weighted_ranking_loss(torch.zeros(5), examples)


def test_recipe_is_fixed_to_bounded_one_epoch_current_runtime_continuation():
    assert RECIPE["epochs"] == 1
    assert RECIPE["top_layers"] == 2
    assert RECIPE["head_lr"] == 5e-5
    assert RECIPE["encoder_lr"] == 1e-5
    assert RECIPE["distillation_weight"] == 1.
    assert RECIPE["temperature"] == 2.
    assert RECIPE["teacher_wrong_natural_weight"] == 3.
    assert RECIPE["threads"] == 2
    assert RECIPE["rss_limit_bytes"] == 3 * 1024**3
    assert RECIPE["training_seconds_limit"] == 1800
