from dataclasses import replace

import pytest
import torch

from scripts.train_current_word_ranker import (
    RECIPE,
    WeightedExample,
    build_examples,
    build_weighted_example,
    weighted_ranking_loss,
)
from scripts.train_preserved_context_ranker import PreservedExample, ranking_loss


def group(**changes):
    value = {
        "id": "train-1:0:4",
        "context": "주변 [MASK] 문맥",
        "candidates": ["정답", "오답", "다른 오답", "허용 정답"],
        "positive_indices": [0, 3],
        "base_scores": [0., 0., 0., 0.],
        "teacher_scores": [1., 5., 4., 0.],
        "label_aligned": True,
    }
    value.update(changes)
    return value


def rows(groups, origin="natural"):
    return [{"origin": origin, "groups": groups}]


def test_teacher_wrong_natural_gets_weight_three_before_candidate_cap():
    example = build_weighted_example(group(), "natural_train",
                                     {**RECIPE, "candidate_limit": 2})

    assert example.loss_weight == 3.
    assert example.teacher_correct is False
    assert example.example.distill_teacher is False
    assert example.example.candidates == ("정답", "허용 정답")


def test_teacher_correct_natural_and_all_retention_keep_unit_weight_and_kd():
    correct = build_weighted_example(
        group(teacher_scores=[6., 5., 4., 0.]), "natural_train"
    )
    retention = build_weighted_example(group(), "retention_train")

    assert correct.loss_weight == retention.loss_weight == 1.
    assert correct.teacher_correct is True
    assert correct.example.distill_teacher is True
    assert retention.teacher_correct is False
    assert retention.example.distill_teacher is True


def test_unaligned_and_uncovered_groups_are_explicitly_excluded():
    assert build_weighted_example(
        group(label_aligned=False), "natural_train"
    ) is None
    assert build_weighted_example(
        group(positive_indices=[]), "natural_train"
    ) is None
    with pytest.raises(ValueError, match="training roles"):
        build_weighted_example(group(), "natural_dev")


def test_natural_selection_is_bounded_but_retention_uses_every_group():
    natural = [group(id=f"natural:{index}") for index in range(4)]
    retention = [group(id=f"retention:{index}") for index in range(3)]
    recipe = {**RECIPE, "source_group_limit": 2, "selection_salt": "fixed-test"}
    examples, eligibility = build_examples(
        {"natural_train": rows(natural),
         "retention_train": rows(retention, "retention")}, recipe
    )

    assert len(examples["natural_train"]) == 2
    assert len(examples["retention_train"]) == 3
    assert eligibility["natural_train"]["selected_before_labels"] == 2
    assert eligibility["retention_train"]["selected_before_labels"] == 3
    again, _ = build_examples(
        {"natural_train": rows(list(reversed(natural))),
         "retention_train": rows(retention, "retention")}, recipe
    )
    assert [item.example.case_id for item in examples["natural_train"]] == [
        item.example.case_id for item in again["natural_train"]
    ]


def test_weighted_loss_reuses_multi_positive_listwise_kd_math():
    first = PreservedExample("", ("a", "b", "c"), (0, 2),
                             (0., 0., 0.), (0., 1., 0.),
                             "first", "natural", False)
    second = replace(first, candidates=("a", "b"), positives=(0,),
                     base_scores=(0., 0.), teacher_scores=(2., 0.),
                     case_id="second", origin="retention", distill_teacher=True)
    examples = [
        WeightedExample(first, 3., False, "natural_train"),
        WeightedExample(second, 1., True, "retention_train"),
    ]
    scores = torch.tensor([0., 1., 0., 2., 0.], requires_grad=True)
    expected = (3 * ranking_loss(scores[:3], [first], RECIPE)
                + ranking_loss(scores[3:], [second], RECIPE)) / 4

    actual = weighted_ranking_loss(scores, examples)

    assert torch.allclose(actual, expected)
    actual.backward()
    assert scores.grad is not None and bool(torch.isfinite(scores.grad).all())
    with pytest.raises(ValueError, match="every logit"):
        weighted_ranking_loss(torch.zeros(6), examples)


def test_recipe_is_fixed_to_requested_bounded_continuation():
    assert RECIPE["epochs"] == 1
    assert RECIPE["top_layers"] == 2
    assert RECIPE["head_lr"] == 5e-5
    assert RECIPE["encoder_lr"] == 1e-5
    assert RECIPE["source_group_limit"] == 16384
    assert RECIPE["base_weight"] == .15
    assert RECIPE["distillation_weight"] == 1.
    assert RECIPE["temperature"] == 2.
    assert RECIPE["teacher_wrong_natural_weight"] == 3.
    assert RECIPE["threads"] == 2
    assert RECIPE["rss_limit_bytes"] == 3 * 1024**3
    assert RECIPE["training_seconds_limit"] == 1800
