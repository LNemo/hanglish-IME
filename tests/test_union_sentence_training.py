from dataclasses import replace

import pytest
import torch

from scripts.train_union_sentence_ranker import (
    UnionExample, build_example, evaluate_pool, ranking_loss, teacher_choice,
)


def row(**changes):
    value = {"id": "train-1", "input": "a b", "context": "문맥",
             "candidates": ["정답", "허용 정답", "오답", "다른 오답"],
             "parent_candidates": ["정답", "허용 정답"],
             "positive_indices": [0, 1], "expected": ["정답", "허용 정답"],
             "teacher_scores": [1., 0., 4., 3.], "sentence_eligible": True}
    value.update(changes)
    return value


def test_positives_preserved_and_hardest_negative_selected():
    example = build_example(row(), limit=3)
    assert example.candidates == ("정답", "허용 정답", "오답")
    assert example.positives == (0, 1)
    assert example.teacher_scores == (1., 0., 4.)
    assert example.distill_teacher is False
    # Even if capping excludes every negative, a wrong full-pool teacher isn't distilled.
    assert build_example(row(), limit=2).distill_teacher is False
    assert build_example(row(), retention=True).distill_teacher is True


def test_teacher_correct_natural_is_distilled_and_large_positive_set_rejected():
    assert build_example(row(teacher_scores=[5., 0., 4., 3.])).distill_teacher is True
    with pytest.raises(ValueError, match="positive set"):
        build_example(row(), limit=1)


def test_uncovered_and_fallback_rows_excluded_from_training():
    assert build_example(row(positive_indices=[])) is None
    assert build_example(row(sentence_eligible=False, teacher_scores=[])) is None


def test_loss_sums_accepted_probability_without_word_bias():
    example = UnionExample("", ("a", "b", "c"), (0, 1), (0., 0., 0.), "id", False)
    scores = torch.tensor([.2, .7, 1.1], requires_grad=True)
    expected = -torch.log(torch.softmax(scores, dim=0)[:2].sum())
    actual = ranking_loss(scores, [example])
    assert torch.allclose(actual, expected)
    actual.backward()
    assert scores.grad[0] < 0 and scores.grad[1] < 0 and scores.grad[2] > 0
    teacher = replace(example, teacher_scores=tuple(scores.detach().tolist()), distill_teacher=True)
    assert torch.allclose(ranking_loss(scores, [teacher]), expected, atol=1e-6)
    with pytest.raises(ValueError, match="every logit"):
        ranking_loss(torch.tensor([0., 0., 0., 1.]), [example])


def test_parent_prefix_baseline_and_no_label_driven_choice():
    case = row()
    assert teacher_choice(case, parent_only=True) == 0
    assert teacher_choice(case) == 2
    assert evaluate_pool([case], parent_only=True)["correct"] == 1
    assert evaluate_pool([case])["correct"] == 0
    changed_label = row(expected=["허용 정답"])
    assert evaluate_pool([changed_label], parent_only=True)["outcomes"][0]["result"] == "정답"


def test_fallback_does_not_call_sentence_ranker():
    class ForbiddenRanker:
        def score_pairs(self, *_):
            raise AssertionError("fallback must not score")
    case = row(sentence_eligible=False, teacher_scores=[])
    result = evaluate_pool([case], ForbiddenRanker())
    assert result["rows"] == 1 and result["fallbacks"] == 1
    assert result["outcomes"][0]["result"] == "정답"


def test_evaluation_uses_text_tie_order():
    case = row(candidates=["나", "가"], parent_candidates=["나"],
               positive_indices=[1], expected=["가"], teacher_scores=[1., 1.])
    assert evaluate_pool([case])["outcomes"][0]["result"] == "가"
