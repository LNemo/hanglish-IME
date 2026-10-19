import math

import pytest
import torch

from scripts.train_preserved_context_ranker import (
    PreservedExample,
    RECIPE,
    full_teacher_correct,
    preserved_example,
    ranking_loss,
)


def group(*, positives=(0,), teacher=None, base=None):
    candidates = [f"후보{i}" for i in range(10)]
    return {
        "id": "case:0:5",
        "context": "주변 [MASK] 문맥",
        "candidates": candidates,
        "positive_indices": list(positives),
        "teacher_scores": list(teacher or [0.] * len(candidates)),
        "base_scores": list(base or [0.] * len(candidates)),
    }


def example(*, positives=(0,), teacher=(0., 0.), base=(0., 0.),
            origin="natural", distill=False):
    return PreservedExample(
        "[MASK]", ("표기", "포기"), positives, base, teacher,
        "case:0:5", origin, distill,
    )


def test_teacher_correctness_uses_full_group_before_candidate_cap():
    teacher = [0.] * 10
    teacher[9] = 20.
    source = group(positives=(0,), teacher=teacher)

    assert full_teacher_correct(source) is False
    prepared = preserved_example("natural", source, {**RECIPE, "candidate_limit": 2})

    assert prepared is not None
    assert len(prepared.candidates) == 2
    assert prepared.distill_teacher is False
    assert source["candidates"][9] in prepared.candidates


def test_natural_teacher_wrong_is_supervised_only_but_correct_and_retention_distill():
    wrong = group(positives=(0,), teacher=[0., 4.] + [0.] * 8)
    correct = group(positives=(0,), teacher=[4., 0.] + [0.] * 8)

    assert preserved_example("natural", wrong).distill_teacher is False
    assert preserved_example("natural", correct).distill_teacher is True
    assert preserved_example("retention", wrong).distill_teacher is True


def test_capping_away_wrong_teacher_winner_does_not_make_it_trusted():
    teacher = [0.] * 10
    teacher[9] = 20.
    source = group(positives=tuple(range(8)), teacher=teacher)
    prepared = preserved_example("natural", source)

    assert len(prepared.candidates) == 8
    assert source['candidates'][9] not in prepared.candidates
    assert len(prepared.positives) == 8
    assert prepared.distill_teacher is False


def test_teacher_wrong_natural_loss_excludes_kl_divergence():
    scores = torch.tensor([1., -1.], requires_grad=True)
    supervised = ranking_loss(scores, [example(teacher=(-4., 4.), distill=False)])
    no_kd = ranking_loss(scores, [example(teacher=(-4., 4.), distill=False)],
                         {**RECIPE, "distillation_weight": 0.})
    with_kd = ranking_loss(scores, [example(teacher=(-4., 4.), distill=True)])

    assert float(supervised.detach()) == pytest.approx(float(no_kd.detach()))
    assert with_kd > supervised


def test_multi_positive_listwise_and_correct_teacher_kd_are_finite_and_group_local():
    all_valid = example(positives=(0, 1), distill=False)
    scores = torch.tensor([2., -3.], requires_grad=True)
    plain = ranking_loss(scores, [all_valid])
    plain.backward()
    assert float(plain.detach()) == 0.
    assert torch.equal(scores.grad, torch.zeros(2))

    first = example(positives=(0, 1), teacher=(3., 1.), distill=True)
    second = example(positives=(1,), origin="retention", teacher=(0., 2.), distill=True)
    combined_scores = torch.tensor([0., 0., -1., 1.], requires_grad=True)
    together = ranking_loss(combined_scores, [first, second])
    apart = (ranking_loss(combined_scores[:2], [first])
             + ranking_loss(combined_scores[2:], [second])) / 2
    assert math.isfinite(float(together.detach()))
    assert float(together.detach()) == pytest.approx(float(apart.detach()))
    with pytest.raises(ValueError, match="consume all logits"):
        ranking_loss(combined_scores, [first])


def test_base_bias_is_shared_by_supervision_and_teacher_distribution():
    scores = torch.tensor([0., 0.])
    row = example(base=(0., 10.), teacher=(0., 0.), distill=False)
    loss = ranking_loss(scores, [row])

    assert float(loss.detach()) == pytest.approx(math.log1p(math.exp(1.5)))
