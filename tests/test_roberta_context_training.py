import math

import pytest
import torch

from scripts.train_natural_word_ranker import WordExample
from scripts.train_roberta_context_ranker import RECIPE, ranking_loss


def example(positives=(0,), origin='natural', base=(0., 0.), teacher=(0., 0.)):
    return WordExample('[MASK]', ('표기', '포기'), positives, base, teacher, 'case:0:5', origin)


def test_listwise_loss_preserves_multiple_valid_spellings():
    scores = torch.tensor([1., -2.], requires_grad=True)
    loss = ranking_loss(scores, [example(positives=(0, 1))])
    loss.backward()
    assert float(loss.detach()) == 0.
    assert torch.equal(scores.grad, torch.zeros(2))


def test_training_bias_matches_runtime_and_pushes_wrong_winner_down():
    scores = torch.zeros(2, requires_grad=True)
    loss = ranking_loss(scores, [example(base=(0., 10.))])
    assert float(loss.detach()) == pytest.approx(math.log1p(math.exp(1.5)))
    loss.backward()
    assert scores.grad[0] < 0 < scores.grad[1]


def test_retention_teacher_matches_distribution_with_static_bias():
    scores = torch.tensor([1., -1.], requires_grad=True)
    row = example(origin='retention', base=(4., 8.), teacher=(1., -1.))
    actual = ranking_loss(scores, [row])
    plain = ranking_loss(scores, [row], {**RECIPE, 'distillation_weight': 0.})
    assert float(actual.detach()) == pytest.approx(float(plain.detach()), abs=1e-6)
    changed = ranking_loss(torch.tensor([-1., 1.]), [row])
    changed_plain = ranking_loss(torch.tensor([-1., 1.]), [row], {**RECIPE, 'distillation_weight': 0.})
    assert changed > changed_plain


def test_group_loss_is_mean_without_cross_candidate_leakage():
    scores = torch.tensor([3., -2., -1., 4.])
    first, second = example(), example(positives=(1,))
    together = ranking_loss(scores, [first, second])
    apart = (ranking_loss(scores[:2], [first]) + ranking_loss(scores[2:], [second])) / 2
    assert float(together) == pytest.approx(float(apart))
    with pytest.raises(ValueError, match='consume all logits'):
        ranking_loss(scores, [first])
