import pytest
import torch

from scripts.calibrate_context_evidence import corrected_scores, has_context, fit, objective, pack, predict


def row(**changes):
    value = dict(id='a', group_id='doc-a', kind='source', context='얼굴을 이르는 말 [MASK]',
                 candidates=['낫', '낯'], teacher_scores=[2., 1.], base_scores=[6., 6.], expected=['낯'])
    value.update(changes)
    return value


def test_zero_coefficient_preserves_exact_baseline():
    assert corrected_scores(row(), {'낫': 4., '낯': 0.}, 0.) == [2.9, 1.9]


def test_empty_context_does_not_remove_word_preference():
    value = row(context='  [MASK]  ')
    assert corrected_scores(value, {'낫': 4., '낯': 0.}, 1.) == [2.9, 1.9]
    assert not has_context(' \n ')
    assert has_context('literal [MASK] [MASK]')


def test_training_learns_to_remove_a_misleading_candidate_prior():
    result = fit([row()], {'낫': 4., '낯': 0.})
    assert .25 < result['alpha'] <= 1
    assert result['final_loss'] < result['initial_loss']
    assert predict([row()], {'낫': 4., '낯': 0.}, result['alpha'])[0]['correct']


def test_missing_gold_retention_has_finite_loss_and_gradient():
    rows = [row(), row(id='b', kind='retention', candidates=['목'], teacher_scores=[1.],
                       base_scores=[6.], expected=['몫'])]
    alpha = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    loss = objective(alpha, pack(rows, {'낫': 4., '낯': 0., '목': 1.}))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(alpha.grad)


def test_noninformative_source_rejected():
    with pytest.raises(ValueError, match='informative'):
        pack([row(expected=['없는후보'])], {'낫': 4., '낯': 0.})


def test_fixed_failure_and_tie_break_are_preserved():
    rows = [dict(id='fixed', kind='fixed', fixed_result=None, expected=['몫']),
            row(candidates=['낯', '낫'], teacher_scores=[1., 1.], expected=['낫'])]
    result = predict(rows, {'낫': 0., '낯': 0.}, .5)
    assert result[0] == dict(id='fixed', result=None, correct=False)
    assert result[1] == dict(id='a', result='낫', correct=True)
