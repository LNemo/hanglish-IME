from copy import deepcopy

import pytest
import torch

from hanglish.models.candidate_evidence import CandidateEvidenceHead, features, objective, pack, predict


def row(**changes):
    value = dict(id='a', kind='source', context='얼굴 [MASK]', candidates=['낫', '낯'],
                 teacher_scores=[2., 1.], base_scores=[6., 6.], kinds=['lexicon', 'morphology'], expected=['낯'])
    value.update(changes)
    return value


NEUTRAL = {'낫': 4., '낯': 0.}


def test_initial_head_preserves_parent_and_fixed_failures():
    model = CandidateEvidenceHead()
    values = predict(model, [row(), dict(id='b', kind='fixed', fixed_result=None, expected=['몫'])], NEUTRAL)
    assert values[0] == dict(id='a', result='낫', correct=False, delta=[0., 0.])
    assert values[1] == dict(id='b', result=None, correct=False, delta=[])


def test_features_do_not_read_expected_labels():
    assert features(row(), NEUTRAL) == features(row(expected=['낫']), NEUTRAL)


def test_candidate_permutation_equivariance():
    value = row()
    reversed_row = deepcopy(value)
    for key in ('candidates', 'teacher_scores', 'base_scores', 'kinds'):
        reversed_row[key].reverse()
    model = CandidateEvidenceHead()
    with torch.no_grad():
        model.output.weight.fill_(.4)
    a, b = predict(model, [value, reversed_row], NEUTRAL)
    assert a['result'] == b['result']
    assert a['delta'] == list(reversed(b['delta']))


def test_training_gradient_is_finite_and_reaches_head():
    model = CandidateEvidenceHead()
    loss = objective(model, pack([row(), row(id='b', kind='retention', expected=['없음'])], NEUTRAL))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.output.weight.grad).all()
    assert model.output.weight.grad.abs().sum() > 0


def test_restored_weights_reproduce_scores_and_residuals_are_bounded():
    model = CandidateEvidenceHead()
    with torch.no_grad():
        model.output.weight.fill_(1e6)
    restored = CandidateEvidenceHead.restore(model.export())
    a, b = predict(model, [row()], NEUTRAL), predict(restored, [row()], NEUTRAL)
    assert a == b
    assert all(-2 <= v <= 2 for v in a[0]['delta'])


def test_invalid_state_and_candidate_metadata_rejected():
    model = CandidateEvidenceHead()
    bad = model.export()
    bad['output.bias'] = [float('nan')]
    with pytest.raises(ValueError, match='tensor'):
        CandidateEvidenceHead.restore(bad)
    with pytest.raises(ValueError, match='lengths'):
        features(row(kinds=['lexicon']), NEUTRAL)
