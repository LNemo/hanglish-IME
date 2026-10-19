import copy
import numpy as np
import pytest
from scripts.verify_phonetic_score_scale import check_resources, replay_updates, row_values, scale_for


def example():
    return {'base_scores': [1., -1.], 'features': [[0., 0., 0., 0.], [1., 0., 0., 0.]],
            'positive_indices': [1]}


def first_record():
    p = 1 / (1 + np.exp(-2.))
    gradient = np.array([2*p, -p, 0., 0., 0.])
    norm = np.linalg.norm(gradient)
    clipped = gradient / norm
    parameters = np.array([1., 0., 0., 0., 0.]) - .05 * clipped / (np.abs(clipped) + 1e-8)
    return {'step': 1, 'gradient': gradient.tolist(), 'gradient_norm_before_clip': norm,
            'loss_before_update': np.logaddexp(0, 2.), 'parameters_after_update': parameters.tolist()}


def test_replay_accepts_closed_form_update():
    record = first_record()
    actual = replay_updates([record], [example()], np.ones(4), True, 1)
    np.testing.assert_allclose(actual, record['parameters_after_update'])


@pytest.mark.parametrize('field', ['gradient', 'parameters_after_update', 'loss_before_update'])
def test_replay_rejects_tampered_update(field):
    record = first_record()
    if isinstance(record[field], list):
        record[field][0] += .1
    else:
        record[field] += .1
    with pytest.raises(AssertionError):
        replay_updates([record], [example()], np.ones(4), True, 1)


def test_positive_mass_and_ragged_row_scale():
    row = example()
    row['positive_indices'] = [0, 1]
    assert row_values(row, np.array([1., 0., 0., 0., 0.]), np.ones(4))[0] == pytest.approx(0.)
    rows = [{'features': [[0]*4, [2]*4]}, {'features': [[0]*4, [4]*4, [0]*4, [4]*4]}]
    np.testing.assert_allclose(scale_for(rows, True), np.full(4, np.sqrt(2.5)))


def trace():
    base = {'elapsed_s': 1., 'rss_bytes': 100, 'swap_growth_bytes': 0,
            'operation_elapsed_s': .1, 'operation': 'fit'}
    return [base | {'stage': 'operation_start'}, base | {'stage': 'operation_end'},
            base | {'stage': 'after_completion', 'operation': None, 'operation_elapsed_s': None}]


def test_resource_closure_and_limits():
    assert check_resources(trace(), ['fit'])['operations'] == 1
    broken = copy.deepcopy(trace())
    broken[1]['rss_bytes'] = 2 * 1024**3 + 1
    with pytest.raises(ValueError, match='RSS'):
        check_resources(broken, ['fit'])
    with pytest.raises(ValueError, match='coverage'):
        check_resources(trace(), ['fit', 'missing'])


def test_resource_rejects_unclosed_operation():
    with pytest.raises(ValueError, match='coverage'):
        check_resources([trace()[0], trace()[-1]], ['fit'])
