import math

import pytest
import torch

from scripts.preflight_source_span_residual import select_cohort, train_loss


def eligible():
    return [dict(id=f'row{i}', input='a' * (i + 1), component_id=f'c{i}',
                 multiple_paths=i < 3, unresolved_ascii=3 <= i < 9) for i in range(25)]


def test_cohort_is_deterministic_stratified_and_component_disjoint():
    rows = eligible()
    rows.append(dict(rows[0], id='same-component'))
    selected = select_cohort(rows)
    assert selected == select_cohort(list(reversed(rows)))
    assert len(selected) == len({r['component_id'] for r in selected}) == 16
    assert [r['stratum'] for r in selected] == ['multiple_paths'] * 2 + ['unresolved_ascii'] * 4 + ['long_source'] * 2 + ['hash'] * 8


def test_empty_specialized_strata_fill_by_hash_without_changing_size():
    rows = eligible()
    for row in rows:
        row.update(multiple_paths=False, unresolved_ascii=False)
    selected = select_cohort(rows)
    assert len(selected) == 16
    assert sum(r['stratum'] == 'hash_fill' for r in selected) == 6
    with pytest.raises(ValueError, match='distinct FIT'):
        select_cohort(rows[:10])


def test_loss_credits_all_allowed_targets_and_has_finite_gradients():
    scores = torch.tensor([0., 0., 0.], requires_grad=True)
    loss = train_loss(scores, [0, 2])
    assert math.isclose(loss.item(), math.log(3 / 2), abs_tol=1e-6)
    loss.backward()
    assert scores.grad[0] < 0 and scores.grad[2] < 0 and scores.grad[1] > 0
