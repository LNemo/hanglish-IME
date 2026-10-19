import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import difference_pairwise_lora_model as pairwise


@pytest.mark.parametrize('keep_count', [2, 3])
def test_balanced_unit_matches_independent_autograd(keep_count, monkeypatch):
    """Variable control count must not dilute the repair's half of the loss."""
    utils = ModuleType('mlx.utils')
    utils.tree_map = lambda fn, *trees: {key: fn(*(tree[key] for tree in trees))
                                       for key in trees[0]}
    monkeypatch.setitem(sys.modules, 'mlx.utils', utils)
    monkeypatch.setattr(pairwise.shared, 'flat_tree', lambda tree: tree)
    monkeypatch.setattr(pairwise.shared, 'require_matching_gradient_tree',
                        lambda model, gradient: gradient)
    events = []
    mx = SimpleNamespace(eval=lambda *args: events.append('materialized'),
                         all=np.all, isfinite=np.isfinite)
    targets = ['SWITCH'] + ['KEEP'] * keep_count
    observations = [3., *(-float(i + 1) for i in range(keep_count))]
    components = [{'target': target, 'tokens': [value]}
                  for target, value in zip(targets, observations)]

    def gradient(model, tokens, target):
        events.append(target)
        delta = .2 - tokens[0]
        return np.array(delta * delta), {'w': np.array(2 * delta)}

    result = pairwise.materialize_unit(mx, object(), components, gradient,
                                       observe=lambda record: events.append('observed'))
    theta = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    objective = .5 * (theta - observations[0]) ** 2
    objective += .5 * torch.stack([(theta - value) ** 2 for value in observations[1:]]).mean()
    objective.backward()
    assert result['loss'] == pytest.approx(objective.item())
    assert result['gradient']['w'].item() == pytest.approx(theta.grad.item())
    assert sum(record['weight'] for record in result['components']) == pytest.approx(1.)
    assert events[0] == 'SWITCH'
    assert events.count('observed') == len(components)
    for index, event in enumerate(events):
        if event == 'observed':
            assert events[index - 1] == 'materialized'


@pytest.mark.parametrize('targets', [[], ['SWITCH'], ['KEEP'] * 3,
                                    ['SWITCH', 'KEEP'], ['KEEP', 'SWITCH', 'KEEP'],
                                    ['SWITCH'] + ['KEEP'] * 4])
def test_reject_non_recipe_units(targets):
    with pytest.raises(ValueError):
        pairwise.unit_weights(targets)


def test_nonfinite_or_misaligned_reference_losses_fail():
    for losses in ([1., 2.], [1., float('nan'), 3.]):
        with pytest.raises(ValueError):
            pairwise.reference_unit_loss(losses, ['SWITCH', 'KEEP', 'KEEP'])


@pytest.mark.parametrize('count', [3, 4])
def test_optimizer_counts_units_and_checks_state(count, monkeypatch):
    calls = []
    monkeypatch.setattr(pairwise.shared, 'average_and_clip_gradients',
                        lambda mx, gradient, n: (calls.append(n) or gradient, np.array(.7)))
    monkeypatch.setattr(pairwise.shared, 'apply_materialized_update', lambda *args: None)
    monkeypatch.setattr(pairwise.shared, 'flat_tree', lambda tree: tree)
    mx = SimpleNamespace(all=np.all, isfinite=np.isfinite)
    optimizer = SimpleNamespace(state={'m': np.array(0.), 'v': np.array(1.)})
    assert pairwise.apply_unit_update(mx, object(), optimizer, {'w': np.array(1.)}, count) == .7
    assert calls == [count]
    optimizer.state['m'] = np.array(float('inf'))
    with pytest.raises(FloatingPointError):
        pairwise.apply_unit_update(mx, object(), optimizer, {'w': np.array(1.)}, count)


@pytest.mark.parametrize('count', [0, 1, 2, 5, True])
def test_optimizer_rejects_unapproved_unit_counts(count):
    with pytest.raises(ValueError):
        pairwise.apply_unit_update(None, None, None, None, count)
