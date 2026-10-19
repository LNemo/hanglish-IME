from copy import deepcopy
import json
import math

import pytest

from scripts import probe_difference_pairwise_lora_resources as probe


@pytest.fixture
def token_rows():
    tokens = {length: list(range(length)) for length in (2, 4, 6, 8)}
    return [{'pair_id': f'{index:064x}', 'prompt_tokens': (index % 4 + 1) * 2,
             'tokens': tokens[(index % 4 + 1) * 2],
             'tokens_sha256': probe.digest(tokens[(index % 4 + 1) * 2]),
             'digit_continuation_ids': {'1': 16, '2': 17}} for index in range(1928)]


def test_real_quantile_shapes_and_synthetic_unit_balance(token_rows):
    shapes, units = probe.representatives_and_schedule(token_rows)
    assert [shape['prompt_tokens'] for shape in shapes] == [2, 4, 8, 8]
    assert shapes[0]['pair_id'] == f'{0:064x}'
    assert shapes[1]['pair_id'] == f'{1925:064x}'
    assert shapes[-1]['pair_id'] == f'{1927:064x}'
    assert len({shape['pair_id'] for shape in shapes}) == 4
    assert len(units) == 20
    assert [unit['update'] for unit in units] == [update for update in range(1, 6) for _ in range(4)]
    assert sum(len(unit['components']) == 3 for unit in units) == 4
    assert sum(len(unit['components']) for unit in units) == 76
    for unit in units:
        switch, *keeps = unit['components']
        assert switch['target'] == 'SWITCH' and switch['weight'] == .5
        assert all(component['target'] == 'KEEP' for component in keeps)
        assert sum(component['weight'] for component in keeps) == pytest.approx(.5)
    flat = [component['pair_id'] for unit in units for component in unit['components']]
    assert flat == [shapes[index % 4]['pair_id'] for index in range(76)]


def test_schedule_ignores_any_labels_and_input_order(token_rows):
    expected = probe.representatives_and_schedule(token_rows)
    mutated = list(reversed(deepcopy(token_rows)))
    for row in mutated:
        row.update({'target': 'untrusted', 'expected': 'NOT READ'})
    assert probe.representatives_and_schedule(mutated) == expected


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'token', 'count', 'digit', 'overflow'])
def test_invalid_qualified_token_inventory_fails(token_rows, mutation):
    if mutation == 'missing':
        token_rows.pop()
    elif mutation == 'duplicate':
        token_rows[-1] = deepcopy(token_rows[0])
    elif mutation == 'token':
        token_rows[0]['tokens'] = [99, 99]
    elif mutation == 'count':
        token_rows[0]['prompt_tokens'] += 1
    elif mutation == 'digit':
        token_rows[0]['digit_continuation_ids'] = {'1': 16, '2': 16}
    else:
        token_rows[0].update({'tokens': [1] * 1025, 'prompt_tokens': 1025,
                              'tokens_sha256': probe.digest([1] * 1025)})
    with pytest.raises(ValueError):
        probe.representatives_and_schedule(token_rows)


def test_runtime_estimate_uses_units_and_both_evaluation_sets():
    counts = {'repair_units': 59, 'updates': 15, 'E_mechanism': 481, 'E_train640': 1556}
    timing = {'component_gradient_p95': 2., 'unit_gradient_p95': 10.,
              'optimizer_update_p95': 1., 'pair_inference_p95': 1.,
              'save_reload_finalization_seconds': 20.}
    assert probe.estimate_seconds(counts, timing) == {'seconds': 4293., 'within_7200': True}
    timing['pair_inference_p95'] = 3.
    assert probe.estimate_seconds(counts, timing)['within_7200'] is False
    for bad in (-1., float('nan'), float('inf')):
        timing['pair_inference_p95'] = bad
        with pytest.raises(ValueError):
            probe.estimate_seconds(counts, timing)


def test_logit_parity_rejects_changed_decision_and_excess_error():
    probe.check_logits([0., 1.], [.125, 1.125])
    probe.check_logits([0., 1.], [0., 1.00001], reload=True)
    for left, right, reload in [([0., 1.], [0., 1.126], False),
                                 ([1., 1.01], [1.01, 1.], False),
                                 ([0., 1.], [0., 1.001], True),
                                 ([math.nan, 1.], [0., 1.], False)]:
        with pytest.raises(ValueError):
            probe.check_logits(left, right, reload=reload)


def test_unqualified_data_stops_before_model_import_and_preserves_rejection(tmp_path, monkeypatch):
    def reject():
        raise ValueError('qualified data mismatch')

    monkeypatch.setattr(probe, 'load_qualified_data', reject)
    output = tmp_path / 'probe'
    report = probe.run(output)
    assert report['completed'] is False and report['resource_feasible'] is False
    assert report['failure'] == 'qualified data mismatch'
    assert report['units'] == report['updates'] == []
    assert json.loads((output / 'report.json').read_text()) == report
    with pytest.raises(ValueError, match='already exists'):
        probe.run(output)
