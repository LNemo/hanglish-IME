import pytest
from scripts.probe_pko_source_conditioning import sample_pairs, summarize


def test_selection_ignores_targets_and_input_order():
    rows = [{'id': str(i), 'source': 'a'*i+str(i), 'target': 'target'} for i in range(40)]
    groups = {r['id']: r['id'] for r in rows}
    selected = sample_pairs(rows, groups, per_bin=4)
    assert len(selected) == 16
    assert selected == sample_pairs([{**r, 'target': 'changed'} for r in reversed(rows)], groups, per_bin=4)
    assert all(r['source'] != r['shuffled_source'] and r['id'] != r['donor_id'] for r in selected)
    for bin_index in range(4):
        bucket = [r for r in selected if r['bin'] == bin_index]
        assert {r['id'] for r in bucket} == {r['donor_id'] for r in bucket}


def test_insufficient_groups_fail():
    rows = [{'id': str(i), 'source': str(i)} for i in range(40)]
    with pytest.raises(ValueError, match='insufficient'):
        sample_pairs(rows, {r['id']: 'same' for r in rows}, per_bin=4)


def test_source_signal_and_no_signal_are_distinct():
    rows = [{'normal_nll': 1., 'shuffled_nll': 2., 'target_tokens': i+1} for i in range(8)]
    assert summarize(rows)['source_signal_gate'] is True
    assert summarize(rows)['token_weighted_delta'] == 1.
    assert summarize([{**r, 'shuffled_nll': 1.} for r in rows])['source_signal_gate'] is False


def test_bad_loss_fails_instead_of_silent_denominator_loss():
    with pytest.raises(ValueError, match='NLL'):
        summarize([{'normal_nll': float('nan'), 'shuffled_nll': 2., 'target_tokens': 1}])
