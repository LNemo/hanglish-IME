from copy import deepcopy

import pytest

from scripts.current_fit_recombination_coverage import compare_row, summarize


def fixture(old=None, expected=None, new=None):
    old = ['가 다'] if old is None else old
    expected = ['나 라'] if expected is None else expected
    new = ['나 라'] if new is None else new
    fallback = [{'text': 'roman'}]
    reference = dict(id='fit-1', component_id='component-1', input='roman',
                     candidates=old, expected=expected, ranked=bool(old), source_shard=0,
                     public_candidates=['roman'])
    capture = dict(id='fit-1', component_id='component-1',
                   request={'input': 'roman', 'context': ''},
                   source_reference={'ranked': bool(old), 'pool_rows': len(old), 'source_shard': 0},
                   status='generated' if old else 'no_pool',
                   proposals=[{'text': value} for value in new], fallback_public=fallback)
    return capture, reference


def test_missing_positive_becomes_covered_without_claiming_first_choice():
    capture, reference = fixture()
    before = deepcopy((capture, reference))
    row = compare_row(capture, reference)
    assert row['newly_covered'] and row['expanded_pool_covered']
    assert not row['old_pool_covered']
    assert 'top1' not in summarize([row])
    assert (capture, reference) == before


def test_already_covered_row_is_not_counted_as_correction():
    capture, reference = fixture(expected=['가 다', '나 라'])
    row = compare_row(capture, reference)
    assert row['old_pool_covered'] and row['added_pool_covered']
    assert not row['newly_covered']


def test_no_pool_retains_denominator_and_separate_fallback_coverage():
    capture, reference = fixture(old=[], expected=['roman'], new=[])
    result = summarize([compare_row(capture, reference)])
    assert result['rows'] == result['unranked'] == result['fallback_public_covered'] == 1
    assert result['expanded_pool_covered'] == 0


@pytest.mark.parametrize('change', [
    lambda c: c['request'].update(context='external hint'),
    lambda c: c.update(id='another-row'),
    lambda c: c['proposals'].append({'text': '나 라'}),
    lambda c: c['proposals'].append({'text': '가 다'}),
    lambda c: c['source_reference'].update(pool_rows=99),
])
def test_rejects_identity_and_denominator_corruption(change):
    capture, reference = fixture()
    change(capture)
    with pytest.raises(ValueError):
        compare_row(capture, reference)


def test_rejects_changed_unranked_fallback():
    capture, reference = fixture(old=[], new=[])
    capture['fallback_public'] = [{'text': 'silently changed'}]
    with pytest.raises(ValueError):
        compare_row(capture, reference)
