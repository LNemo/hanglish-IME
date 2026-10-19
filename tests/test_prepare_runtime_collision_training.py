from copy import deepcopy

import pytest

from scripts.prepare_runtime_collision_training import assemble, partition, validate_capture


def source(identity, group, sources):
    return dict(id=identity, group_id=group, source_ids=sources)


def test_partition_closes_shared_sources_and_exposure_transitively():
    rows = [source('a', 'g-a', ['a']), source('b', 'g-b', ['a', 'b']),
            source('c', 'g-c', ['b']), source('d', 'g-d', ['d'])]
    actual = partition(rows, (set(), {'b'}), (set(), set()))
    assert actual == partition(list(reversed(rows)), (set(), {'b'}), (set(), set()))
    assert {actual[k]['role'] for k in 'abc'} == {'fit'}
    assert {actual[k]['component'] for k in 'abc'} == {'g-a'}
    assert all(actual[k]['reason'] == 'previously_exposed_document' for k in 'abc')


@pytest.mark.parametrize('protected', [({'g-a'}, set()), (set(), {'s-a'})])
def test_previous_holdout_rejected(protected):
    with pytest.raises(ValueError, match='previous holdout'):
        partition([source('a', 'g-a', ['s-a'])], (set(), set()), protected)


def fixture():
    target = dict(id='a', group_id='g-a', source_ids=['s-a'], split='train',
                  text='가치', word='가치', word_input='gachi', word_start=0,
                  left_context='높은')
    capture = dict(id='a', input='gachi', context='높은', focus_start=0, focus_end=5,
                   preliminary=[dict(source='gachi', start=0, end=5, text='같이')],
                   groups=[dict(source='gachi', start=0, end=5, context='높은  [MASK] ',
                                candidates=['같이', '가치'], base_scores=[6., 6.],
                                teacher_scores=[1., 2.], kinds=['lexicon', 'lexicon'])],
                   ranked_pieces=[dict(source='gachi', start=0, end=5, options=[
                       dict(text='가치', score=2.9, kind='lexicon'),
                       dict(text='같이', score=1.9, kind='lexicon')])])
    return target, capture


def test_source_label_join_and_quarantine():
    target, capture = fixture()
    pools, outcomes = assemble([capture], {'a': target}, {'a': {'role': 'fit'}}, set())
    assert pools['fit'][0]['expected'] == ['가치']
    assert outcomes[0]['status'] == 'informative'
    pools, outcomes = assemble([capture], {'a': target}, {'a': {'role': 'hold'}}, {'a'})
    assert not pools['fit'] and not pools['hold']
    assert outcomes[0]['status'] == 'quarantined'


def test_unknown_quarantine_id_rejected():
    target, capture = fixture()
    with pytest.raises(ValueError, match='quarantine ID'):
        assemble([capture], {'a': target}, {'a': {'role': 'hold'}}, {'not-in-source'})


@pytest.mark.parametrize('aligned', [True, False])
def test_unscored_hold_stays_in_denominator(aligned):
    target, capture = fixture()
    capture['groups'] = []
    if not aligned:
        capture['ranked_pieces'] = []
    pools, _ = assemble([capture], {'a': target}, {'a': {'role': 'hold'}}, set())
    assert len(pools['hold']) == 1
    assert pools['hold'][0]['kind'] == 'fixed'
    assert pools['hold'][0]['fixed_result'] == ('가치' if aligned else None)


def test_missing_gold_hold_stays_but_cannot_train():
    target, capture = fixture()
    target['word'] = '갗이'
    for role in ('fit', 'hold'):
        pools, outcomes = assemble([capture], {'a': target}, {'a': {'role': role}}, set())
        assert len(pools[role]) == (1 if role == 'hold' else 0)
        assert outcomes[0]['status'] == 'missing_gold'


@pytest.mark.parametrize('mutation', ['context', 'score', 'span', 'duplicate'])
def test_capture_tampering_rejected(mutation):
    target, capture = fixture()
    capture = deepcopy(capture)
    if mutation == 'context':
        capture['groups'][0]['context'] = 'gold-leaking text'
    elif mutation == 'score':
        capture['ranked_pieces'][0]['options'][0]['score'] += 1
    elif mutation == 'span':
        capture['preliminary'][0]['start'] = 1
    else:
        capture['groups'] *= 2
    with pytest.raises(ValueError):
        validate_capture(capture, target)
