import json

import pytest
import torch

from scripts.train_character_edge_full_fit import (
    EXPECTED_BEFORE,
    authenticate_inputs,
    build_row,
    iter_captures,
    listwise_loss,
    row_metrics,
    sequence_digest,
)


CALIBRATION = {'parameters': [1., 0., 0., 0., 0.],
               'scale': [1., 1., 1., 1.]}


def segment(source, text):
    return [{'input_start': 0, 'input_end': len(source), 'source': source,
             'text': text, 'kind': 'test'}]


def scored_candidate(source, text, score, status):
    return {'text': text, 'segments': segment(source, text), 'status': status,
            'sentence_score': score, 'features': [0., 0., 0., 0.]}


def test_ranked_row_keeps_partial_gold_and_unsupported_in_denominators():
    source = 'ga'
    scored = {
        'request': {'input': source, 'context': ''}, 'ranked': True,
        'fallback_public': None,
        'original': [scored_candidate(source, '가', 2., 'preserved')],
        'novel': [
            scored_candidate(source, '카', 1., 'scored'),
            {**scored_candidate(source, '까', 0., 'unsupported'),
             'sentence_score': None, 'reason': 'unsupported'},
        ],
    }
    label = {'id': 'row', 'input': source, 'expected': ['카', 'missing'],
             'component_id': 'component'}
    row = build_row('row', scored, label, CALIBRATION)
    assert [candidate['text'] for candidate in row['candidates']] == ['가', '카']
    assert row['base_scores'] == [2., 1.]
    assert row['positive_indices'] == [1]
    assert row['unsupported'] == 1
    metric = row_metrics(row, torch.tensor([2., 3.], dtype=torch.float64))
    assert metric == {'id': 'row', 'first': True, 'top5': True,
                      'pool_covered': True, 'public': ['카', '가', source]}


def test_missing_gold_and_no_pool_rows_remain_in_public_metrics():
    source = 'raw'
    ranked = {
        'request': {'input': source, 'context': ''}, 'ranked': True,
        'fallback_public': None,
        'original': [scored_candidate(source, '롸', 2., 'preserved')], 'novel': [],
    }
    label = {'id': 'missing', 'input': source, 'expected': ['정답'],
             'component_id': 'c'}
    missing = build_row('missing', ranked, label, CALIBRATION)
    assert missing['positive_indices'] == [] and not missing['no_pool']
    assert row_metrics(missing, torch.tensor([2.]))['pool_covered'] is False
    with pytest.raises(ValueError, match='gold-present'):
        listwise_loss(torch.tensor([2.]), [])

    fallback = [{'text': source, 'segments': segment(source, source), 'score': None}]
    empty = {'request': {'input': source, 'context': ''}, 'ranked': False,
             'fallback_public': fallback, 'original': [], 'novel': []}
    no_pool = build_row('missing', empty, label, CALIBRATION)
    assert no_pool['no_pool'] and no_pool['candidates'] == []
    assert row_metrics(no_pool, []) == {
        'id': 'missing', 'first': False, 'top5': False,
        'pool_covered': False, 'public': [source]}


def test_capture_iteration_is_lazy_and_preserves_shard_then_file_order(tmp_path):
    directories = [tmp_path / 'zero', tmp_path / 'one']
    for directory in directories:
        directory.mkdir()
    first = {'id': 'a', 'scoring': {'value': 1}}
    (directories[0] / 'capture.jsonl').write_text(json.dumps(first) + '\n')
    (directories[1] / 'capture.jsonl').write_text(
        json.dumps({'id': 'b', 'scoring': {'value': 2}}) + '\n')
    shards = [{'shard': index, 'directory': str(directory)}
              for index, directory in enumerate(directories)]
    rows = iter_captures(shards)
    assert next(rows) == ('a', {'value': 1})
    # The second shard is not opened or retained before the generator reaches it.
    (directories[1] / 'capture.jsonl').write_text(
        json.dumps({'id': 'changed', 'scoring': {'value': 3}}) + '\n')
    assert list(rows) == [('changed', {'value': 3})]


def test_frozen_full_fit_closures_and_canonical_denominators_authenticate():
    authenticated = authenticate_inputs()
    report = authenticated['report']
    assert len(authenticated['labels']) == len(report['ids']) == EXPECTED_BEFORE['rows']
    assert authenticated['ordered_ids_sha256'] == sequence_digest(report['ids'])
    assert report['summary']['branches']['expanded_calibrated'] == EXPECTED_BEFORE
    assert report['summary']['candidate_counts']['unsupported_novel'] == 31
    assert len(authenticated['shards']) == 6
    # Ten top-level/code/pilot pins plus seven authenticated files per shard.
    assert len(authenticated['pins']) == 10 + 6 * 7
