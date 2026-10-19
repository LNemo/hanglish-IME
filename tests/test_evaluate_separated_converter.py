import copy

import pytest

from scripts.evaluate_separated_converter import (
    compare_legacy_response, compare_public, measure, source_rows,
)


def response(texts, pool=None):
    return {
        'result': texts[0],
        'candidates': [{'text': text, 'score': float(5-i), 'segments': []}
                       for i, text in enumerate(texts)],
        'pipeline': {'pool': [{'text': text} for text in (pool or texts)]},
    }


def test_reference_projects_only_source_fields():
    rows = source_rows([{'id': 'a', 'request': {'input': 'ga na', 'context': ''},
                         'proposed': {'result': 'irrelevant'}}], expected_rows=1)
    assert rows == [{'id': 'a', 'input': 'ga na'}]
    with pytest.raises(ValueError, match='context'):
        source_rows([{'id': 'a', 'request': {'input': 'ga na', 'context': '힌트'}}],
                    expected_rows=1)


def test_parity_checks_segments_scores_and_selected_text():
    before = response(['가 나', 'ga na'])
    after = copy.deepcopy(before)
    compare_public(before, after)
    after['candidates'][0]['score'] += 0.001
    with pytest.raises(ValueError, match='parity'):
        compare_public(before, after)
    after = copy.deepcopy(before)
    after['candidates'][0]['segments'] = [{'text': '가 나'}]
    with pytest.raises(ValueError, match='parity'):
        compare_public(before, after)


def test_coverage_gain_is_separate_from_unchanged_first_choice():
    captures = [{'id': 'a', 'request': {'input': 'ga na', 'context': ''},
                 'legacy': response(['오답', 'ga na']),
                 'independent': response(['오답', 'ga na'], ['오답', '정답', 'ga na'])},
                {'id': 'b', 'request': {'input': 'da ra', 'context': ''},
                 'legacy': response(['다른 답']),
                 'independent': response(['다른 답'])}]
    for row in captures:
        row['legacy_pipeline'] = row['legacy'].pop('pipeline')
    labels = [{'id': 'a', 'input': 'ga na', 'expected': ['정답']},
              {'id': 'b', 'input': 'da ra', 'expected': ['다른 답']}]
    result = measure(captures, labels)
    assert result['legacy']['counts']['rows'] == 2
    assert result['legacy']['counts']['top1'] == 1
    assert result['independent']['counts']['top1'] == 1
    assert result['pool_coverage'] == {'legacy': 1, 'independent': 2}
    assert result['pool_gains'] == ['a'] and result['corrections'] == []
    assert result['quality_improvement_gate'] is False
    with pytest.raises(ValueError, match='identities'):
        measure(captures, labels[::-1])


def test_full_legacy_response_allows_only_declared_timing_changes():
    before = {'result': '가 나', 'candidates': [], 'version': 'frozen',
              'engine_ms': 1.0, 'ctc_word': {'generation_ms': 2, 'prediction': 'same'}}
    after = copy.deepcopy(before)
    after['engine_ms'] = 5
    after['ctc_word']['generation_ms'] = 6
    compare_legacy_response(before, after)
    after['ctc_word']['prediction'] = 'different'
    with pytest.raises(ValueError, match='response parity'):
        compare_legacy_response(before, after)
