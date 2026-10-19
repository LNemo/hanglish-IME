from copy import deepcopy

import pytest

from scripts.capture_ungated_mixed_runtime import compare_runtime


def fixture():
    raw = {'text': '3wi'}
    parent = {'id': 'fit', 'request': {'input': '3wi', 'context': ''},
              'fallback_public': [raw], 'parents': [], 'proposals': [], 'generated_by_source': {}}
    new = {'text': '3위', 'segments': [{'input_start': 0, 'input_end': 3, 'text': '3위'}]}
    scored = {'id': 'fit', 'request': parent['request'],
              'supported_scores': [{'text': '3위', 'score': 1.}],
              'arms': {'mixed_union': {'pool': [{'text': '3위', 'support': 'supported'},
                                               {'text': '긴후보', 'support': 'unsupported'}],
                                        'public': [{'text': '3위'}, raw]}}}
    trace = {'baseline': {'sentence_ranking_applied': False, 'candidates': [raw]},
             'joint_trace': {'generated': {'generated_by_source': {}, 'parents': [], 'proposals': []}, 'pool': []},
             'mixed_proposals': [new],
             'union_pool': [{'text': '3위', 'supported': True, 'score': 1.},
                            {'text': '긴후보', 'supported': False, 'score': None}]}
    response = {'input': '3wi', 'context': '', 'external_context_used': False,
                'candidates': [{'text': '3위'}, raw]}
    return parent, scored, {'proposals': [deepcopy(new)]}, response, trace


def test_runtime_comparison_accepts_complete_match():
    assert compare_runtime(*fixture())['public_exact']


def test_runtime_rejects_changed_generated_source_path():
    values = fixture()
    values[-1]['mixed_proposals'][0]['segments'][0]['input_end'] = 2
    with pytest.raises(ValueError, match='mixed paths'):
        compare_runtime(*values)


def test_runtime_rejects_scored_unsupported_candidate():
    values = fixture()
    values[-1]['union_pool'][-1]['score'] = 0.
    with pytest.raises(ValueError, match='unsupported scored'):
        compare_runtime(*values)


def test_runtime_rejects_public_order_change():
    values = fixture()
    values[-2]['candidates'].reverse()
    with pytest.raises(ValueError, match='public order'):
        compare_runtime(*values)


def test_runtime_checks_actual_scoring_calls_exclude_unsupported():
    assert compare_runtime(*fixture(), score_calls=[{'contexts': [''], 'texts': ['3위']}])['public_exact']
    with pytest.raises(ValueError, match='runtime scored other'):
        compare_runtime(*fixture(), score_calls=[{'contexts': ['', ''], 'texts': ['3위', '긴후보']}])


def test_runtime_rejects_changed_raw_reserve_segments():
    values = fixture()
    values[-2]['candidates'][-1] = {'text': '3wi', 'segments': []}
    with pytest.raises(ValueError, match='raw reserve changed'):
        compare_runtime(*values)
