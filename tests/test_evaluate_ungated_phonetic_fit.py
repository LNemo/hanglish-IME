import pytest
from scripts.evaluate_ungated_phonetic_fit import compare, graph_reachability


def test_unsupported_gold_is_counted_as_available_but_not_supported():
    row = {'id': 'a', 'request': {'input': 'imnida', 'context': ''},
           'fallback_public': [{'text': '임니다'}, {'text': 'imnida'}],
           'proposals': [{'text': '입니다', 'sentence_support': 'unsupported'}]}
    result = compare(row, {'id': 'a', 'input': 'imnida', 'expected': ['입니다']})
    assert not result['original_available'] and result['expanded_available']
    assert result['new_unsupported_gold'] == ['입니다']
    assert not result['new_supported_gold']


def test_gold_already_in_fallback_is_not_a_new_recovery():
    row = {'id': 'a', 'request': {'input': 'x', 'context': ''},
           'fallback_public': [{'text': '가'}], 'proposals': []}
    result = compare(row, {'id': 'a', 'input': 'x', 'expected': ['가', '나']})
    assert result['original_available'] and result['expanded_available']
    assert not result['new_gold']
    with pytest.raises(ValueError, match='identity'):
        compare(row, {'id': 'b', 'input': 'x', 'expected': ['가']})


def test_edge_upper_bound_distinguishes_joint_repair_from_candidate_recovery():
    def candidate(left, right):
        return {'text': left + right, 'segments': [
            {'input_start': 0, 'input_end': 2, 'source': 'aa', 'text': left},
            {'input_start': 2, 'input_end': 4, 'source': 'bb', 'text': right}]}
    row = {'request': {'input': 'aabb'}, 'parents': [candidate('오', '류')],
           'proposals': [candidate('정', '류'), candidate('오', '답')],
           'generated_by_source': {}}
    result = graph_reachability(row, ['정답'])
    assert not result['parent_edge_reachable']
    assert result['expanded_edge_reachable'] and result['all_beams_edge_reachable']
    row['proposals'] = row['proposals'][:1]
    row['generated_by_source'] = {'bb': [('답', -1.)]}
    result = graph_reachability(row, ['정답'])
    assert not result['expanded_edge_reachable'] and result['all_beams_edge_reachable']
