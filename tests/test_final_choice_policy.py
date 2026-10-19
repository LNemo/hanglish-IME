import copy

import pytest

from scripts.final_choice_policy import (
    decision_features, finite_candidate_indices, proposal_allowed,
    reorder_candidates,
)


def candidate(text, score=1.0, source='imnida', kind='neural'):
    return {'text': text, 'score': score, 'segments': [{
        'input_start': 0, 'input_end': len(source), 'source': source,
        'text': text, 'kind': kind,
    }]}


def test_valid_correction_does_not_require_wrong_parent_to_roundtrip():
    assert proposal_allowed('imnida', candidate('입이다'), candidate('입니다'))
    assert not proposal_allowed('imnida', candidate('입니다'), candidate('입이다'))


@pytest.mark.parametrize('kind', ['raw', 'mixed', 'numeric', 'unknown'])
def test_protected_parent_or_proposal_is_never_changed(kind):
    assert not proposal_allowed('imnida', candidate('임니다', kind=kind), candidate('입니다'))
    assert not proposal_allowed('imnida', candidate('임니다'), candidate('입니다', kind=kind))


def test_segment_alignment_and_complete_source_are_required():
    parent, proposed = candidate('임니다'), candidate('입니다')
    bad = copy.deepcopy(proposed)
    bad['segments'][0]['input_end'] -= 1
    assert not proposal_allowed('imnida', parent, bad)
    bad = copy.deepcopy(proposed)
    bad['text'] = '불일치'
    assert not proposal_allowed('imnida', parent, bad)


def test_none_candidate_is_excluded_from_six_reproducible_features():
    candidates = [candidate('임니다', 2.0), candidate('입니다', 1.0),
                  candidate('imnida', None, kind='raw')]
    result = decision_features('imnida', ' 문맥 ', candidates, [0.1, 0.9, 100.0])
    assert result['proposal_index'] == 1
    assert result['finite_indices'] == [0, 1]
    assert result['features'] == pytest.approx([1.0, -1.0, 0.8, 0.8, 1.0, 2/4096])
    assert finite_candidate_indices(candidates) == [0, 1]


def test_raw_missing_parent_and_ties_keep_parent():
    candidates = [candidate('임니다', 2.0), candidate('입니다', 1.0)]
    assert decision_features('imnida', '', candidates, [], raw=True)['reason'] == 'raw_mode'
    assert decision_features('imnida', '', candidates, [1.0, 1.0])['reason'] == 'parent_selected'
    candidates[0]['score'] = None
    assert decision_features('imnida', '', candidates, [0.1, 0.9])['reason'] == 'insufficient_finite_candidates'


def test_nonfinite_and_boolean_scores_are_rejected():
    for score in [float('inf'), float('nan'), True]:
        with pytest.raises(ValueError):
            finite_candidate_indices([candidate('입니다', score)])
        with pytest.raises(ValueError):
            decision_features('imnida', '', [candidate('입니다')], [score])


def test_selected_candidate_objects_and_remaining_order_are_preserved():
    values = [candidate('임니다'), candidate('입니다'), candidate('imnida', None)]
    reordered = reorder_candidates(values, 1)
    assert reordered == [values[1], values[0], values[2]]
    assert all(item is values[index] for item, index in zip(reordered, [1, 0, 2]))
    assert reorder_candidates(values, 0) == values
    with pytest.raises(ValueError):
        reorder_candidates(values, True)


def test_english_prefix_is_preserved_while_hangul_suffix_can_change():
    source = 'test imnida'
    def mixed(text):
        return {'text': 'test ' + text, 'score': 1.0, 'segments': [
            {'input_start': 0, 'input_end': 5, 'source': 'test ', 'text': 'test ', 'kind': 'raw'},
            {'input_start': 5, 'input_end': 11, 'source': 'imnida', 'text': text, 'kind': 'neural'},
        ]}
    parent, proposed = mixed('임니다'), mixed('입니다')
    assert proposal_allowed(source, parent, proposed)
    proposed['segments'][0]['text'] = '테스트 '
    proposed['text'] = '테스트 입니다'
    assert not proposal_allowed(source, parent, proposed)


def test_finite_scored_original_is_not_a_proposal_but_remains_in_features():
    candidates = [candidate('임니다', 2.0), candidate('입니다', 1.0),
                  candidate('imnida', 0.0, kind='raw')]
    result = decision_features('imnida', '', candidates, [0.1, 0.9, 100.0])
    assert result['proposal_index'] == 1
    assert result['finite_indices'] == [0, 1, 2]
    assert result['features'][3] == pytest.approx(99.1)
