from copy import deepcopy

import pytest

from scripts.finite_final_choice_policy import candidate_view, select_candidates


def candidate(text, score, kind='neural'):
    return {'text': text, 'score': score, 'segments': [
        {'input_start': 0, 'input_end': 6, 'source': 'imnida', 'text': text, 'kind': kind},
    ]}


def rows():
    return [candidate('임니다', 3.), candidate('imnida', None, 'raw'), candidate('입니다', 2.)]


class Ranker:
    def __init__(self, scores=(0., 2.), oversize='imnida'):
        self.scores, self.oversize = scores, oversize
        self.checked, self.calls = [], []

    def can_score(self, text):
        self.checked.append(text)
        return text != self.oversize

    def score_pairs(self, contexts, texts):
        self.calls.append((contexts, texts))
        return self.scores


def test_unscored_original_survives_without_budget_or_inference_and_indices_map():
    values, ranker = rows(), Ranker()
    before = deepcopy(values)
    result = select_candidates(ranker, 'imnida', '문맥', values)
    assert ranker.checked == ['임니다', '입니다']
    assert ranker.calls == [(['문맥', '문맥'], ['임니다', '입니다'])]
    assert result['chosen_index'] == 2
    assert result['scored_source_indices'] == [0, 2]
    assert result['candidates'] == [values[2], values[0], values[1]]
    assert all(any(item is old for old in values) for item in result['candidates'])
    assert values == before


def test_finite_oversize_cannot_be_dropped_to_force_a_choice():
    values, ranker = rows(), Ranker(oversize='입니다')
    result = select_candidates(ranker, 'imnida', '', values)
    assert result['reason'] == 'candidate_budget'
    assert result['candidates'] is values
    assert ranker.calls == []


def test_raw_mode_and_insufficient_candidates_do_not_access_model():
    values = rows()
    assert select_candidates(None, 'imnida', '', values, raw=True)['candidates'] is values
    assert candidate_view(None, 'imnida', [values[0], values[1]]) == ([0], 'insufficient_finite_candidates')
    assert candidate_view(None, 'imnida', [values[1], values[0]]) == ([1], 'insufficient_finite_candidates')


@pytest.mark.parametrize('mutation', ['non_raw', 'wrong_text', 'short_span', 'duplicate', 'missing_score'])
def test_only_valid_single_raw_original_can_be_excluded(mutation):
    values = rows()
    if mutation == 'non_raw':
        values[1]['segments'][0]['kind'] = 'neural'
    elif mutation == 'wrong_text':
        values[1]['text'] = '비정상누락'
    elif mutation == 'short_span':
        values[1]['segments'][0]['input_end'] = 5
    elif mutation == 'duplicate':
        values.append(deepcopy(values[1]))
    else:
        del values[1]['score']
    ranker = Ranker()
    with pytest.raises(ValueError):
        select_candidates(ranker, 'imnida', '', values)
    assert ranker.checked == []
    assert ranker.calls == []


def test_scored_original_is_still_budget_checked():
    values = rows()
    values[1]['score'] = 1.
    result = select_candidates(Ranker(), 'imnida', '', values)
    assert result['reason'] == 'candidate_budget'


@pytest.mark.parametrize('scores', [(1.,), (1., 2., 3.), (1., float('nan')), (1., float('inf')), (1., True)])
def test_malformed_scores_fail(scores):
    with pytest.raises(ValueError, match='every finite candidate'):
        select_candidates(Ranker(scores), 'imnida', '', rows())


def test_tie_preserves_parent_list_identity():
    values = rows()
    assert select_candidates(Ranker((1., 1.)), 'imnida', '', values)['candidates'] is values


def test_forbidden_best_candidate_does_not_fall_through_to_next_best():
    values = [candidate('임니다', 3.), candidate('입니다', 2.), candidate('입니다', 1., 'mixed')]
    result = select_candidates(Ranker((0., 2., 3.)), 'imnida', '', values)
    assert result['reason'] == 'protected_or_unaligned'
    assert result['candidates'] is values
