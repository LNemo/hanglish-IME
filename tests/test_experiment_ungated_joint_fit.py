from copy import deepcopy
import pytest

from scripts.experiment_ungated_joint_fit import counts


def record(proposals):
    return {'arms': {mode: {'proposals': deepcopy(proposals)}
                     for mode in ('original', 'roman_repaired')}}


def test_counts_retain_supported_and_unsupported_candidates():
    rows = [record([{'text': '가', 'sentence_support': 'supported'},
                    {'text': '나', 'sentence_support': 'unsupported'}])]
    for value in counts(rows).values():
        assert value == {'rows': 1, 'proposals': 2, 'supported': 1, 'unsupported': 1}


@pytest.mark.parametrize('proposals', [
    [{'text': '가', 'sentence_support': 'supported'}] * 2,
    [{'text': '가', 'sentence_support': 'unknown'}],
    [{'text': str(i), 'sentence_support': 'supported'} for i in range(97)],
])
def test_counts_reject_duplicate_unclassified_and_over_budget_candidates(proposals):
    with pytest.raises(ValueError):
        counts([record(proposals)])
