from scripts.evaluate_ungated_mixed_dev import measure, public_texts
from scripts.evaluate_ungated_mixed_union_fit import transitions


def test_multiple_valid_answers_use_minimum_distance():
    result = measure('testreul', [{'text': 'test를'}], ['test를', '테스트를'])
    assert result['first'] and result['ascii_exact'] and result['edit_distance'] == 0


def test_partial_damage_counts_even_without_sentence_accuracy_change():
    a = measure('imnida', [{'text': '임니다'}], ['입니다'])
    b = measure('imnida', [{'text': '임니가'}], ['입니다'])
    change = transitions(a, b)
    assert change['regressions'] == 0 and change['partial_regressions'] == 1


def test_all_unsupported_keeps_original_fallback():
    baseline = {'input': 'test', 'candidates': [{'text': '테스트'}, {'text': 'test'}]}
    assert public_texts({'pool': [{'text': '길어요', 'supported': False, 'score': None}]}, baseline) == ['테스트', 'test']
