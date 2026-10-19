import pytest

from scripts.prepare_document_phonetic_inputs import project


def test_single_letter_input_and_same_source_variants_are_preserved():
    words = {'이': 'i', '있는': 'inneun', '잇는': 'inneun'}
    requests, counts = project([{'id': 'a', 'text': '이 있는 잇는 있는'}], words.get)
    assert [r['input'] for r in requests] == ['i', 'inneun']
    assert counts['reference_pairs'] == 3
    assert counts['reference_occurrences'] == 4
    assert all(set(r) == {'id', 'input'} for r in requests)


def test_maximal_runs_and_english_remain_separate_from_phonetic_evaluation():
    requests, counts = project([{'id': 'a', 'text': 'test를 시행한다.'}])
    assert [r['input'] for r in requests] == ['reul', 'sihaenghanda']
    assert counts['reference_occurrences'] == 2


def test_unsupported_references_counted_instead_of_silently_dropped():
    requests, counts = project([{'id': 'a', 'text': '가' * 25 + ' 나'}],
                              lambda s: 'x' * 49 if len(s) > 24 else 'na')
    assert [r['input'] for r in requests] == ['na']
    assert counts['reference_pairs'] == 2
    assert counts['input_unsupported_pairs'] == counts['output_bound_exceeded_pairs'] == 1


def test_duplicates_rejected_and_order_independent():
    a, b = {'id': 'a', 'text': '한국어'}, {'id': 'b', 'text': '입니다'}
    assert project([a, b]) == project([b, a])
    with pytest.raises(ValueError, match='duplicate'):
        project([a, a])
