import pytest

from hanglish.core.source_correspondence import align
from hanglish.models.span_supervision import extract_preferences


def path(parts):
    source = ''.join(a for a, _ in parts)
    cursor = 0; segments = []
    for a, b in parts:
        segments.append(dict(input_start=cursor, input_end=cursor + len(a), source=a, text=b))
        cursor += len(a)
    return align(source, ''.join(b for _, b in parts), segments)


def extract(paths, positives=(0,), groups=None):
    return extract_preferences(paths[0].source, [p.text for p in paths], list(positives),
                               groups or [[p] for p in paths])


def test_only_changed_block_is_labeled_and_outside_context_is_retained():
    a = path([('test', 'test'), ('reul', '를'), (' ', ' '), ('hamnida', '합니다')])
    b = path([('test', 'test'), ('reul', '를'), (' ', ' '), ('hamnida', '함니다')])
    pairs, counts = extract([a, b])
    pair, = pairs
    assert (pair['input_start'], pair['input_end'], pair['source']) == (9, 16, 'hamnida')
    assert (pair['prefix'], pair['suffix']) == ('test를 ', '')
    assert pair['witnesses'][0]['preferred']['span_indices'] == [3]
    assert counts['distinct_preferences'] == 1


def test_multiple_changes_are_excluded_instead_of_labeling_every_negative_span():
    a = path([('a', '아'), (' ', ' '), ('b', '비')])
    b = path([('a', '어'), (' ', ' '), ('b', '바')])
    pairs, counts = extract([a, b])
    assert not pairs and counts['excluded_multiple_or_zero_changes'] == 1


def test_whole_source_coarse_alignment_is_not_local_evidence():
    a = path([('ab', '아비')]); b = path([('ab', '아바')])
    pairs, counts = extract([a, b])
    assert not pairs and counts['excluded_whole_source'] == 1


def test_same_outside_text_with_different_alignment_is_excluded():
    a = path([('a', '아'), ('bc', '비시')])
    b = path([('a', '어'), ('b', '비'), ('c', '시')])
    pairs, counts = extract([a, b])
    assert not pairs and counts['excluded_outside_segmentation_changes'] == 1


def test_ambiguous_change_interval_excludes_entire_text_pair():
    fine = path([('a', 'A'), ('b', 'B'), ('c', 'C')])
    coarse = path([('ab', 'AB'), ('c', 'C')])
    negative = path([('a', 'X'), ('b', 'B'), ('c', 'C')])
    pairs, counts = extract([fine, negative], groups=[[fine, coarse], [negative]])
    assert not pairs and counts['excluded_ambiguous_interval_path_pairs'] == 2


def test_multiple_accepted_candidates_are_not_negative_examples():
    a = path([('a', '아'), ('x', 'x')]); b = path([('a', '어'), ('x', 'x')])
    c = path([('a', '오'), ('x', 'x')])
    pairs, _ = extract([a, b, c], (0, 1))
    assert len(pairs) == 2
    assert {p['preferred_index'] for p in pairs} == {0, 1}
    assert {p['dispreferred_index'] for p in pairs} == {2}


def test_different_boundaries_preserve_witnesses_without_inventing_copy():
    coarse = path([('testreul', 'test를'), (' ', ' ')])
    fine = path([('test', 'test'), ('reul', '를'), (' ', ' ')])
    negative = path([('testreul', '테스트를'), (' ', ' ')])
    pairs, _ = extract([coarse, negative], groups=[[coarse, fine, coarse], [negative]])
    pair, = pairs
    assert len(pair['witnesses']) == 2
    sides = [w['preferred'] for w in pair['witnesses']]
    assert {tuple(s['actions']) for s in sides} == {('convert',), ('copy', 'convert')}
    assert {s['unresolved_ascii_output'] for s in sides} == {True, False}


def test_unicode_and_deletion_offsets_are_preserved():
    a = path([('😀', '😀'), ('x', ''), ('a', '아')])
    b = path([('😀', '😀'), ('x', '엑스'), ('a', '아')])
    pairs, _ = extract([a, b]); pair, = pairs
    assert (pair['input_start'], pair['input_end']) == (1, 2)
    assert pair['preferred_text'] == ''
    assert pair['prefix'] == '😀' and pair['suffix'] == '아'
    assert pair['witnesses'][0]['preferred']['output_start'] == 1
    assert pair['witnesses'][0]['preferred']['output_end'] == 1


@pytest.mark.parametrize('positives', [[], [True], [0, 0], [-1], [2], [0, 1]])
def test_invalid_accepted_domain_fails(positives):
    a = path([('a', '아'), ('x', 'x')]); b = path([('a', '어'), ('x', 'x')])
    with pytest.raises(ValueError):
        extract([a, b], positives)


def test_output_is_deterministic_under_duplicate_and_reordered_paths():
    coarse = path([('testreul', 'test를'), (' ', ' ')])
    fine = path([('test', 'test'), ('reul', '를'), (' ', ' ')])
    negative = path([('testreul', '테스트를'), (' ', ' ')])
    a = extract([coarse, negative], groups=[[coarse, fine], [negative]])
    b = extract([coarse, negative], groups=[[fine, coarse, fine], [negative]])
    assert a == b
