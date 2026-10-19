from copy import deepcopy

import pytest

from hanglish.core.source_correspondence import align, compact_paths, contrast, expand_path


def path(parts):
    source = ''.join(a for a, _ in parts)
    cursor = 0
    segments = []
    for a, b in parts:
        segments.append(dict(input_start=cursor, input_end=cursor + len(a), source=a, text=b))
        cursor += len(a)
    return align(source, ''.join(b for _, b in parts), segments)


def test_unicode_offsets_and_exact_copy_do_not_depend_on_origin():
    actual = path([('😀', '😀'), ('test', 'test'), ('reul', '를'), (' ', ' '), ('hae', '해')])
    assert actual.spans[0].input_end == 1
    assert actual.spans[2].output_start == 5
    assert actual.spans[2].output_end == 6
    assert [s.action for s in actual.spans] == ['copy', 'copy', 'convert', 'copy', 'convert']
    assert ''.join(s.source for s in actual.spans) == actual.source
    assert ''.join(s.text for s in actual.spans) == actual.text


def test_mixed_edge_does_not_invent_an_internal_copy_alignment():
    coarse = path([('testreul', 'test를')])
    fine = path([('test', 'test'), ('reul', '를')])
    assert coarse.spans[0].action == 'convert'
    assert coarse.spans[0].unresolved_ascii_output
    assert not any(s.unresolved_ascii_output for s in fine.spans)
    block, = contrast(coarse, fine)
    assert block['relation'] == 'segmentation_only'
    assert block['input_start'] == 0 and block['input_end'] == 8
    assert block['left']['actions'] == ['convert']
    assert block['right']['actions'] == ['copy', 'convert']


def test_local_differences_keep_unchanged_context_and_source_mapping():
    left = path([('test', 'test'), ('reul', '를'), (' ', ' '), ('hamnida', '합니다')])
    right = path([('test', 'test'), ('reul', '를'), (' ', ' '), ('hamnida', '함니다')])
    blocks = contrast(left, right)
    assert [b['source'] for b in blocks if b['relation'] == 'output_changed'] == ['hamnida']
    assert ''.join(b['source'] for b in blocks) == left.source
    assert ''.join(b['left']['text'] for b in blocks) == left.text
    assert ''.join(b['right']['text'] for b in blocks) == right.text
    reverse = contrast(right, left)
    assert all(a['left'] == b['right'] and a['right'] == b['left'] for a, b in zip(blocks, reverse))


def test_different_boundaries_only_compare_at_shared_source_boundaries():
    left = path([('a', '아'), ('bc', '비시'), ('d', '디')])
    right = path([('ab', '압'), ('c', '시'), ('d', '디')])
    blocks = contrast(left, right)
    assert [(b['input_start'], b['input_end']) for b in blocks] == [(0, 3), (3, 4)]
    assert blocks[0]['source'] == 'abc'


def test_empty_output_does_not_break_offsets_or_contrast():
    left = path([('a', '아'), ('b', ''), ('c', '씨')])
    right = path([('a', '아'), ('b', '비'), ('c', '씨')])
    block = contrast(left, right)[1]
    assert block['left']['output_start'] == block['left']['output_end'] == 1
    assert block['relation'] == 'output_changed'
    empty = align('', '', [])
    assert contrast(empty, empty) == []


@pytest.mark.parametrize('patch', [
    {'input_start': True}, {'input_end': True}, {'input_start': 1},
    {'input_end': 0}, {'input_end': 3}, {'source': 'different'}, {'text': None},
])
def test_malformed_or_noncontiguous_source_edges_rejected(patch):
    segment = dict(input_start=0, input_end=2, source='ab', text='압')
    segment.update(patch)
    with pytest.raises(ValueError):
        align('ab', '압', [segment])


def test_complete_output_and_same_source_are_required():
    with pytest.raises(ValueError, match='complete output'):
        align('a', 'wrong', [dict(input_start=0, input_end=1, source='a', text='아')])
    with pytest.raises(ValueError, match='same source'):
        contrast(path([('a', '아')]), path([('b', '비')]))


def test_compaction_preserves_alternative_paths_and_deduplicates_only_identical_paths():
    coarse = path([('testreul', 'test를')])
    fine = path([('test', 'test'), ('reul', '를')])
    record = compact_paths([coarse, fine, coarse])
    assert len(record['paths']) == 2
    assert [expand_path(record, r) for r in record['paths']] == [coarse, fine]
    assert coarse.path_id != fine.path_id
    changed = deepcopy(record)
    changed['edges'][0]['text'] = '조작'
    with pytest.raises(ValueError):
        expand_path(changed, changed['paths'][0])


def test_metadata_scores_and_labels_do_not_enter_representation():
    segment = dict(input_start=0, input_end=1, source='a', text='아')
    base = align('a', '아', [segment])
    segment.update(kind='raw', score=float('nan'), expected=['틀린값'], context='정답 힌트')
    assert align('a', '아', [segment]) == base


@pytest.mark.parametrize('index', [-1, 1, True, '0'])
def test_compact_edge_indices_fail_closed(index):
    record = compact_paths([path([('a', '아')])])
    record['paths'][0]['edge_indices'] = [index]
    with pytest.raises(ValueError, match='edge index'):
        expand_path(record, record['paths'][0])
