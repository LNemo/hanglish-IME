from copy import deepcopy

import pytest

from scripts.audit_source_correspondence_fit import project_fit


def fixture():
    source = 'testreul hae'
    parts = [('test', 'test'), ('reul', '를'), (' ', ' '), ('hae', '해')]
    cursor = 0; segments = []
    for a, b in parts:
        segments.append(dict(input_start=cursor, input_end=cursor + len(a), source=a, text=b))
        cursor += len(a)
    candidate = {'text': 'test를 해', 'segments': segments}
    opaque = deepcopy(candidate)
    opaque['segments'][:2] = [dict(input_start=0, input_end=8, source='testreul', text='test를')]
    runtime = {'id': 'fit-1', 'request': {'input': source, 'context': ''},
               'proposed': {'input': source, 'context': '', 'result': candidate['text'],
                            'pipeline': {'pool': [candidate, opaque]}, 'candidates': [candidate]}}
    generated = {'id': 'fit-1', 'request': runtime['request'], 'proposals': []}
    return ({'id': 'fit-1', 'input': source},
            {'id': 'fit-1', 'split': 'fit', 'component_id': 'global-component'}, runtime, generated)


def test_projection_retains_paths_without_creating_or_selecting_a_gold_target():
    inputs = fixture()
    record, stats = project_fit(*inputs)
    assert set(record) == {'schema', 'input', 'edges', 'paths', 'id', 'component_id', 'reference_path_id'}
    assert len(record['paths']) == 2
    assert stats['counts']['input_path_occurrences'] == 3
    assert stats['counts']['outputs_with_multiple_paths'] == 1
    assert stats['counts']['paths_with_unresolved_ascii'] == 1
    assert stats['counts']['outputs_with_only_unresolved_ascii_paths'] == 0
    assert stats['counts']['segmentation_only_blocks'] == 1


def test_hold_is_rejected_before_candidate_access():
    selected, assignment, _, _ = fixture()
    assignment['split'] = 'inner_hold'
    with pytest.raises(ValueError, match='FIT-only'):
        project_fit(selected, assignment, {}, {})


def test_source_context_and_identity_mismatch_fail_closed():
    selected, assignment, runtime, generated = fixture()
    runtime['request']['context'] = '정답 문맥'
    with pytest.raises(ValueError, match='source identity'):
        project_fit(selected, assignment, runtime, generated)


def test_unused_metadata_does_not_change_source_projection():
    inputs = fixture()
    before = project_fit(*inputs)
    inputs[2]['proposed']['pipeline']['pool'][0].update(score=1000, expected=['ignored'])
    assert project_fit(*inputs) == before


def test_raw_output_with_different_boundaries_is_still_contrasted():
    selected, assignment, runtime, generated = fixture()
    response = runtime['proposed']
    fine = deepcopy(response['candidates'][0])
    fine['text'] = selected['input']
    for segment in fine['segments']:
        segment['text'] = segment['source']
    coarse = {'text': selected['input'], 'segments': [dict(
        input_start=0, input_end=len(selected['input']),
        source=selected['input'], text=selected['input'])]}
    response.update(result=selected['input'], candidates=[fine], pipeline={'pool': [fine, coarse]})
    record, stats = project_fit(selected, assignment, runtime, generated)
    assert len(record['paths']) == 2
    assert stats['counts']['contrasts'] == stats['counts']['raw_contrasts'] == 1
    assert stats['counts']['nonraw_contrasts'] == stats['counts']['output_changed_blocks'] == 0
    assert stats['counts']['segmentation_only_blocks'] == 1
