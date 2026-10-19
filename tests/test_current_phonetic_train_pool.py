from copy import deepcopy
import pytest

from scripts.current_phonetic_train_pool import source_rows, partition, estimates, check_capture, validate_engine_identity, ZERO_VERSION


def test_projection_excludes_targets_and_prior_candidates():
    row = dict(id='s3-train-1', input='testreul sihaeng', component_id='group1',
               expected=['secret target'], paths=['old candidates'], reference_path_id='gold')
    assert source_rows([row]) == [dict(id='s3-train-1', input='testreul sihaeng', component_id='group1')]


@pytest.mark.parametrize('update', [dict(id='s3-test-1'), dict(input='한국어 hint'), dict(component_id='')])
def test_source_domain_rejects_test_hint_or_missing_group(update):
    row = dict(id='s3-train-1', input='roman source', component_id='component') | update
    with pytest.raises(ValueError):
        source_rows([row])


def test_duplicate_ids_are_rejected():
    row = dict(id='s3-train-1', input='roman source', component_id='component')
    with pytest.raises(ValueError):
        source_rows([row, row])


def test_partition_keeps_every_component_together_and_every_row_once():
    rows = [dict(id=str(i), input='a'*(i+1), component_id=str(i//3)) for i in range(30)]
    shards = partition(rows)
    assert sorted(i for shard in shards for i in shard) == sorted(r['id'] for r in rows)
    locations = {identifier: index for index, shard in enumerate(shards) for identifier in shard}
    for group in range(10):
        assert len({locations[str(i)] for i in range(group*3, group*3+3)}) == 1
    assert partition(list(reversed(rows))) == [list(reversed(shard)) for shard in shards]


def test_projection_charges_every_row_in_each_shard():
    groups = [dict(probe_id='a', ids=['a','b']), dict(probe_id='c', ids=['c','d'])]
    assert estimates(groups, [['a','c'],['b','d']], {'a':2.,'c':5.}, 10.) == [82.1,82.1]


def test_zero_capture_cannot_hide_changed_scores_or_missing_paths():
    segment = dict(input_start=0, input_end=4, source='test', text='test', kind='raw')
    paths = [('test',1.,[segment])]
    response = dict(input='test', context='', external_context_used=False, sentence_ranking_applied=True,
        candidates=[dict(text='test',score=1.,segments=[segment])],
        phonetic_word_union=dict(union=['test']),
        phonetic_choice=dict(effective_weights=[0.]*4,calibration_source='override',
            zero_weight_score_parity=True,zero_weight_order_parity=True,
            rows=[dict(text='test',base_score=1.,adjusted_score=1.,residual=0.,features=[0.]*4,segments=[segment])]))
    check_capture('test',response,paths)
    changed=deepcopy(response); changed['phonetic_choice']['rows'][0]['adjusted_score']=2.
    with pytest.raises(ValueError):
        check_capture('test',changed,paths)
    with pytest.raises(ValueError):
        check_capture('test',response,[])
    changed=deepcopy(response); changed['phonetic_choice']['rows'][0]['features']=[float('nan')]*4
    with pytest.raises(ValueError, match='features'):
        check_capture('test',changed,paths)


def test_identity_rejects_mutated_phonetic_pipeline_despite_same_ctc_parent():
    from types import SimpleNamespace
    engine=SimpleNamespace(phonetic_parent_version='experimental-ctc-union-1-488fc89732af',
                           version=ZERO_VERSION, model_artifacts={'model.pt':'frozen'})
    validate_engine_identity(engine, {'model.pt':'frozen'})
    engine.model_artifacts={'model.pt':'changed'}
    with pytest.raises(ValueError, match='identity'):
        validate_engine_identity(engine, {'model.pt':'frozen'})
