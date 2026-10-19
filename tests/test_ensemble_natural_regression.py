import json

import pytest

from scripts import evaluate_ensemble_natural_regression as evaluation


def test_fixed_source_hashes_are_complete_sha256_values():
    for value in [evaluation.PREVIOUS_SHA, *(sha for _,_,sha in evaluation.SOURCES)]:
        assert len(value)==64 and all(c in '0123456789abcdef' for c in value)


def test_remaining_projection_keeps_expected_out_of_converter_input(tmp_path):
    row=dict(id='one',input='imnida',context='context',expected=['입니다'],child_candidates=[])
    path=tmp_path/'cases.jsonl';path.write_text(json.dumps(row)+'\n')
    cases=evaluation.read_cases(tmp_path,[('train','cases.jsonl',evaluation.base.file_sha256(path))],{'natural_train':1})
    projected=evaluation.source_rows(cases)
    assert projected==[dict(id='one',input='imnida',context='context',suite='natural_train',category='natural_train')]
    with pytest.raises(ValueError,match='denominator'):
        evaluation.read_cases(tmp_path,[('train','cases.jsonl',evaluation.base.file_sha256(path))],{'natural_train':2})


def test_baseline_parity_checks_complete_candidates():
    candidate=dict(text='입니다',score=1.,segments=[dict(source='imnida',text='입니다')])
    source=dict(id='one',input='imnida',context='',child_candidates=[candidate])
    capture=dict(suite='natural_train',id='one',input='imnida',context='',baseline=dict(candidates=[candidate]))
    evaluation.compare_baselines([capture],[('natural_train',source)])
    capture['baseline']={'candidates':[dict(candidate,score=2.)]}
    with pytest.raises(ValueError,match='candidate objects'):
        evaluation.compare_baselines([capture],[('natural_train',source)])
