import copy
import json
from types import SimpleNamespace

import pytest

from scripts import evaluate_ctc_union_runtime as module


def response():
    source = 'ga na'
    def candidate(text, score, kind):
        return dict(text=text, score=score, segments=[dict(
            input_start=0, input_end=5, source=source, text=text, kind=kind)])
    info = dict(experimental=True, generation_requests=1, completed_generations=1,
                generated_proposals=3, accepted_proposals=2,
                prediction=dict(reason='complete', text='가 나'), generation_ms=2.)
    return dict(sentence_ranking_applied=True, result='가 나',
                candidates=[candidate('가 나', 1., 'fixture'), candidate(source, None, 'raw')],
                ctc_word=info, ctc_union=copy.deepcopy(info))


def test_both_generation_stages_must_obey_runtime_contract():
    before = dict(sentence_ranking_applied=True)
    module.verify_union_contract('ga na', before, response())
    for stage in ('ctc_word', 'ctc_union'):
        for field, value in [('generation_requests', 2), ('accepted_proposals', 4)]:
            broken = response()
            broken[stage][field] = value
            with pytest.raises(ValueError):
                module.verify_union_contract('ga na', before, broken)


def test_original_ctc_predictions_and_provenance_are_preserved():
    parent = SimpleNamespace(version='ctc1')
    proposed = SimpleNamespace(version='union', growth_artifacts={'growth': 'sha'})
    before = response()
    before['version'] = 'ctc1'
    after = copy.deepcopy(before)
    after['version'] = 'union'
    after['ctc_union'].update(parent_version='ctc1', artifacts={'growth': 'sha'})
    after['ctc_word']['generation_ms'] = 5.
    module.verify_identity(before, after, parent, proposed)
    after['ctc_word']['prediction']['text'] = 'changed'
    with pytest.raises(ValueError, match='original CTC1'):
        module.verify_identity(before, after, parent, proposed)


def test_reference_hash_contract_and_source_checked(tmp_path, monkeypatch):
    row = dict(id='x', request=dict(input='ga na', context=''))
    capture = tmp_path / 'capture.jsonl'
    report = tmp_path / 'report.json'
    metadata = dict(completed=True, split='dev', input_contract='roman_sentence_only',
                    proposed=dict(counts=dict(rows=1, top1=1, top5=1, errors=0)))
    def write():
        capture.write_text(json.dumps(row) + '\n')
        metadata['capture_sha256'] = module.sha(capture)
        report.write_text(json.dumps(metadata))
        monkeypatch.setitem(module.REFERENCE, 'dev',
                            (module.sha(report), module.sha(capture), 1, 1, 1))
    inputs = [dict(id='x', input='ga na')]
    write()
    assert module.load_reference('dev', tmp_path, inputs) == [row]
    report.write_text(report.read_text() + ' ')
    with pytest.raises(ValueError, match='identity'):
        module.load_reference('dev', tmp_path, inputs)
    row['request']['context'] = 'hint'
    write()
    with pytest.raises(ValueError, match='source'):
        module.load_reference('dev', tmp_path, inputs)


def test_test_gate_precedes_model_loading_and_rejects_other_experiments(tmp_path):
    with pytest.raises(ValueError, match='--final'):
        module.run('test', tmp_path / 'unused')
    with pytest.raises(ValueError, match='DEV qualification'):
        module.run('test', tmp_path / 'unused', final=True)
    capture = tmp_path / 'capture.jsonl'
    capture.write_text('{}\n')
    source = module.Path(module.__file__).resolve()
    report = dict(completed=True, split='dev', input_contract='roman_sentence_only',
                  experiment='different_model', improvement_gate=True,
                  capture_sha256=module.sha(capture), gold_not_passed_to_predictor=True,
                  proposed=dict(counts=dict(rows=316)),
                  parent=dict(counts=dict(rows=316, top1=260, top5=280, errors=0)),
                  pins={str(source): module.sha(source)})
    (tmp_path / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='current union runtime'):
        module.qualify(tmp_path)
    report['experiment'] = 'ctc_growth_union_v1'
    (tmp_path / 'report.json').write_text(json.dumps(report))
    assert module.qualify(tmp_path)['capture_sha256'] == module.sha(capture)
