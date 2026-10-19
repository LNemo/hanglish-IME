import pytest
import json
from types import SimpleNamespace
from scripts.evaluate_neural_word_runtime import compare, latency, load_reference, verify_response_identity, sha


def response(*texts, ms=1.):
    return dict(candidates=[dict(text=t) for t in texts], engine_ms=ms)


def test_comparison_counts_complete_denominator_and_regressions():
    labels = [dict(id=str(i), input='roman sentence', expected=['right']) for i in range(3)]
    captures = [
        dict(id='0', parent=response('wrong'), proposed=response('right')),
        dict(id='1', parent=response('right'), proposed=response('wrong')),
        dict(id='2', parent=response('wrong', 'right'), proposed=response('wrong')),
    ]
    report = compare(captures, labels)
    assert report['proposed']['counts']['rows'] == 3
    assert report['corrections'] == ['0']
    assert report['regressions'] == ['1']
    assert report['top5_losses'] == ['1', '2']
    assert not report['improvement_gate']


def test_displayed_original_text_option_is_scored_as_actually_returned():
    labels = [dict(id='x', input='roman sentence', expected=['gold'])]
    row = dict(id='x', parent=response('wrong', 'gold'),
               proposed=response('a', 'b', 'c', 'd', 'roman sentence'))
    result = compare([row], labels)
    assert result['top5_losses'] == ['x']
    assert result['proposed']['counts']['top5'] == 0


def test_latency_includes_complete_engine_measurement():
    captures = [dict(parent=response(ms=i), proposed=response(ms=i + 1000)) for i in range(1, 21)]
    assert latency(captures)['proposed']['p95_ms'] == 1019
    captures[0]['proposed']['engine_ms'] = float('nan')
    with pytest.raises(ValueError):
        latency(captures)


def test_reference_identity_contract_and_source_are_all_checked(tmp_path, monkeypatch):
    from scripts import evaluate_neural_word_runtime as module
    capture = tmp_path / 'capture.jsonl'
    report = tmp_path / 'report.json'
    inputs = [dict(id='x', input='roman sentence')]
    reference = dict(id='x', request=dict(input='roman sentence', context=''))
    metadata = dict(completed=True, split='dev', input_contract='roman_sentence_only',
                    mixed=dict(counts=dict(rows=1, top1=1, top5=1, errors=0)))

    def write_and_pin():
        capture.write_text(json.dumps(reference) + '\n')
        metadata['capture_sha256'] = sha(capture)
        report.write_text(json.dumps(metadata))
        monkeypatch.setitem(module.REFERENCE, 'dev', (sha(report), sha(capture), 1, 1, 1))

    write_and_pin()
    assert load_reference('dev', tmp_path, inputs) == [reference]
    report.write_text(report.read_text() + ' ')
    with pytest.raises(ValueError, match='identity'):
        load_reference('dev', tmp_path, inputs)
    metadata['mixed']['counts']['top1'] = 0
    write_and_pin()
    with pytest.raises(ValueError, match='contract'):
        load_reference('dev', tmp_path, inputs)
    metadata['mixed']['counts']['top1'] = 1
    reference['request']['context'] = 'Korean hint'
    write_and_pin()
    with pytest.raises(ValueError, match='source'):
        load_reference('dev', tmp_path, inputs)


def test_response_identity_rejects_version_and_checkpoint_drift():
    parent = SimpleNamespace(version='parent')
    proposed = SimpleNamespace(version='child', neural_word_artifacts={'model.pt': 'hash'})
    before = dict(version='parent')
    after = dict(version='child', neural_word=dict(parent_version='parent', artifacts={'model.pt': 'hash'}))
    verify_response_identity(before, after, parent, proposed)
    after['neural_word']['artifacts']['model.pt'] = 'changed'
    with pytest.raises(ValueError, match='identity'):
        verify_response_identity(before, after, parent, proposed)
