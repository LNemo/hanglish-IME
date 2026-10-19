import json
import hashlib

import pytest

from scripts.evaluate_challenge import evaluate


def test_evaluator_records_failures_and_preserves_first_exposure(tmp_path):
    cases = [
        {'id': 'ok', 'input': 'imnida', 'context': '', 'expected': ['입니다'], 'category': 'copula'},
        {'id': 'bad', 'input': 'imnida', 'context': '', 'expected': ['NOT THE ANSWER'], 'category': 'copula'},
    ]
    source = tmp_path / 'cases.jsonl'
    source.write_text('\n'.join(json.dumps(row) for row in cases))
    output = tmp_path / 'report.json'
    result = evaluate(source, 'baseline', output)
    assert result['counts']['total'] == 2
    assert result['counts']['top1'] == 1
    assert result['passed'] is False
    assert result['corpus_identity']['verified'] is False
    assert result['outcomes'][1]['actual'][0] == '입니다'
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        evaluate(source, 'rules', output)
    assert output.read_bytes() == before


def test_empty_evaluation_cannot_pass(tmp_path):
    source = tmp_path / 'empty.jsonl'
    source.write_text('')
    assert evaluate(source, 'baseline', tmp_path / 'report.json')['passed'] is False


def test_cer_uses_length_of_closest_reference(tmp_path, monkeypatch):
    import scripts.evaluate_challenge as module
    class Converter:
        def convert(self, *args, **kwargs):
            return {'candidates': [{'text': 'a' * 99}], 'version': 'test', 'ranker': 'semantic-checkpoint'}
    monkeypatch.setattr(module, 'WorkbenchEngine', Converter)
    source = tmp_path / 'multi.jsonl'
    source.write_text(json.dumps({'id':'multi', 'input':'x', 'context':'',
                                 'expected':['a' * 100, 'b'], 'category':'multi'}))
    report = evaluate(source, 'baseline', tmp_path / 'report.json')
    assert report['cer'] == .01
    assert report['requested_ranker'] == 'auto'
    assert report['resolved_rankers'] == ['semantic-checkpoint']
    assert report['outcomes'][0]['ranker'] == 'semantic-checkpoint'


def test_versioned_evaluation_checks_identity_before_inference(tmp_path, monkeypatch):
    import scripts.evaluate_challenge as module
    source = tmp_path / 'test.jsonl'
    source.write_text(json.dumps({'id': 'one', 'input': 'imnida', 'context': '',
                                 'expected': ['입니다'], 'category': 'copula', 'split': 'test'}))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema_version': 2, 'corpus_version': 'sample',
                                   'sha256': {'test.jsonl': hashlib.sha256(source.read_bytes()).hexdigest()},
                                   'counts': {'test': 1}}))
    report = evaluate(source, 'baseline', tmp_path / 'report.json', manifest=manifest)
    assert report['corpus_identity']['verified']
    assert report['corpus_identity']['corpus_version'] == 'sample'
    assert report['corpus_identity']['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    source.write_text(source.read_text() + '\n')
    monkeypatch.setattr(module, 'WorkbenchEngine', lambda: pytest.fail('must reject before model construction'))
    with pytest.raises(ValueError, match='manifest'):
        evaluate(source, 'baseline', tmp_path / 'bad.json', manifest=manifest)
    assert not (tmp_path / 'bad.json').exists()


def test_explicit_missing_manifest_cannot_be_reported_as_verified(tmp_path):
    source = tmp_path / 'test.jsonl'
    source.write_text('')
    with pytest.raises(FileNotFoundError):
        evaluate(source, 'baseline', tmp_path / 'report.json', manifest=tmp_path / 'missing.json')
