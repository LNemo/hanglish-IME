import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_copy_child_fit_metrics import independent_outcome, verify


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture(tmp_path):
    shard = tmp_path / 'shard'
    capture = tmp_path / 'capture'
    evaluation = tmp_path / 'evaluation'
    shard.mkdir(); capture.mkdir(); evaluation.mkdir()
    sources = [
        {'id': 'a', 'request': {'input': 'a', 'context': ''},
         'scoring': {'ranked': True, 'original': [{'text': '아'}], 'novel': [],
                     'fallback_public': []}},
        {'id': 'b', 'request': {'input': 'b', 'context': ''},
         'scoring': {'ranked': False, 'original': [], 'novel': [],
                     'fallback_public': [{'text': '비'}]}},
    ]
    proposals = [
        {'id': 'a', 'request': {'input': 'a', 'context': ''},
         'proposals': [{'text': '앗'}]},
        {'id': 'b', 'request': {'input': 'b', 'context': ''}, 'proposals': []},
    ]
    labels = [
        {'id': 'a', 'input': 'a', 'expected': ['앗']},
        {'id': 'b', 'input': 'b', 'expected': ['비']},
    ]
    (shard / 'capture.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in sources))
    write_json(shard / 'independent-verification.json', {'verified': True})
    (capture / 'capture.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in proposals))
    capture_report = {'completed': True, 'rows': 2, 'labels_opened': False, 'pins': {}}
    write_json(capture / 'report.json', capture_report)
    write_json(capture / 'independent-verification.json', {
        'verified': True, 'rows': 2, 'labels_opened': False,
        'capture_sha256': sha(capture / 'capture.jsonl'),
        'report_sha256': sha(capture / 'report.json'),
    })
    outcomes = [
        {'id': 'a', 'old_scorable_covered': False, 'old_available_covered': False,
         'expanded_available_covered': True, 'coverage_gain': True, 'coverage_loss': False,
         'new_gold_proposals': 1, 'new_proposals': 1, 'no_pool': False},
        {'id': 'b', 'old_scorable_covered': False, 'old_available_covered': True,
         'expanded_available_covered': True, 'coverage_gain': False, 'coverage_loss': False,
         'new_gold_proposals': 0, 'new_proposals': 0, 'no_pool': True},
    ]
    (evaluation / 'outcomes.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in outcomes))
    labels_path = tmp_path / 'labels.jsonl'
    labels_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in labels))
    evaluator = tmp_path / 'evaluator.py'; evaluator.write_text('# frozen evaluator\n')
    counts = {'rows': 2, 'old_scorable_covered': 0, 'old_available_covered': 1,
              'expanded_available_covered': 2, 'coverage_gain': 1, 'coverage_loss': 0,
              'new_gold_proposals': 1, 'new_proposals': 1, 'no_pool': 1}
    write_json(evaluation / 'report.json', {
        'completed': True, 'rows': 2, 'counts': counts,
        'capture_directory': str(capture.resolve()),
        'capture_sha256': sha(capture / 'capture.jsonl'),
        'capture_report_sha256': sha(capture / 'report.json'),
        'capture_proof_sha256': sha(capture / 'independent-verification.json'),
        'labels_sha256': sha(labels_path),
        'outcomes_sha256': sha(evaluation / 'outcomes.jsonl'),
        'evaluator_sha256': sha(evaluator),
        'labels_opened_after_verified_capture': True,
    })
    hashes = {name: sha(shard / name) for name in ('capture.jsonl', 'independent-verification.json')}
    intake = tmp_path / 'intake.json'
    write_json(intake, {'shards': [{'shard': 0, 'directory': str(shard), 'hashes': hashes}]})
    return evaluation, capture, intake, labels_path, evaluator


def test_full_independent_recount_and_exclusive_proof(tmp_path):
    evaluation, _, intake, labels, evaluator = build_fixture(tmp_path)
    result = verify(evaluation, intake_path=intake, intake_sha=sha(intake),
                    labels_path=labels, labels_sha=sha(labels), evaluator_path=evaluator,
                    expected_rows=2, expected_shards=1, expected_no_pool=1,
                    expected_old_scorable=0)
    assert result['verified'] and result['counts']['coverage_gain'] == 1
    with pytest.raises(ValueError, match='already exists'):
        verify(evaluation, intake_path=intake, intake_sha=sha(intake),
               labels_path=labels, labels_sha=sha(labels), evaluator_path=evaluator,
               expected_rows=2, expected_shards=1, expected_no_pool=1,
               expected_old_scorable=0)


def test_source_proof_failure_precedes_label_hashing(tmp_path, monkeypatch):
    evaluation, capture, intake, labels, evaluator = build_fixture(tmp_path)
    proof = json.loads((capture / 'independent-verification.json').read_text())
    proof['verified'] = False
    write_json(capture / 'independent-verification.json', proof)
    report = json.loads((evaluation / 'report.json').read_text())
    report['capture_proof_sha256'] = sha(capture / 'independent-verification.json')
    write_json(evaluation / 'report.json', report)
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path):
        assert path != labels, 'labels were hashed before source proof validation'
        return original_read_bytes(path)

    monkeypatch.setattr(Path, 'read_bytes', guarded_read_bytes)
    with pytest.raises(ValueError, match='verified source-only'):
        verify(evaluation, intake_path=intake, intake_sha=sha(intake),
               labels_path=labels, labels_sha='unused', evaluator_path=evaluator,
               expected_rows=2, expected_shards=1, expected_no_pool=1,
               expected_old_scorable=0)


def test_outcome_rejects_duplicate_old_proposal():
    source = {'id': 'a', 'request': {'input': 'a', 'context': ''},
              'scoring': {'ranked': True, 'original': [{'text': '아'}],
                          'novel': [], 'fallback_public': []}}
    capture = {'id': 'a', 'request': source['request'], 'proposals': [{'text': '아'}]}
    label = {'id': 'a', 'input': 'a', 'expected': ['아']}
    with pytest.raises(ValueError, match='duplicates'):
        independent_outcome(source, capture, label)
