"""Independently recount late FIT coverage for COPY-child proposals."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_recombination_tokenization import digest, require

INTAKE = ROOT / '.omx/artifacts/current-fit-recombination-selection-intake-v1.json'
INTAKE_SHA = '70e5c963c2f78ae15f36e2bededb67f65c12b8430b6f9eb359b7cf44648d17aa'
LABELS = ROOT / '.omx/artifacts/current-phonetic-train-features-v1/rows.jsonl'
LABEL_SHA = '4c13063656d9a108316b9ce1ceec394a61e2c45c2ba86e7a8c8361f7480d6cb4'
EVALUATOR = ROOT / 'scripts/evaluate_copy_child_fit.py'


def independent_outcome(source, capture, label):
    require(source['id'] == capture['id'] == label['id']
            and source['request'] == capture['request'] == {'input': label['input'], 'context': ''},
            'FIT identity differs')
    scoring = source['scoring']
    gold = set(label['expected'])
    require(gold and all(isinstance(text, str) for text in gold), 'FIT gold missing')
    scorable = {candidate['text'] for candidate in scoring['original']}
    scorable.update(candidate['text'] for candidate in scoring['novel']
                    if candidate['status'] == 'scored')
    available = {candidate['text'] for candidate in scoring['original'] + scoring['novel']}
    if not scoring['ranked']:
        available.update(candidate['text'] for candidate in scoring['fallback_public'])
    proposed = [candidate['text'] for candidate in capture['proposals']]
    new = set(proposed)
    require(len(new) == len(proposed) and not new & available,
            'new proposal duplicates old pool')
    before = bool(gold & available)
    after = bool(gold & (available | new))
    return {
        'id': source['id'],
        'old_scorable_covered': bool(gold & scorable),
        'old_available_covered': before,
        'expanded_available_covered': after,
        'coverage_gain': not before and after,
        'coverage_loss': before and not after,
        'new_gold_proposals': len(gold & new),
        'new_proposals': len(new),
        'no_pool': not scoring['ranked'],
    }


def _read_json(path):
    return json.loads(path.read_text())


def _source_preflight(evaluation, *, intake_path, intake_sha, evaluator_path,
                      expected_rows, expected_shards):
    report_path = evaluation / 'report.json'
    report = _read_json(report_path)
    require(report.get('completed') is True and report.get('rows') == expected_rows
            and report.get('labels_opened_after_verified_capture') is True,
            'evaluation incomplete')
    capture = Path(report['capture_directory'])
    capture_report_path = capture / 'report.json'
    capture_proof_path = capture / 'independent-verification.json'
    capture_path = capture / 'capture.jsonl'
    outcomes_path = evaluation / 'outcomes.jsonl'
    require(capture.resolve() == Path(report['capture_directory']).resolve()
            and digest(capture_path) == report['capture_sha256']
            and digest(capture_report_path) == report['capture_report_sha256']
            and digest(capture_proof_path) == report['capture_proof_sha256']
            and digest(outcomes_path) == report['outcomes_sha256']
            and digest(evaluator_path) == report['evaluator_sha256'],
            'evaluation evidence changed')
    capture_report = _read_json(capture_report_path)
    source_proof = _read_json(capture_proof_path)
    require(capture_report.get('completed') is True
            and capture_report.get('rows') == expected_rows
            and capture_report.get('labels_opened') is False
            and source_proof.get('verified') is True
            and source_proof.get('rows') == expected_rows
            and source_proof.get('labels_opened') is False
            and source_proof.get('capture_sha256') == report['capture_sha256']
            and source_proof.get('report_sha256') == report['capture_report_sha256'],
            'verified source-only capture required')
    for path, sha in capture_report['pins'].items():
        require(digest(Path(path)) == sha, 'capture source pin changed')
    require(digest(intake_path) == intake_sha, 'FIT intake changed')
    intake = _read_json(intake_path)
    require(len(intake['shards']) == expected_shards, 'FIT shard count differs')
    for expected_index, shard in enumerate(intake['shards']):
        require(shard['shard'] == expected_index, 'FIT shard order differs')
        directory = Path(shard['directory'])
        for name, sha in shard['hashes'].items():
            require(digest(directory / name) == sha, 'closed FIT shard changed')
        require(_read_json(directory / 'independent-verification.json').get('verified') is True,
                'FIT scoring proof missing')
    return report, capture, intake


def verify(evaluation, *, intake_path=INTAKE, intake_sha=INTAKE_SHA,
           labels_path=LABELS, labels_sha=LABEL_SHA, evaluator_path=EVALUATOR,
           expected_rows=1642, expected_shards=6, expected_no_pool=13,
           expected_old_scorable=1571):
    evaluation = Path(evaluation)
    proof_path = evaluation / 'independent-verification.json'
    require(not proof_path.exists(), 'metrics proof already exists')
    # This closes every source artifact and source-only proof before the first label read.
    report, capture, intake = _source_preflight(
        evaluation, intake_path=Path(intake_path), intake_sha=intake_sha,
        evaluator_path=Path(evaluator_path), expected_rows=expected_rows,
        expected_shards=expected_shards)

    require(digest(Path(labels_path)) == labels_sha == report['labels_sha256'],
            'FIT labels changed')
    labels = {}
    with Path(labels_path).open() as stream:
        for line in stream:
            row = json.loads(line)
            require(row['id'] not in labels, 'duplicate FIT label')
            labels[row['id']] = row
    require(len(labels) == expected_rows, 'FIT label denominator differs')

    totals = Counter(rows=0, old_scorable_covered=0, old_available_covered=0,
                     expanded_available_covered=0, coverage_gain=0, coverage_loss=0,
                     new_gold_proposals=0, new_proposals=0, no_pool=0)
    seen = set()
    with (capture / 'capture.jsonl').open() as captures, \
            (evaluation / 'outcomes.jsonl').open() as recorded:
        for shard in intake['shards']:
            with (Path(shard['directory']) / 'capture.jsonl').open() as sources:
                for line in sources:
                    source = json.loads(line)
                    proposed_line = next(captures, None)
                    outcome_line = next(recorded, None)
                    require(proposed_line is not None and outcome_line is not None,
                            'evaluation ended before FIT sources')
                    proposed = json.loads(proposed_line)
                    require(source['id'] not in seen and source['id'] in labels,
                            'duplicate or unlabeled FIT row')
                    seen.add(source['id'])
                    result = independent_outcome(source, proposed, labels[source['id']])
                    require(json.loads(outcome_line) == result, 'per-row coverage outcome differs')
                    totals['rows'] += 1
                    for key, value in result.items():
                        if key != 'id':
                            totals[key] += int(value)
        require(next(captures, None) is None and next(recorded, None) is None
                and len(seen) == expected_rows and seen == set(labels),
                'FIT evaluation denominator differs')

    require(totals['old_scorable_covered'] == expected_old_scorable
            and totals['no_pool'] == expected_no_pool
            and totals['coverage_loss'] == 0,
            'frozen baseline or union loss differs')
    require(dict(totals) == report['counts'], 'reported coverage counts differ')

    # Recheck all evidence after the recount to detect changes during label use.
    report_again, capture_again, _ = _source_preflight(
        evaluation, intake_path=Path(intake_path), intake_sha=intake_sha,
        evaluator_path=Path(evaluator_path), expected_rows=expected_rows,
        expected_shards=expected_shards)
    require(report_again == report and capture_again == capture
            and digest(Path(labels_path)) == labels_sha
            and digest(evaluation / 'outcomes.jsonl') == report['outcomes_sha256'],
            'evaluation evidence changed during recount')
    result = {
        'verified': True,
        'rows': expected_rows,
        'counts': dict(totals),
        'report_sha256': digest(evaluation / 'report.json'),
        'outcomes_sha256': digest(evaluation / 'outcomes.jsonl'),
        'capture_sha256': report['capture_sha256'],
        'capture_proof_sha256': report['capture_proof_sha256'],
        'labels_sha256': labels_sha,
        'verifier_sha256': digest(Path(__file__)),
        'scope': ('Independent full-row FIT coverage recount from six frozen scoring shards, '
                  'closed COPY-child proposals, and late labels; no evaluator outcome import, '
                  'model inference, sentence ranking, DEV, or TEST.'),
    }
    with proof_path.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    verify(parser.parse_args().directory)
