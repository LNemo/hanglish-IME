"""Late FIT gold availability for preserved full pool plus mixed-span proposals."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capture_ungated_mixed_boundary_fit import need, read, sha, verify_pins, write

FIT = ROOT / '.omx/artifacts/current-fit-recombination-coverage-v1'
LABELS = ROOT / '.omx/artifacts/current-phonetic-train-features-v1/rows.jsonl'
LABEL_SHA = '4c13063656d9a108316b9ce1ceec394a61e2c45c2ba86e7a8c8361f7480d6cb4'


def coverage(before, proposals, expected):
    gold = set(expected)
    extra = [p for p in proposals if p['text'] in gold]
    supported = any(p['token_support']['supported'] for p in extra)
    return {'old_full_pool_covered': bool(before), 'proposal_gold_present': bool(extra),
            'supported_proposal_gold_present': supported,
            'union_covered': bool(before or extra),
            'new_gold_row': bool(not before and extra),
            'new_supported_gold_row': bool(not before and supported)}


def authenticate(directory):
    report = read(directory / 'report.json')
    closure = read(directory / 'capture.complete.json')
    proof = read(directory / 'independent-verification.json')
    need(report.get('completed') is True and report.get('counts', {}).get('rows') == 66
         and report.get('labels_opened') is False
         and report.get('sentence_scoring_run') is False
         and closure.get('completed') is True and closure.get('rows') == 66
         and closure.get('report_sha256') == sha(directory / 'report.json')
         and closure.get('capture_sha256') == report.get('capture_sha256')
             == sha(directory / 'capture.jsonl')
         and proof.get('verified') is True and proof.get('rows') == 66
         and proof.get('report_sha256') == sha(directory / 'report.json')
         and proof.get('capture_sha256') == report['capture_sha256'],
         'closed independently verified proposal capture required')
    verify_pins(report['pins'])
    return report


def evaluate(directory):
    report = authenticate(directory)
    fit_proof = read(FIT / 'independent-verification.json')
    fit_report = read(FIT / 'report.json')
    fit_closure = read(FIT / 'completion.json')
    need(fit_proof.get('verified') is True
         and fit_report.get('completed') is True and fit_closure.get('completed') is True
         and fit_proof.get('report_sha256') == fit_closure.get('report_sha256')
             == sha(FIT / 'report.json')
         and fit_proof.get('rows_sha256') == fit_report.get('rows_sha256')
             == fit_closure.get('rows_sha256') == sha(FIT / 'rows.jsonl'),
         'verified full FIT coverage required')
    need(sha(LABELS) == LABEL_SHA, 'FIT labels changed')
    labels = {}
    with LABELS.open() as stream:
        for line in stream:
            row = json.loads(line)
            need(row['id'] not in labels, 'duplicate FIT label')
            labels[row['id']] = {'input': row['input'], 'expected': row['expected']}
    metrics = {row['id']: row for row in (json.loads(line) for line in (FIT / 'rows.jsonl').open())}
    need(len(labels) == len(metrics) == 1642 and set(labels) == set(metrics)
         and sum(row['expanded_pool_covered'] for row in metrics.values()) == 1571,
         'full FIT denominator/baseline differs')
    counts, outcomes, seen = Counter(), [], set()
    with (directory / 'capture.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            need(row['id'] not in seen and row['id'] in labels, 'capture FIT identity differs')
            seen.add(row['id'])
            label = labels[row['id']]
            need(row['request'] == {'input': label['input'], 'context': ''}, 'FIT input differs')
            result = coverage(metrics[row['id']]['expanded_pool_covered'], row['proposals'], label['expected'])
            counts.update({'rows': 1, **{key: int(value) for key,value in result.items()}})
            outcomes.append({'id': row['id'], **result})
    need(len(seen) == 66 and counts['old_full_pool_covered'] == 47,
         'source-selected FIT denominator differs')
    need(authenticate(directory) == report and sha(LABELS) == LABEL_SHA,
         'inputs changed during evaluation')
    with (directory / 'outcomes.jsonl').open('x') as stream:
        for row in outcomes: stream.write(json.dumps(row) + '\n')
    result = {'completed': True, 'selected_rows': 66, 'selected_counts': dict(counts),
              'full_fit': {'rows': 1642, 'before_pool_covered': 1571,
                           'after_pool_covered': 1571 + counts['new_gold_row'],
                           'new_supported_gold_rows': counts['new_supported_gold_row']},
              'report_sha256': sha(directory / 'report.json'),
              'capture_sha256': sha(directory / 'capture.jsonl'),
              'proof_sha256': sha(directory / 'independent-verification.json'),
              'fit_baseline_sha256': sha(FIT / 'rows.jsonl'),
              'fit_baseline': str(FIT.relative_to(ROOT)), 'labels_sha256': LABEL_SHA,
              'outcomes_sha256': sha(directory / 'outcomes.jsonl'),
              'evaluator_sha256': sha(Path(__file__)),
              'first_top5_evaluated': False, 'dev_opened': False, 'test_opened': False,
              'production_changed': False,
              'scope': 'Previously exposed FIT; availability only, full original pool preserved.'}
    write(directory / 'evaluation.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    evaluate(parser.parse_args().directory)
