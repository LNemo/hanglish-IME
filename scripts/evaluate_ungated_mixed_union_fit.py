"""Late FIT metrics for the verified joint + mixed candidate union."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.score_ungated_joint_fit import LABELS, LABEL_SHA, need, read, sha, verify_pins, write
from scripts.evaluate_current_phonetic_choice_dev import ascii_words, literal_signature

ARMS = ('baseline', 'mixed_union')


def edit_distance(left, right):
    """Unicode code point distance, including whitespace and literal content."""
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j-1] + (a != b)))
        previous = current
    return previous[-1]


def metrics(source, public, pool, supported, expected):
    texts = [p['text'] for p in public]
    need(1 <= len(texts) <= 5 and len(set(texts)) == len(texts), 'public output differs')
    need(expected and all(isinstance(text, str) for text in expected), 'missing FIT answers')
    gold = set(expected)
    english = {ascii_words(text) for text in gold}
    return {'first': texts[0] in gold, 'top5': bool(set(texts) & gold),
            'all_pool_covered': bool({p['text'] for p in pool} & gold),
            'scorable_pool_covered': bool(set(supported) & gold),
            'ascii_reference': bool(any(english)),
            'ascii_exact': bool(any(english) and ascii_words(texts[0]) in english),
            'literal_exact': literal_signature(texts[0]) == literal_signature(source),
            'edit_distance': min(edit_distance(texts[0], answer) for answer in gold)}


def transitions(before, after):
    return {'corrections': int(not before['first'] and after['first']),
            'regressions': int(before['first'] and not after['first']),
            'top5_gains': int(not before['top5'] and after['top5']),
            'top5_losses': int(before['top5'] and not after['top5']),
            'ascii_regressions': int(before['ascii_exact'] and not after['ascii_exact']),
            'literal_regressions': int(before['literal_exact'] and not after['literal_exact']),
            'partial_improvements': int(after['edit_distance'] < before['edit_distance']),
            'partial_regressions': int(after['edit_distance'] > before['edit_distance']),
            'edit_distance_delta': after['edit_distance'] - before['edit_distance']}


def evaluate_row(row, label):
    source = row['request']['input']
    need(row['id'] == label['id'] and row['request'] == {'input': label['input'], 'context': ''},
         'FIT identity differs')
    branches = {}
    for arm in ARMS:
        branch = row['arms'][arm]
        branches[arm] = metrics(source, branch['public'], branch['pool'],
                                branch['supported_order'], label['expected'])
    fallback = row['fallback_public']
    branches['fallback'] = metrics(source, fallback, fallback, [], label['expected'])
    return {'id': row['id'], 'branches': branches,
            'transitions': {name: transitions(branches[before], branches['mixed_union'])
                            for name, before in [('baseline_to_union', 'baseline'),
                                                 ('fallback_to_union', 'fallback')]}}


def authenticate(directory):
    report = read(directory / 'report.json')
    closure = read(directory / 'capture.complete.json')
    proof = read(directory / 'independent-verification.json')
    need(report.get('completed') is True and report.get('rows') == 13
         and report.get('labels_opened') is False
         and report.get('production_changed') is False
         and closure.get('completed') is True and closure.get('rows') == 13
         and proof.get('verified') is True and proof.get('rows') == 13
         and proof.get('report_sha256') == closure.get('report_sha256') == sha(directory / 'report.json')
         and proof.get('capture_sha256') == closure.get('capture_sha256') == sha(directory / 'capture.jsonl'),
         'closed independently verified score capture required')
    verify_pins(report['pins'])
    return report


def evaluate(directory):
    report = authenticate(directory)
    rows = [json.loads(line) for line in (directory / 'capture.jsonl').open()]
    ids = {r['id'] for r in rows}
    need(len(rows) == len(ids) == 13, 'FIT capture denominator differs')
    need(sha(LABELS) == LABEL_SHA, 'FIT labels changed')
    labels = {}
    with LABELS.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['id'] in ids:
                need(row['id'] not in labels, 'duplicate FIT label')
                labels[row['id']] = {key: row[key] for key in ('id', 'input', 'expected')}
    need(set(labels) == ids, 'FIT label coverage differs')
    outcomes = [evaluate_row(row, labels[row['id']]) for row in rows]
    branches = {name: Counter(rows=0) for name in (*ARMS, 'fallback')}
    changes = {name: Counter() for name in ('baseline_to_union', 'fallback_to_union')}
    for row in outcomes:
        for name, values in row['branches'].items():
            branches[name].update({'rows': 1, **{k: int(v) for k, v in values.items()}})
        for name, values in row['transitions'].items(): changes[name].update(values)
    need(authenticate(directory) == report and sha(LABELS) == LABEL_SHA, 'evaluation inputs changed')
    with (directory / 'outcomes.jsonl').open('x') as stream:
        for row in outcomes: stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    result = {'completed': True, 'rows': 13,
              'branches': {k: dict(v) for k,v in branches.items()},
              'transitions': {k: dict(v) for k,v in changes.items()},
              'score_report_sha256': sha(directory / 'report.json'),
              'score_capture_sha256': sha(directory / 'capture.jsonl'),
              'score_proof_sha256': sha(directory / 'independent-verification.json'),
              'labels_sha256': LABEL_SHA, 'outcomes_sha256': sha(directory / 'outcomes.jsonl'),
              'evaluator_sha256': sha(Path(__file__)), 'labels_opened_after_verified_closure': True,
              'production_changed': False, 'dev_opened': False, 'test_opened': False,
              'scope': 'Previously exposed13 source-selected unranked FIT rows; no generalization or deployment claim.'}
    write(directory / 'evaluation.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    evaluate(parser.parse_args().directory)
