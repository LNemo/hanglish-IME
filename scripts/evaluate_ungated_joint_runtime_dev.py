"""Join DEV labels only after source-only runtime recovery is closed."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.capture_ungated_joint_dev import (
    COMPARATOR, PARENT_VERSION, authenticate_sources, need, read, sha, verify_pins, write,
)
from scripts.evaluate_current_phonetic_choice_dev import ascii_words, literal_signature

LABELS = ROOT / 'data/challenge/sentence-v3/dev.jsonl'
LABEL_SHA = '593e1a032b54bf4d2c541da6754cfc68daf759cdcee7f8fa1f0a746459910ac9'


def row_metrics(source, public, expected):
    texts = [row['text'] for row in public]
    need(1 <= len(texts) <= 5 and len(set(texts)) == len(texts), 'public candidates differ')
    english = {ascii_words(text) for text in expected}
    has_english = any(english)
    return {'first': texts[0] in expected, 'top5': bool(set(texts) & set(expected)),
            'ascii_reference': bool(has_english),
            'ascii_exact': bool(has_english and ascii_words(texts[0]) in english),
            'literal_exact': literal_signature(texts[0]) == literal_signature(source)}


def compare_metrics(before, after):
    return {'corrections': int(after['first'] and not before['first']),
            'regressions': int(before['first'] and not after['first']),
            'top5_gains': int(after['top5'] and not before['top5']),
            'top5_losses': int(before['top5'] and not after['top5']),
            'ascii_regressions': int(before['ascii_exact'] and not after['ascii_exact']),
            'literal_regressions': int(before['literal_exact'] and not after['literal_exact'])}


def authenticate_capture(directory):
    authenticate_sources()
    report, closure = read(directory / 'report.json'), read(directory / 'capture.complete.json')
    need(report.get('completed') is True and report.get('rows') == 316
         and report.get('selected_rows') == 4 and report.get('labels_opened') is False
         and report.get('production_changed') is False
         and report.get('parent_version') == PARENT_VERSION
         and closure.get('completed') is True and closure.get('rows') == 316
         and closure.get('selected_rows') == 4 and closure.get('labels_opened') is False
         and closure.get('report_sha256') == sha(directory / 'report.json')
         and closure.get('capture_sha256') == report.get('capture_sha256')
             == sha(directory / 'capture.jsonl')
         and report.get('resources_sha256') == sha(directory / 'resources.jsonl'),
         'closed source-only capture required')
    verify_pins(report['pins'])
    rows = [json.loads(line) for line in (directory / 'capture.jsonl').open()]
    need(len(rows) == len({row['id'] for row in rows}) == 316, 'capture row identity differs')
    references = {}
    with (COMPARATOR / 'capture.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            need(row['id'] not in references, 'duplicate production row')
            references[row['id']] = row
    need(set(references) == {row['id'] for row in rows}, 'production source coverage differs')
    selected = []
    for row in rows:
        fixed = references[row['id']]
        need(row['request'] == fixed['request'] and fixed['response']['version'] == PARENT_VERSION,
             'production request/version differs')
        if row['selected_unranked']:
            selected.append(row['id'])
            need(not fixed['response']['sentence_ranking_applied']
                 and row['controls']['parent_candidates_exact_cached'] is True
                 and row['fresh_baseline']['candidates'] == fixed['response']['candidates']
                 and row['trace']['baseline'] == row['fresh_baseline'],
                 'selected baseline control differs')
            pool = row['trace']['pool']
            supported = sorted((item for item in pool if item['supported']),
                               key=lambda item: (-item['score'], item['text']))
            if supported:
                expected = [item['text'] for item in supported[:4]] + [row['request']['input']]
                need([item['text'] for item in row['response']['candidates']] == expected,
                     'selected public ordering differs')
            else:
                need(row['response'] == row['fresh_baseline'], 'unscorable fallback changed')
            need(all(item['score'] is None for item in pool if not item['supported']),
                 'unsupported candidate acquired score')
        else:
            need(fixed['response']['sentence_ranking_applied'] and row['passthrough_exact'],
                 'ranked passthrough control differs')
    need(selected == report['selected_ids'] and len(selected) == 4,
         'selected source coverage differs')
    return report, rows, references


def evaluate(directory):
    report, rows, references = authenticate_capture(directory)
    # No label access above: all source controls close before reading expected outputs.
    need(sha(LABELS) == LABEL_SHA, 'DEV labels changed')
    labels = {row['id']: row for row in (json.loads(line) for line in LABELS.open())}
    need(len(labels) == 316 and set(labels) == set(references), 'DEV coverage differs')
    branches = {name: Counter(rows=0) for name in ('baseline', 'recovered')}
    transitions = Counter()
    outcomes = []
    for row in rows:
        label = labels[row['id']]
        need(row['request'] == {'input': label['input'], 'context': ''}
             and label['split'] == 'dev' and label['expected'], 'DEV label identity differs')
        before = references[row['id']]['response']
        after = row['response'] if row['selected_unranked'] else before
        metrics = {name: row_metrics(label['input'], response['candidates'], label['expected'])
                   for name, response in [('baseline', before), ('recovered', after)]}
        for name, values in metrics.items():
            branches[name].update({'rows': 1, **{key: int(value) for key, value in values.items()}})
        change = compare_metrics(metrics['baseline'], metrics['recovered'])
        transitions.update(change)
        outcomes.append({'id': row['id'], 'selected_unranked': row['selected_unranked'],
                         'metrics': metrics, 'transitions': change})
    closing, _, _ = authenticate_capture(directory)
    need(closing == report and sha(LABELS) == LABEL_SHA, 'evaluation inputs changed')
    with (directory / 'outcomes.jsonl').open('x') as stream:
        for row in outcomes:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    result = {'completed': True, 'rows': 316, 'selected_rows': 4,
              'branches': {name: dict(value) for name, value in branches.items()},
              'transitions': dict(transitions),
              'quality_gate': branches['recovered']['first'] > branches['baseline']['first']
                  and not any(transitions[key] for key in
                              ['regressions', 'top5_losses', 'ascii_regressions', 'literal_regressions']),
              'capture_sha256': sha(directory / 'capture.jsonl'),
              'capture_report_sha256': sha(directory / 'report.json'),
              'labels_sha256': LABEL_SHA, 'outcomes_sha256': sha(directory / 'outcomes.jsonl'),
              'evaluator_sha256': sha(Path(__file__)), 'labels_opened_after_closed_capture': True,
              'production_changed': False, 'test_opened': False,
              'scope': 'Previously exposed DEV;312 cached exact passthrough controls and4 fresh runtime rows.'}
    write(directory / 'evaluation.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    evaluate(parser.parse_args().directory)
