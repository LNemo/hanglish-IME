"""Score saved dev generation and its complementarity with displayed candidates.

This is an offline diagnostic, not a selector: reference-dependent union coverage
must never be reported as achievable top-1 accuracy or full internal pool coverage.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sentence_data import DEFAULT_CORPUS, read_split


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def diagnose(rows, predictions, baseline):
    ids = [row['id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate reference ID')
    if [row['id'] for row in predictions] != ids or [row['id'] for row in baseline] != ids:
        raise ValueError('prediction/baseline coverage or ordering differs')
    outcomes = []
    for row, prediction, previous in zip(rows, predictions, baseline):
        if set(prediction) != {'id', 'input', 'response'}:
            raise ValueError('prediction carries unexpected fields')
        if prediction['input'] != row['input'] or previous['request'] != dict(input=row['input'], context=''):
            raise ValueError('source-only request differs')
        response = prediction['response']
        if set(response) != {'text', 'reason', 'generated_tokens'}:
            raise ValueError('generation response schema differs')
        if response['reason'] not in {'complete', 'token_limit', 'invalid_encoding'}:
            raise ValueError('unknown generation outcome')
        tokens = response['generated_tokens']
        if type(tokens) is not int or not 0 <= tokens <= 512:
            raise ValueError('invalid generation length')
        text = response['text']
        if text is not None and not isinstance(text, str):
            raise ValueError('invalid generated text')
        complete = response['reason'] == 'complete'
        if complete and text is None:
            raise ValueError('completed generation without text')
        if not complete and text is not None:
            raise ValueError('incomplete generation carries decoded text')
        if response['reason'] == 'token_limit' and tokens != 512:
            raise ValueError('token limit differs from model contract')
        if complete and tokens != sum(3 if '가' <= char <= '힣' else len(char.encode('utf-8')) for char in text):
            raise ValueError('generated token count differs from decoded text')
        generated = text if complete else None
        reference = min(row['expected'], key=lambda value: (edit_distance(generated or '', value), value))
        candidates = {item['text'] for item in previous['mixed']['candidates']}
        baseline_top1 = previous['mixed']['result'] in row['expected']
        baseline_displayed = bool(candidates.intersection(row['expected']))
        correct = generated in row['expected']
        # Ordered literal sequences against accepted references; this measures
        # output fidelity, not a claim that a learned copy mechanism exists.
        latin = lambda value: re.findall(r'[A-Za-z]+', value)
        numbers = lambda value: re.findall(r'\d+(?:\.\d+)*', value)
        outcomes.append(dict(
            id=row['id'], input=row['input'], generated=text, expected=row['expected'],
            reason=response['reason'], correct=correct, baseline_top1=baseline_top1,
            baseline_displayed=baseline_displayed,
            displayed_union=baseline_displayed or correct,
            new_correct_candidate=correct and not baseline_displayed,
            edit_distance=edit_distance(generated or '', reference), reference_characters=len(reference),
            latin_exact=complete and any(latin(generated) == latin(ref) for ref in row['expected']),
            numbers_exact=complete and any(numbers(generated) == numbers(ref) for ref in row['expected']),
            has_latin=any(latin(ref) for ref in row['expected']),
            has_numbers=any(numbers(ref) for ref in row['expected']),
        ))
    totals = {key: sum(item[key] for item in outcomes) for key in (
        'correct', 'baseline_top1', 'baseline_displayed', 'displayed_union',
        'new_correct_candidate', 'edit_distance', 'reference_characters')}
    totals['character_error_rate'] = totals['edit_distance'] / max(1, totals['reference_characters'])
    totals['reasons'] = dict(Counter(item['reason'] for item in outcomes))
    for kind in ('latin', 'numbers'):
        selected = [item for item in outcomes if item['has_' + kind]]
        totals[kind] = dict(rows=len(selected), exact=sum(item[kind + '_exact'] for item in selected))
    return dict(rows=len(rows), metrics=totals, outcomes=outcomes,
                union_is_oracle_coverage_not_top1=True, production_changed=False)


def run(run_dir, baseline_dir, output):
    training_report = json.loads((run_dir / 'report.json').read_text())
    baseline_report = json.loads((baseline_dir / 'report.json').read_text())
    prediction_path = run_dir / 'dev-predictions.jsonl'
    capture_path = baseline_dir / 'capture.jsonl'
    dev_path = DEFAULT_CORPUS / 'dev.jsonl'
    if (not training_report['completed'] or not baseline_report['completed']
            or baseline_report['split'] != 'dev'
            or sha(prediction_path) != training_report['predictions_sha256']
            or sha(capture_path) != baseline_report['capture_sha256']
            or training_report['pins'].get(str(dev_path)) != sha(dev_path)
            or baseline_report['pins'].get(str(dev_path)) != sha(dev_path)):
        raise ValueError('unverified or mismatched saved captures')
    read_jsonl = lambda path: [json.loads(line) for line in path.read_text().splitlines()]
    report = diagnose(read_split(DEFAULT_CORPUS, 'dev'), read_jsonl(prediction_path), read_jsonl(capture_path))
    report['pins'] = {str(path): sha(path) for path in (
        Path(__file__), prediction_path, capture_path, dev_path, run_dir / 'report.json', baseline_dir / 'report.json')}
    report['baseline_version'] = baseline_report['mixed_version']
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report['metrics'], ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.run, args.baseline, args.output)
