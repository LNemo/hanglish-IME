"""Evaluate a side-by-side natural ranker through the actual hybrid runtime."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_natural_ranker import validate_prepared, file_sha256, OUTPUT
from hanglish.core.neural_converter import NeuralConverter
from hanglish.models.sentence_ranker import SentenceRanker


def evaluate(checkpoint, output, *, final=False):
    if output.exists():
        raise FileExistsError(output)
    candidate_ranker = SentenceRanker.load(checkpoint)
    provenance = candidate_ranker.metadata['natural_preparation']
    _, prepared = validate_prepared(Path(provenance['path']), provenance['manifest_sha256'])
    training_report = json.loads((checkpoint.parent / 'report.json').read_text())
    if training_report['final']['correct'] <= training_report['initial']['correct']:
        raise ValueError('natural development improvement gate not met')
    if final:
        gate = json.loads((checkpoint.parent / 'runtime-dev.json').read_text())
        if not gate['passed'] or gate['checkpoint_sha256'] != file_sha256(checkpoint / 'model.safetensors'):
            raise ValueError('actual runtime development gate not met')
    converter = NeuralConverter(ranker_kind='hybrid')
    converter.load_models()
    parent_version = converter.version
    converter.sentence_ranker = candidate_ranker
    checkpoint_hash = file_sha256(checkpoint / 'model.safetensors')
    converter.version = parent_version + '-sentence-' + checkpoint_hash[:12]
    converter.model_artifacts = {key: value for key, value in converter.model_artifacts.items()
                                 if not key.startswith('checkpoints/sentence_ranker/')}
    converter.model_artifacts.update({str(path.relative_to(ROOT)): file_sha256(path)
                                      for path in checkpoint.iterdir() if path.is_file()})
    artifact_root = ROOT / '.omx/artifacts'
    if final:
        suites = [('final_v22', ROOT / 'data/challenge/v2.2/test.jsonl',
                   artifact_root / 'challenge-v22-test-numeric-v2.json')]
    else:
        suites = [('original_dev', ROOT / 'data/challenge/dev.jsonl', artifact_root / 'challenge-dev-numeric-v2.json'),
                  ('selection_dev', ROOT / 'data/challenge/selection-dev-v1.1/cases.jsonl', artifact_root / 'selection-dev-numeric-v2.json'),
                  ('reasoning_dev', ROOT / 'data/challenge/reasoning-dev-v1.1/cases.jsonl', artifact_root / 'reasoning-dev-numeric-v2.json')]
    sources = {}
    results = {}
    all_outcomes = []
    for name, path, baseline_path in suites:
        raw = path.read_bytes()
        cases = [json.loads(line) for line in raw.splitlines() if line]
        baseline = json.loads(baseline_path.read_text())
        baseline_hash = baseline.get('sha256') or baseline['corpus_identity']['sha256']
        if file_sha256(path) != baseline_hash:
            raise ValueError('development/final source differs from frozen baseline')
        sources[name] = {'path': str(path), 'sha256': baseline_hash,
                         'baseline_report_sha256': file_sha256(baseline_path)}
        old = {row['id']: row for row in baseline['outcomes']}
        if set(old) != {r['id'] for r in cases}:
            raise ValueError('baseline IDs differ')
        outcomes = []
        for case in cases:
            previous = old[case['id']]
            if previous['version'] != parent_version or any(
                previous[k] != case[k] for k in ('input', 'context', 'expected')):
                raise ValueError('baseline version or cases differ')
            converted = converter.convert(case['input'], case['context'])
            actual = [c['text'] for c in converted['candidates']]
            outcomes.append({'id': case['id'], 'suite': name, 'input': case['input'],
                'context': case['context'], 'expected': case['expected'], 'actual': actual,
                'top1': actual[0] in case['expected'], 'top5': any(c in case['expected'] for c in actual),
                'before_top1': previous['top1'], 'before_actual': previous['actual'][0]})
        if path.read_bytes() != raw:
            raise ValueError('source changed during evaluation')
        results[name] = summarize(outcomes)
        all_outcomes.extend(outcomes)
        print(json.dumps({'suite': name, **results[name]}, ensure_ascii=False), flush=True)
    natural_parity = None
    if not final:
        expected_outputs = {row['id']: row for row in training_report['final']['outcomes']}
        natural = []
        for row in prepared['natural_dev']:
            actual = converter.convert(row['input'], row['context'])['result']
            natural.append({'id': row['id'], 'result': actual, 'correct': actual in row['expected'],
                            'same_as_training_evaluator': actual == expected_outputs[row['id']]['result']})
        natural_parity = {'rows': len(natural), 'correct': sum(r['correct'] for r in natural),
                          'all_predictions_identical': all(r['same_as_training_evaluator'] for r in natural),
                          'outcomes': natural}
    passed = all(not suite['regressions'] for suite in results.values())
    if natural_parity is not None:
        passed &= natural_parity['all_predictions_identical']
    report = {'passed': passed, 'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_hash,
              'candidate_pool_version': parent_version, 'experimental_version': converter.version,
              'model_artifacts': converter.model_artifacts,
              'scope': 'exposed final regression' if final else 'actual hybrid development',
              'sources': sources, 'suites': results, 'natural_runtime_parity': natural_parity,
              'outcomes': all_outcomes, 'deployed': False}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passed': passed, 'suites': results,
                      'natural_runtime_parity': None if natural_parity is None else
                      {k: v for k, v in natural_parity.items() if k != 'outcomes'}}, ensure_ascii=False), flush=True)


def summarize(outcomes):
    return {'rows': len(outcomes), 'before_top1': sum(r['before_top1'] for r in outcomes),
            'top1': sum(r['top1'] for r in outcomes), 'top5': sum(r['top5'] for r in outcomes),
            'corrections': [r['id'] for r in outcomes if r['top1'] and not r['before_top1']],
            'regressions': [r['id'] for r in outcomes if not r['top1'] and r['before_top1']]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    evaluate(args.checkpoint.resolve(), args.output, final=args.final)
