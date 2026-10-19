"""Gate a supervised RoBERTa word ranker on fresh complete hybrid inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.models.semantic_ranker import SemanticRanker
from scripts.prepare_context_training import validate_context_data, source_hashes, RUNTIME_VERSION
from scripts.evaluate_natural_ranker import summarize

DEV_SUITES = (
    ('original_dev', 'data/challenge/dev.jsonl', 'challenge-dev-natural-v3-deployed.json'),
    ('selection_dev', 'data/challenge/selection-dev-v1.1/cases.jsonl', 'selection-dev-natural-v3-deployed.json'),
    ('reasoning_dev', 'data/challenge/reasoning-dev-v1.1/cases.jsonl', 'reasoning-dev-natural-v3-deployed.json'))
FINAL_SUITE = ('final_v22', 'data/challenge/v2.2/test.jsonl', 'challenge-v22-test-natural-v3-deployed.json')
DEV_COUNTS = {'natural_dev': 128, 'original_dev': 140, 'selection_dev': 60, 'reasoning_dev': 40}


class SentencePoolRecorder:
    """Observe candidates actually passed to sentence scoring without another inference."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.candidates = None

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def score_pairs(self, contexts, candidates):
        self.candidates = list(candidates)
        return self.delegate.score_pairs(contexts, candidates)


def evaluation_hashes():
    return {relative: file_sha256(ROOT / relative) for relative in
            ('scripts/evaluate_roberta_context_ranker.py', 'scripts/evaluate_natural_ranker.py')}


def require_development_gate(gate, checkpoint_hashes, training_sha, sources, code_hashes):
    if (gate.get('passed') is not True or gate.get('checkpoint_hashes') != checkpoint_hashes
            or gate.get('training_report_sha256') != training_sha
            or gate.get('source_hashes') != sources or gate.get('parent_version') != RUNTIME_VERSION
            or gate.get('evaluation_code_hashes') != code_hashes
            or gate.get('scope') != 'actual hybrid development'
            or set(gate.get('suites', {})) != set(DEV_COUNTS)):
        raise ValueError('current full hybrid development gate required')
    for name, count in DEV_COUNTS.items():
        suite = gate['suites'][name]
        if suite.get('rows') != count or suite.get('regressions') != []:
            raise ValueError('complete regression-free development required')
    natural = gate['suites']['natural_dev']
    if natural['top1'] <= natural['before_top1']:
        raise ValueError('actual natural development improvement required')


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = args.checkpoint.resolve()
    ranker = SemanticRanker.load(checkpoint)
    provenance = ranker.metadata['context_preparation']
    _, data = validate_context_data(Path(provenance['path']), provenance['manifest_sha256'])
    before = source_hashes()
    code_hashes = evaluation_hashes()
    checkpoint_hashes = {path.name: file_sha256(path) for path in checkpoint.iterdir() if path.is_file()}
    training_path = checkpoint.parent / 'report.json'
    training_sha = file_sha256(training_path)
    training = json.loads(training_path.read_text())
    if (training['checkpoint_sha256'] != checkpoint_hashes['model.safetensors']
            or training['final']['correct'] <= training['initial']['correct']):
        raise ValueError('natural word development improvement gate not met')
    gate_sha = None
    if args.final:
        gate_path = checkpoint.parent / 'runtime-dev.json'
        gate_sha = file_sha256(gate_path)
        gate = json.loads(gate_path.read_text())
        require_development_gate(gate, checkpoint_hashes, training_sha, before, code_hashes)
    converter = NeuralConverter(ranker_kind='hybrid')
    converter.load_models()
    if converter.version != RUNTIME_VERSION or ranker.metadata['parent_runtime_version'] != RUNTIME_VERSION:
        raise ValueError('runtime differs from pinned training parent')
    parent = converter.ranker
    sentence_recorder = SentencePoolRecorder(converter.sentence_ranker)
    converter.sentence_ranker = sentence_recorder
    results, outcomes, sources = {}, [], {}
    started = time.monotonic()

    def run_suite(name, cases):
        current = []
        for index, case in enumerate(cases):
            converter.ranker = parent
            sentence_recorder.candidates = None
            previous = converter.convert(case['input'], case['context'])
            previous_pool = sentence_recorder.candidates
            converter.ranker = ranker
            sentence_recorder.candidates = None
            converted = converter.convert(case['input'], case['context'])
            pool = sentence_recorder.candidates
            actual = [candidate['text'] for candidate in converted['candidates']]
            current.append({'id': case['id'], 'suite': name, 'input': case['input'],
                'context': case['context'], 'expected': case['expected'], 'actual': actual,
                'top1': actual[0] in case['expected'], 'top5': any(c in case['expected'] for c in actual),
                'before_top1': previous['result'] in case['expected'], 'before_actual': previous['result'],
                'before_sentence_pool_covered': None if previous_pool is None else any(c in case['expected'] for c in previous_pool),
                'sentence_pool_covered': None if pool is None else any(c in case['expected'] for c in pool),
                'before_sentence_pool_size': None if previous_pool is None else len(previous_pool),
                'sentence_pool_size': None if pool is None else len(pool),
                'before_engine_ms': previous['engine_ms'], 'engine_ms': converted['engine_ms']})
            if (index + 1) % 32 == 0:
                print(json.dumps({'suite': name, 'cases': index + 1,
                                  'elapsed_seconds': round(time.monotonic()-started, 2)}), flush=True)
        results[name] = summarize(current)
        results[name]['sentence_scored_rows'] = sum(r['sentence_pool_covered'] is not None for r in current)
        results[name]['before_sentence_scored_rows'] = sum(r['before_sentence_pool_covered'] is not None for r in current)
        results[name]['before_sentence_pool_covered'] = sum(r['before_sentence_pool_covered'] is True for r in current)
        results[name]['sentence_pool_covered'] = sum(r['sentence_pool_covered'] is True for r in current)
        outcomes.extend(current)
        print(json.dumps({'suite': name, **results[name]}, ensure_ascii=False), flush=True)

    if not args.final:
        run_suite('natural_dev', data['natural_dev'])
        sources['natural_dev'] = {'manifest_sha256': provenance['manifest_sha256']}
    for name, relative, baseline_name in ([FINAL_SUITE] if args.final else DEV_SUITES):
        path = ROOT / relative
        baseline_path = ROOT / '.omx/artifacts' / baseline_name
        baseline = json.loads(baseline_path.read_text())
        expected_hash = baseline.get('sha256') or baseline['corpus_identity']['sha256']
        if file_sha256(path) != expected_hash:
            raise ValueError('evaluation suite differs from frozen source')
        cases = [json.loads(line) for line in path.read_text().splitlines() if line]
        run_suite(name, cases)
        if file_sha256(path) != expected_hash:
            raise ValueError('evaluation source changed')
        sources[name] = {'sha256': expected_hash, 'identity_report_sha256': file_sha256(baseline_path),
                         'baseline': 'fresh current-parent inference; historical predictions unused'}
    validate_context_data(Path(provenance['path']), provenance['manifest_sha256'])
    if (before != source_hashes() or code_hashes != evaluation_hashes() or file_sha256(training_path) != training_sha
            or checkpoint_hashes != {p.name: file_sha256(p) for p in checkpoint.iterdir() if p.is_file()}):
        raise ValueError('evaluation inputs changed')
    passed = all(not result['regressions'] for result in results.values())
    if not args.final:
        passed &= results['natural_dev']['top1'] > results['natural_dev']['before_top1']
    report = {'passed': bool(passed), 'parent_version': RUNTIME_VERSION,
              'experimental_version': RUNTIME_VERSION + '-context-' + checkpoint_hashes['model.safetensors'][:12],
              'base_model': ranker.metadata.get('base_model'),
              'checkpoint': str(checkpoint), 'checkpoint_hashes': checkpoint_hashes,
              'training_report_sha256': training_sha, 'development_gate_sha256': gate_sha,
              'evaluation_code_hashes': code_hashes, 'source_hashes': before,
              'sources': sources, 'suites': results, 'outcomes': outcomes,
              'scope': 'exposed final regression' if args.final else 'actual hybrid development',
              'timing_policy': 'Parent runs first with shared candidate caches; recorded times are descriptive, not a controlled speed comparison.',
              'elapsed_seconds': time.monotonic()-started, 'deployed': False}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passed': bool(passed), 'suites': results}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    evaluate(parser.parse_args())
