"""Compare a word-ranker candidate through the complete deployed hybrid pipeline."""
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
from scripts.prepare_natural_word_ranker import validate_word_data, source_hashes
from scripts.evaluate_natural_ranker import summarize


def evaluate(checkpoint, output, final=False):
    if output.exists(): raise FileExistsError(output)
    ranker = SemanticRanker.load(checkpoint)
    provenance = ranker.metadata['natural_word_preparation']
    _, data = validate_word_data(Path(provenance['path']), provenance['manifest_sha256'])
    before = source_hashes()
    checkpoint_hash = file_sha256(checkpoint / 'model.safetensors')
    training_path = checkpoint.parent / 'report.json'
    training = json.loads(training_path.read_text())
    if training['checkpoint_sha256'] != checkpoint_hash or training['final']['correct'] <= training['initial']['correct']:
        raise ValueError('word-level natural development improvement gate not met')
    gate_hash = None
    if final:
        gate_path = checkpoint.parent / 'runtime-dev.json'
        gate = json.loads(gate_path.read_text())
        if (not gate['passed'] or gate['checkpoint_sha256'] != checkpoint_hash
                or gate['training_report_sha256'] != file_sha256(training_path)):
            raise ValueError('actual hybrid development gate not met')
        gate_hash = file_sha256(gate_path)
    converter = NeuralConverter(ranker_kind='hybrid'); converter.load_models()
    parent_version = converter.version
    if parent_version != ranker.metadata['parent_runtime_version']:
        raise ValueError('production version differs from training baseline')
    parent_ranker = converter.ranker
    experimental_version = parent_version + '-word-' + checkpoint_hash[:12]
    artifacts = {key:value for key,value in converter.model_artifacts.items()
                 if not key.startswith('checkpoints/semantic_ranker/')}
    artifacts.update({str(p.relative_to(ROOT)): file_sha256(p) for p in checkpoint.iterdir() if p.is_file()})
    converter.model_artifacts = artifacts
    converter.version = experimental_version
    results, outcomes, sources = {}, [], {}
    started = time.monotonic()
    def run_case(case, previous, suite):
        converter.ranker = ranker
        converted = converter.convert(case['input'], case['context'])
        actual = [candidate['text'] for candidate in converted['candidates']]
        return {'id': case['id'], 'suite': suite, 'input': case['input'], 'context': case['context'],
                'expected': case['expected'], 'actual': actual,
                'top1': actual[0] in case['expected'], 'top5': any(c in case['expected'] for c in actual),
                'before_top1': previous['top1'], 'before_actual': previous['actual'][0],
                'version': experimental_version, 'engine_ms': converted['engine_ms']}
    if not final:
        natural = []
        for index, case in enumerate(data['natural_dev']):
            converter.ranker = parent_ranker
            old = converter.convert(case['input'], case['context'])
            previous = {'top1': old['result'] in case['expected'], 'actual': [old['result']]}
            natural.append(run_case(case, previous, 'natural_dev'))
            if (index + 1) % 32 == 0:
                print(json.dumps({'suite': 'natural_dev', 'cases': index+1, 'elapsed_seconds': time.monotonic()-started}), flush=True)
        results['natural_dev'] = summarize(natural); outcomes.extend(natural)
        sources['natural_dev'] = {'prepared_manifest_sha256': provenance['manifest_sha256'],
                                  'baseline': 'fresh actual deployed hybrid inference'}
        print(json.dumps({'suite': 'natural_dev', **results['natural_dev']}), flush=True)
    artifact_root = ROOT / '.omx/artifacts'
    suites = ([('final_v22', 'data/challenge/v2.2/test.jsonl', 'challenge-v22-test-natural-v3-deployed.json')]
              if final else [('original_dev', 'data/challenge/dev.jsonl', 'challenge-dev-natural-v3-deployed.json'),
              ('selection_dev', 'data/challenge/selection-dev-v1.1/cases.jsonl', 'selection-dev-natural-v3-deployed.json'),
              ('reasoning_dev', 'data/challenge/reasoning-dev-v1.1/cases.jsonl', 'reasoning-dev-natural-v3-deployed.json')])
    for name, relative, baseline_name in suites:
        path = ROOT / relative; baseline_path = artifact_root / baseline_name
        baseline = json.loads(baseline_path.read_text())
        source_hash = baseline.get('sha256') or baseline['corpus_identity']['sha256']
        if file_sha256(path) != source_hash: raise ValueError('suite differs from pinned baseline')
        cases = [json.loads(line) for line in path.read_text().splitlines() if line]
        old = {row['id']:row for row in baseline['outcomes']}
        if set(old) != {case['id'] for case in cases}: raise ValueError('baseline IDs differ')
        current = []
        for index, case in enumerate(cases):
            previous = old[case['id']]
            if previous['version'] != parent_version or any(previous[k] != case[k] for k in ('input', 'context', 'expected')):
                raise ValueError('baseline version or cases differ')
            current.append(run_case(case, previous, name))
            if (index+1) % 64 == 0:
                print(json.dumps({'suite': name, 'cases': index+1, 'elapsed_seconds': time.monotonic()-started}), flush=True)
        if file_sha256(path) != source_hash: raise ValueError('evaluation source changed')
        sources[name] = {'sha256': source_hash, 'baseline_report_sha256': file_sha256(baseline_path)}
        results[name] = summarize(current); outcomes.extend(current)
        print(json.dumps({'suite': name, **results[name]}), flush=True)
    validate_word_data(Path(provenance['path']), provenance['manifest_sha256'])
    if before != source_hashes() or file_sha256(checkpoint / 'model.safetensors') != checkpoint_hash:
        raise ValueError('runtime artifacts changed during evaluation')
    passed = all(not result['regressions'] for result in results.values())
    if not final:
        passed &= results['natural_dev']['top1'] > results['natural_dev']['before_top1']
    report = {'passed': passed, 'checkpoint_sha256': checkpoint_hash, 'checkpoint': str(checkpoint),
              'parent_version': parent_version, 'experimental_version': experimental_version,
              'model_artifacts': artifacts, 'sources': sources, 'suites': results, 'outcomes': outcomes,
              'training_report_sha256': file_sha256(training_path), 'development_gate_sha256': gate_hash,
              'scope': 'exposed final regression' if final else 'actual hybrid development',
              'elapsed_seconds': time.monotonic()-started, 'deployed': False}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'passed': passed, 'suites': results}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    evaluate(args.checkpoint.resolve(), args.output, args.final)
