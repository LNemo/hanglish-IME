"""Fresh paired development and gated final regression for phonetic fallback preservation."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.neural_converter import NeuralConverter, file_sha256
from scripts.evaluate_natural_ranker import summarize
from scripts.evaluate_roberta_context_ranker import DEV_COUNTS, DEV_SUITES, FINAL_SUITE
from scripts.prepare_context_training import validate_context_data
from scripts.probe_phonetic_fallback_union import (
    OUTPUT, JOINT, JOINT_MANIFEST, PROFILE, RUNTIME_VERSION,
    SentencePoolRecorder, install_fallback_union, source_hashes, summarize as summarize_probe,
)
from scripts.train_natural_sentence_ranker import rss_bytes


def hashes():
    values = source_hashes()
    for name in ('scripts/evaluate_phonetic_fallback_union.py', 'scripts/evaluate_natural_ranker.py',
                 'scripts/evaluate_roberta_context_ranker.py', 'scripts/train_natural_sentence_ranker.py'):
        values[name] = file_sha256(ROOT / name)
    return values


def development_passed(suites):
    return (set(suites) == set(DEV_COUNTS)
            and all(suites[name]['rows'] == count and not suites[name]['regressions']
                    for name, count in DEV_COUNTS.items())
            and suites['natural_dev']['top1'] > suites['natural_dev']['before_top1'])


def require_final_gate(gate, before, probe_hash):
    if (gate.get('scope') != 'fresh paired development' or gate.get('passed') is not True
            or gate.get('test_data_read') is not False or gate.get('deployed') is not False
            or gate.get('source_hashes') != before or gate.get('probe_report_sha256') != probe_hash
            or gate.get('parent_version') != RUNTIME_VERSION
            or not development_passed(gate.get('suites', {}))):
        raise ValueError('current complete regression-free development gain required')


def validate_probe_cases(probe, cases, sources):
    audit = probe.get('word_option_audit', {})
    if (probe.get('development_inference_run') is not False or probe.get('deployed') is not False
            or audit.get('all_baseline_options_preserved') is not True
            or audit.get('all_additions_forward_valid') is not True
            or type(audit.get('max_additions')) is not int
            or not 0 <= audit['max_additions'] <= PROFILE['fallback_limit']
            or len(cases) != len(sources) or len(cases) != PROFILE['natural_train_rows']):
        raise ValueError('incomplete or invalid training probe policy')
    for case, source in zip(cases, sources):
        if any(case.get(key) != source[key] for key in ('id', 'input', 'context', 'expected')):
            raise ValueError('training probe source or order differs')
        for prefix in ('before', 'new'):
            choices = case[f'{prefix}_top5_candidates']
            if (not isinstance(choices, list) or not 1 <= len(choices) <= 5
                    or any(not isinstance(c, str) for c in choices)
                    or case[f'{prefix}_top1'] != (choices[0] in source['expected'])
                    or case[f'{prefix}_top5'] != any(c in source['expected'] for c in choices)
                    or case[f'{prefix}_pool_covered'] not in (None, False, True)):
                raise ValueError('training probe metrics disagree with recorded choices')
    if (probe['metrics'] != summarize_probe(cases)
            or probe['corrections'] != [c['id'] for c in cases if c['new_top1'] and not c['before_top1']]
            or probe['regressions'] != [c['id'] for c in cases if c['before_top1'] and not c['new_top1']]):
        raise ValueError('training probe aggregate metrics disagree with cases')


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    _, data = validate_context_data(JOINT, JOINT_MANIFEST)
    probe_path = OUTPUT / 'report.json'
    probe_hash = file_sha256(probe_path)
    if probe_hash != args.probe_report_sha256:
        raise ValueError('training probe differs from explicitly pinned report')
    probe = json.loads(probe_path.read_text())
    metrics = probe.get('metrics', {})
    if (probe.get('source_hashes') != source_hashes() or probe.get('profile') != PROFILE
            or probe.get('parent_version') != RUNTIME_VERSION or metrics.get('rows') != 512
            or probe.get('test_data_read') is not False
            or file_sha256(OUTPUT/'cases.jsonl') != probe.get('cases_sha256')
            or metrics.get('new_top1', 0) <= metrics.get('before_top1', 0)
            or metrics.get('new_pool_covered', 0) < metrics.get('before_pool_covered', 0)):
        raise ValueError('trusted complete train probe with actual gain required')
    probe_cases = [json.loads(line) for line in (OUTPUT/'cases.jsonl').read_text().splitlines() if line]
    validate_probe_cases(probe, probe_cases, data['natural_train'])
    before = hashes()
    gate_hash = None
    if args.final:
        gate_path = OUTPUT / 'development.json'
        gate_hash = file_sha256(gate_path)
        require_final_gate(json.loads(gate_path.read_text()), before, probe_hash)
    # The final corpus is not opened before the development gate above.
    suites = [] if args.final else [('natural_dev', data['natural_dev'], None)]
    source_files = {}
    for name, relative, identity_name in ([FINAL_SUITE] if args.final else DEV_SUITES):
        path = ROOT / relative
        identity_path = ROOT / '.omx/artifacts' / identity_name
        identity = json.loads(identity_path.read_text())
        expected_hash = identity.get('sha256') or identity['corpus_identity']['sha256']
        if file_sha256(path) != expected_hash:
            raise ValueError('evaluation corpus differs from frozen identity')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        suites.append((name, rows, path))
        source_files[str(path.relative_to(ROOT))] = expected_hash
        source_files[str(identity_path.relative_to(ROOT))] = file_sha256(identity_path)
    converter = NeuralConverter(ranker_kind='hybrid')
    converter.load_models()
    if converter.version != RUNTIME_VERSION:
        raise ValueError('runtime parent changed')
    parent_options = converter.word_options
    install_fallback_union(converter, PROFILE['fallback_limit'])
    new_options = converter.word_options
    recorder = SentencePoolRecorder(converter.sentence_ranker)
    converter.sentence_ranker = recorder
    results, outcomes = {}, []
    started = time.monotonic()
    for name, rows, _ in suites:
        current = []
        for index, row in enumerate(rows):
            try:
                converter.word_options = parent_options
                recorder.candidates = None
                previous = converter.convert(row['input'], row['context'])
                previous_pool = recorder.candidates
                converter.word_options = new_options
                recorder.candidates = None
                converted = converter.convert(row['input'], row['context'])
                pool = recorder.candidates
            finally:
                converter.word_options = parent_options
            actual = [c['text'] for c in converted['candidates']]
            expected = row['expected']
            current.append({'id':row['id'], 'suite':name, 'input':row['input'], 'context':row['context'],
                            'expected':expected, 'before_actual':previous['result'], 'actual':actual,
                            'before_top1':previous['result'] in expected, 'top1':actual[0] in expected,
                            'top5':any(c in expected for c in actual),
                            'before_pool_covered':None if previous_pool is None else any(c in expected for c in previous_pool),
                            'pool_covered':None if pool is None else any(c in expected for c in pool),
                            'before_engine_ms':previous['engine_ms'], 'engine_ms':converted['engine_ms']})
            if rss_bytes() > PROFILE['rss_limit_bytes'] or time.monotonic()-started > PROFILE['time_limit_seconds']:
                raise RuntimeError('evaluation resource bound exceeded')
            if (index+1)%32 == 0:
                print(json.dumps({'suite':name, 'cases':index+1,
                                  'seconds':round(time.monotonic()-started,2)}),flush=True)
        results[name] = summarize(current)
        results[name].update(before_pool_covered=sum(r['before_pool_covered'] is True for r in current),
                             pool_covered=sum(r['pool_covered'] is True for r in current))
        outcomes.extend(current)
        print(json.dumps({'suite':name, **results[name]},ensure_ascii=False),flush=True)
    validate_context_data(JOINT, JOINT_MANIFEST)
    if before != hashes() or probe_hash != file_sha256(probe_path):
        raise ValueError('evaluation sources changed')
    if file_sha256(OUTPUT/'cases.jsonl') != probe['cases_sha256']:
        raise ValueError('training probe cases changed during evaluation')
    if args.final and gate_hash != file_sha256(gate_path):
        raise ValueError('required development gate changed during final evaluation')
    if any(file_sha256(ROOT / name) != digest for name, digest in source_files.items()):
        raise ValueError('evaluation corpus or identity changed')
    passed = (results['final_v22']['rows'] == 1000 and not results['final_v22']['regressions']
              if args.final else development_passed(results))
    report = {'passed':bool(passed), 'scope':'exposed final regression' if args.final else 'fresh paired development',
              'parent_version':RUNTIME_VERSION, 'source_hashes':before, 'probe_report_sha256':probe_hash,
              'development_gate_sha256':gate_hash, 'source_files':source_files,
              'suites':results, 'outcomes':outcomes, 'seconds':time.monotonic()-started,
              'peak_rss_bytes':rss_bytes(), 'test_data_read':bool(args.final), 'deployed':False,
              'timing_policy':'Parent first, shared generator caches; descriptive only, not a controlled speed comparison.'}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':bool(passed),'suites':results}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe-report-sha256', required=True)
    parser.add_argument('--final', action='store_true')
    main(parser.parse_args())
