"""Development-only paired evaluation gated on the fixed training probe."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from hanglish.core.neural_converter import NeuralConverter, file_sha256
from scripts.probe_constrained_runtime import (
    OUTPUT, VERSION, POLICY, hashes, install, metrics,
    load_sources, source_fields, SentencePoolRecorder, rss_bytes,
)
from scripts.evaluate_roberta_context_ranker import DEV_SUITES, DEV_COUNTS


def verify_probe(report, cases, rows, current_hashes):
    if (report.get('source_hashes') != current_hashes
            or report.get('parent_version') != VERSION or report.get('policy') != POLICY
            or report.get('test_data_read') is not False
            or report.get('development_inference_run') is not False
            or report.get('deployed') is not False
            or len(cases) != 512 or len(rows) != 512):
        raise ValueError('complete current source-only training probe required')
    for case, row in zip(cases, rows):
        if any(case.get(k) != row[k] for k in ('id', 'input', 'context', 'expected')):
            raise ValueError('training source alignment differs')
        for side in ('before', 'new'):
            choices = case[f'{side}_candidates']
            pool = case[f'{side}_pool']
            if (not choices or len(choices) > 5
                    or case[f'{side}_top1'] != (choices[0] in row['expected'])
                    or case[f'{side}_top5'] != any(c in row['expected'] for c in choices)
                    or case[f'{side}_covered'] != (None if pool is None else any(c in row['expected'] for c in pool))):
                raise ValueError('training outcomes disagree with scores')
    summary = metrics(cases)
    if (report.get('metrics') != summary or summary['new_top1'] <= summary['before_top1']
            or summary['new_covered'] < summary['before_covered']):
        raise ValueError('actual training first-choice and coverage gain required')


def development_passed(suites):
    return (set(suites) == set(DEV_COUNTS)
            and all(suites[k]['rows'] == count and not suites[k]['regressions']
                    and not suites[k]['top5_losses'] and not suites[k]['coverage_losses']
                    and suites[k]['new_top5'] >= suites[k]['before_top5']
                    and suites[k]['new_covered'] >= suites[k]['before_covered']
                    for k, count in DEV_COUNTS.items())
            and suites['natural_dev']['new_top1'] > suites['natural_dev']['before_top1'])


def main(args):
    destination = OUTPUT/'development.json'
    if destination.exists():
        raise FileExistsError(destination)
    report_path = OUTPUT/'report.json'
    if file_sha256(report_path) != args.train_report_sha256:
        raise ValueError('training report differs from explicit pin')
    report = json.loads(report_path.read_text())
    if file_sha256(OUTPUT/'cases.jsonl') != report['cases_sha256']:
        raise ValueError('training outcomes changed')
    if file_sha256(OUTPUT/'selection.json') != report['selection_sha256']:
        raise ValueError('training selection changed')
    raw, _ = load_sources()
    train = [source_fields(row, 'natural_train') for row in raw['natural_train']]
    cases = [json.loads(line) for line in (OUTPUT/'cases.jsonl').read_text().splitlines()]
    before = hashes()
    verify_probe(report, cases, train, before)
    evaluator_sha = file_sha256(Path(__file__))
    suites = [('natural_dev', [source_fields(row, 'natural_dev') for row in raw['natural_dev']])]
    extra_sources = {}
    for name, relative, identity in DEV_SUITES:
        path = ROOT/relative
        identity_path = ROOT/'.omx/artifacts'/identity
        frozen = json.loads(identity_path.read_text())
        digest = frozen.get('sha256') or frozen['corpus_identity']['sha256']
        if file_sha256(path) != digest:
            raise ValueError('development corpus differs from pinned identity')
        extra_sources[relative] = digest
        extra_sources[str(identity_path.relative_to(ROOT))] = file_sha256(identity_path)
        suites.append((name, [json.loads(line) for line in path.read_text().splitlines() if line]))
    torch.set_num_threads(2)
    converter = NeuralConverter(ranker_kind='hybrid')
    converter.load_models()
    assert converter.version == VERSION
    parent = converter.word_options
    install(converter)
    changed = converter.word_options
    recorder = SentencePoolRecorder(converter.sentence_ranker)
    converter.sentence_ranker = recorder
    results, outcomes, started = {}, [], time.monotonic()
    with (OUTPUT/'development-cases.jsonl').open('x') as stream:
        for name, rows in suites:
            current = []
            for index, row in enumerate(rows):
                case = {k: row[k] for k in ('id', 'input', 'context', 'expected')}
                case['suite'] = name
                try:
                    for side, options in (('before', parent), ('new', changed)):
                        converter.word_options = options
                        recorder.candidates = None
                        result = converter.convert(row['input'], row['context'])
                        candidates = [c['text'] for c in result['candidates']]
                        pool = recorder.candidates
                        case.update({f'{side}_candidates': candidates, f'{side}_pool': pool,
                                     f'{side}_top1': candidates[0] in row['expected'],
                                     f'{side}_top5': any(c in row['expected'] for c in candidates),
                                     f'{side}_covered': None if pool is None else any(c in row['expected'] for c in pool),
                                     f'{side}_engine_ms': result['engine_ms']})
                finally:
                    converter.word_options = parent
                current.append(case)
                stream.write(json.dumps(case, ensure_ascii=False)+'\n')
                stream.flush()
                if rss_bytes() > 3*1024**3 or time.monotonic()-started > 1800:
                    raise RuntimeError('development resource bound exceeded')
                if (index+1) % 32 == 0:
                    print(json.dumps({'suite': name, 'rows': index+1}), flush=True)
            results[name] = metrics(current)
            results[name]['top5_losses'] = [c['id'] for c in current
                                            if c['before_top5'] and not c['new_top5']]
            results[name]['coverage_losses'] = [c['id'] for c in current
                                                if c['before_covered'] is True
                                                and c['new_covered'] is not True]
            outcomes.extend(current)
            print(json.dumps({'suite': name, **results[name]}), flush=True)
    if (before != hashes() or evaluator_sha != file_sha256(Path(__file__))
            or file_sha256(report_path) != args.train_report_sha256
            or file_sha256(OUTPUT/'cases.jsonl') != report['cases_sha256']
            or file_sha256(OUTPUT/'selection.json') != report['selection_sha256']
            or any(file_sha256(ROOT/path) != digest for path, digest in extra_sources.items())):
        raise ValueError('development dependencies changed')
    evidence = {'scope': 'paired development; no final test inference',
                'passed': development_passed(results), 'suites': results,
                'parent_version': VERSION, 'source_hashes': before,
                'evaluator_sha256': evaluator_sha, 'extra_source_hashes': extra_sources,
                'train_report_sha256': args.train_report_sha256,
                'cases_sha256': file_sha256(OUTPUT/'development-cases.jsonl'),
                'seconds': time.monotonic()-started, 'peak_rss_bytes': rss_bytes(),
                'test_data_read': False, 'deployed': False}
    destination.write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'passed': evidence['passed'], 'suites': results}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-report-sha256', required=True)
    main(parser.parse_args())
