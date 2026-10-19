"""Exposed final regression, opened only after the complete development gate."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts.evaluate_constrained_runtime import development_passed
from scripts.probe_constrained_runtime import (
    OUTPUT, VERSION, hashes, install, metrics, NeuralConverter, file_sha256,
    SentencePoolRecorder, rss_bytes,
)
from scripts.evaluate_roberta_context_ranker import FINAL_SUITE

FINAL_CORPUS_SHA256 = '3d2fe696cb62e3e895017f817e8f5ccca8a651f17a325725ef624b6d3335c720'


def main(args):
    destination = OUTPUT/'final.json'
    if destination.exists():
        raise FileExistsError(destination)
    gate_path = OUTPUT/'development.json'
    gate = json.loads(gate_path.read_text())
    before = hashes()
    if (file_sha256(gate_path) != args.development_sha256
            or gate.get('passed') is not True or gate.get('source_hashes') != before
            or gate.get('parent_version') != VERSION or gate.get('test_data_read') is not False
            or gate.get('scope') != 'paired development; no final test inference'
            or gate.get('deployed') is not False or not development_passed(gate['suites'])
            or file_sha256(ROOT/'scripts/evaluate_constrained_runtime.py') != gate['evaluator_sha256']
            or file_sha256(OUTPUT/'development-cases.jsonl') != gate['cases_sha256']
            or file_sha256(OUTPUT/'report.json') != gate['train_report_sha256']
            or any(file_sha256(ROOT/p) != digest for p, digest in gate['extra_source_hashes'].items())):
        raise ValueError('complete current source-pinned development gate required')
    # No final corpus is read until all development conditions above pass.
    name, relative, identity = FINAL_SUITE
    path, identity_path = ROOT/relative, ROOT/'.omx/artifacts'/identity
    if file_sha256(identity_path) != args.final_identity_sha256:
        raise ValueError('final identity differs from explicit pin')
    frozen = json.loads(identity_path.read_text())
    digest = frozen.get('sha256') or frozen['corpus_identity']['sha256']
    if digest != FINAL_CORPUS_SHA256 or file_sha256(path) != digest:
        raise ValueError('final corpus differs from frozen identity')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != 1000:
        raise ValueError('full final corpus required')
    identity_digest, evaluator_digest = file_sha256(identity_path), file_sha256(Path(__file__))
    torch.set_num_threads(2)
    converter = NeuralConverter(ranker_kind='hybrid')
    converter.load_models()
    assert converter.version == VERSION
    parent = converter.word_options
    install(converter)
    changed = converter.word_options
    recorder = SentencePoolRecorder(converter.sentence_ranker)
    converter.sentence_ranker = recorder
    cases, started = [], time.monotonic()
    with (OUTPUT/'final-cases.jsonl').open('x') as stream:
        for index, row in enumerate(rows):
            case = {k: row[k] for k in ('id', 'input', 'context', 'expected')}
            try:
                for side, options in (('before', parent), ('new', changed)):
                    converter.word_options = options
                    recorder.candidates = None
                    result = converter.convert(row['input'], row['context'])
                    choices = [c['text'] for c in result['candidates']]
                    pool = recorder.candidates
                    case.update({f'{side}_candidates': choices,
                                 f'{side}_top1': choices[0] in row['expected'],
                                 f'{side}_top5': any(c in row['expected'] for c in choices),
                                 f'{side}_covered': None if pool is None else any(c in row['expected'] for c in pool)})
            finally:
                converter.word_options = parent
            cases.append(case)
            stream.write(json.dumps(case, ensure_ascii=False)+'\n')
            stream.flush()
            if rss_bytes() > 3*1024**3 or time.monotonic()-started > 1800:
                raise RuntimeError('final evaluation resource bound exceeded')
            if (index+1) % 100 == 0:
                print(json.dumps({'rows': index+1}), flush=True)
    if (before != hashes() or file_sha256(gate_path) != args.development_sha256
            or file_sha256(path) != digest or file_sha256(identity_path) != identity_digest
            or file_sha256(Path(__file__)) != evaluator_digest
            or file_sha256(ROOT/'scripts/evaluate_constrained_runtime.py') != gate['evaluator_sha256']
            or file_sha256(OUTPUT/'development-cases.jsonl') != gate['cases_sha256']
            or file_sha256(OUTPUT/'report.json') != gate['train_report_sha256']):
        raise ValueError('final evaluation dependencies changed')
    summary = metrics(cases)
    passed = (not summary['regressions'] and summary['new_top5'] == 1000
              and all(not c['before_covered'] or c['new_covered'] for c in cases))
    report = {'scope': 'exposed final regression', 'passed': bool(passed), 'metrics': summary,
              'parent_version': VERSION, 'source_hashes': before,
              'development_sha256': args.development_sha256, 'corpus_sha256': digest,
              'final_identity_sha256': identity_digest,
              'evaluator_sha256': evaluator_digest, 'cases_sha256': file_sha256(OUTPUT/'final-cases.jsonl'),
              'seconds': time.monotonic()-started, 'peak_rss_bytes': rss_bytes(),
              'test_data_read': True, 'deployed': False}
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'passed': passed, 'metrics': summary}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--development-sha256', required=True)
    parser.add_argument('--final-identity-sha256', required=True)
    main(parser.parse_args())
