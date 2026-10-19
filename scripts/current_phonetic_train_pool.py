"""Prepare and capture the current phonetic union on the complete existing FIT split."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from scripts.verify_production_phonetic_choice_dev import need, sha, read, jsonl, render_public
from scripts.evaluate_phonetic_choice_dev import probe_groups
from scripts.evaluate_sentence_only import predict_sentence
from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.omx/artifacts/source-correspondence-fit-v2'
PREPARED = ROOT / '.omx/artifacts/current-phonetic-train-inputs-v1'
PROBE = ROOT / '.omx/artifacts/current-phonetic-train-probe-v1'
PLAN = ROOT / '.omx/plans/current-phonetic-train-pool-v1.md'
MODEL = ROOT / '.omx/artifacts/observed-phonetic-augmented-training-v1/phonetic_transducer.pt'
REFERENCE = ROOT / '.omx/artifacts/phonetic-choice-runtime-dev-v1'
ZERO_VERSION = 'experimental-phonetic-choice-1-2bbe1463904b'
LIMITS = dict(whole_seconds=1800., update_seconds=60., rss_bytes=3*1024**3, swap_growth_bytes=512*1024**2)
FROZEN = {
    SOURCE / 'fit-paths.jsonl': 'f1cd66da23cacc6025e8aa2203944f664f0d74d916a0bde135cac65ef56afd8c',
    SOURCE / 'report.json': '76672ccc2e56269a0b2c40701b8b10f9dc333c3d5ebeca274add7b305d3d65c0',
    SOURCE / 'independent-verification.json': '449a8f286a60cc4a7c542612054c4ca762378caedfb0cbc9ca350801030f2de2',
    MODEL: 'f06b0a7cd6c5ccbdfaa376cc29f59e90b3fcf0ae3a157fd2c4f7efaa73b4180d',
    ROOT / 'scripts/verify_production_phonetic_choice_dev.py': '366d463af2fefc24f7bbf69c6aa52f46e52467335021c442a14eb1a38328003c',
    REFERENCE / 'report.json': 'a0072c79116da9acdbff83b373dccbb8c3ef8c26ac5164f575490941808d4473',
    REFERENCE / 'independent-verification.json': 'f2e2eca118ef05c1397304d89ccea99e1d9ef28841fe109fcd04446d0928e6c4',
    REFERENCE / 'capture.jsonl': 'e0447efa305e1d4f61052866b1ee121a62b400f3adf163dfdeac1b0bb4734bf5',
}


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def append(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush(); os.fsync(stream.fileno())


def source_rows(records):
    result, seen = [], set()
    for row in records:
        source, identifier, component = row['input'], row['id'], row['component_id']
        need(isinstance(source, str) and source and not re.search('[가-힣ᄀ-ᇿ㄰-㆏]', source)
             and isinstance(identifier, str) and identifier.startswith('s3-train-')
             and identifier not in seen and isinstance(component, str) and component,
             'FIT source domain differs')
        seen.add(identifier)
        result.append(dict(id=identifier, input=source, component_id=component))
    return result


def partition(rows, count=6):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['component_id']].append(row)
    loads, assignment = [0]*count, {}
    ordered = sorted(grouped, key=lambda c: (-sum(len(r['input']) for r in grouped[c]), c))
    for component in ordered:
        index = min(range(count), key=lambda i: (loads[i], i))
        assignment[component] = index
        loads[index] += sum(len(r['input']) for r in grouped[component])
    return [[r['id'] for r in rows if assignment[r['component_id']] == i] for i in range(count)]


def prepare(output):
    need(all(sha(p) == value for p, value in FROZEN.items()), 'frozen source differs')
    proof = read(SOURCE / 'independent-verification.json')
    need(proof['verified'] is True and proof['counts']['rows'] == 1642
         and proof['report_sha256'] == sha(SOURCE / 'report.json'), 'FIT proof differs')
    rows = source_rows(jsonl(SOURCE / 'fit-paths.jsonl'))
    need(len(rows) == 1642, 'FIT row count differs')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    groups = probe_groups([(r['id'], r['input'], None) for r in rows])
    write(output / 'requests.json', dict(rows=rows, groups=groups, shards=partition(rows)))
    pins = {str(p): value for p, value in FROZEN.items()}
    for p in (Path(__file__), PLAN):
        pins[str(p)] = sha(p)
    write(output / 'manifest.json', dict(rows=len(rows), components=len({r['component_id'] for r in rows}),
        pins=pins, requests_sha256=sha(output / 'requests.json'), train_only=True,
        previous_eligible_rows=1483, restored_rows=159, labels_opened=False))
    return dict(prepared=len(rows), shards=list(map(len, partition(rows))))


def validate_engine_identity(engine, expected_artifacts):
    need(engine.phonetic_parent_version == 'experimental-ctc-union-1-488fc89732af'
         and engine.version == ZERO_VERSION and engine.model_artifacts == expected_artifacts,
         'frozen phonetic pipeline identity differs')


def load_engine(expected_artifacts):
    import torch
    from hanglish.core.phonetic_choice_converter import PhoneticChoiceConverter
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    engine = PhoneticChoiceConverter(phonetic_checkpoint=MODEL, calibration_weights=[0.]*4)
    engine.load_models(); engine.load_word_model(); engine.load_growth_model(); engine.load_phonetic_model()
    validate_engine_identity(engine, expected_artifacts)
    return engine


def check_capture(source, response, paths):
    choice = response['phonetic_choice']
    need(response['input'] == source and response['context'] == ''
         and response['external_context_used'] is False
         and choice['effective_weights'] == [0.]*4 and choice['calibration_source'] == 'override', 'source/zero contract differs')
    need(len(paths) == len(choice['rows']), 'feature coverage differs')
    if response['sentence_ranking_applied']:
        need(bool(paths) and choice['zero_weight_score_parity'] and choice['zero_weight_order_parity']
             and render_public(source, paths) == response['candidates'], 'zero public parity differs')
        need([p[0] for p in paths] == response['phonetic_word_union']['union'], 'union coverage differs')
    else:
        need(not paths and not choice['score_records'], 'unranked input was scored')
    for path, row in zip(paths, choice['rows']):
        need(row['text'] == path[0] and row['segments'] == path[2]
             and row['base_score'] == row['adjusted_score'] == path[1] and row['residual'] == 0,
             'zero scores differ')
        need(len(row['features']) == 4 and all(type(v) in (int, float) and math.isfinite(v)
             for v in row['features']), 'invalid phonetic features')


def estimates(groups, shards, timings, initialization):
    by_id = {identifier: timings[g['probe_id']] for g in groups for identifier in g['ids']}
    return [1.3*(initialization + sum(by_id[identifier] for identifier in shard))+60. for shard in shards]


def capture(output, mode, shard=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        with AtomicResourceWatchdog(started, system_swap_bytes(), output / 'resources.jsonl', limits=LIMITS, interval_s=1.) as guard:
            with guard.operation('authenticate', update=True):
                manifest, prepared = read(PREPARED / 'manifest.json'), read(PREPARED / 'requests.json')
                prepared_proof = read(PREPARED / 'independent-verification.json')
                need(prepared_proof['verified'] is True
                     and prepared_proof['manifest_sha256'] == sha(PREPARED / 'manifest.json')
                     and prepared_proof['requests_sha256'] == sha(PREPARED / 'requests.json'), 'prepared proof differs')
                pins = dict(manifest['pins'])
                need(manifest['requests_sha256'] == sha(PREPARED / 'requests.json'), 'prepared requests differ')
                need(all(sha(p) == h for p, h in pins.items()), 'source pins differ')
                reference_proof = read(REFERENCE / 'independent-verification.json')
                need(reference_proof['verified'] is True
                     and reference_proof['report_sha256'] == sha(REFERENCE / 'report.json')
                     and reference_proof['capture_sha256'] == sha(REFERENCE / 'capture.jsonl'), 'runtime reference differs')
                with (REFERENCE / 'capture.jsonl').open() as reference:
                    historical = json.loads(next(reference))['proposed']
                expected_artifacts = dict(historical['model_artifacts'])
                expected_artifacts.pop('.omx/artifacts/phonetic-choice-calibration-training-v1/weights.json')
                need(len(prepared['rows']) == 1642 and source_rows(prepared['rows']) == prepared['rows']
                     and partition(prepared['rows']) == prepared['shards'], 'prepared domain differs')
                ids = [g['probe_id'] for g in prepared['groups']] if mode == 'probe' else prepared['shards'][shard]
                if mode == 'capture':
                    qualification, proof = read(PROBE / 'report.json'), read(PROBE / 'independent-verification.json')
                    need(proof['verified'] is True and proof['report_sha256'] == sha(PROBE / 'report.json')
                         and qualification['qualified'] is True, 'probe did not qualify')
                    for path in (PROBE / 'report.json', PROBE / 'independent-verification.json'):
                        pins[str(path)] = sha(path)
                for p in (PREPARED / 'requests.json', PREPARED / 'manifest.json',
                          PREPARED / 'independent-verification.json', Path(__file__), PLAN,
                          ROOT / 'scripts/source_span_resources.py', ROOT / 'scripts/evaluate_sentence_only.py',
                          ROOT / 'scripts/preflight_pko_seq2seq.py',
                          ROOT / 'scripts/evaluate_phonetic_choice_dev.py'):
                    pins[str(p)] = sha(p)
            with guard.operation('load_models', update=True):
                engine = load_engine(expected_artifacts)
                for name, digest in engine.model_artifacts.items():
                    path = ROOT / name
                    need(sha(path) == digest, 'model artifact differs')
                    pins[str(path)] = digest
                write(output / 'pins.json', pins)
            initialization = time.monotonic()-started
            by_id = {r['id']: r for r in prepared['rows']}
            timings = {}
            with (output / 'capture.jsonl').open('x') as stream:
                for index, identifier in enumerate(ids):
                    row = by_id[identifier]; tick = time.monotonic()
                    with guard.operation('predict:' + identifier, update=True):
                        response = predict_sentence(engine, row['input'])
                        paths = engine.choice_base_paths
                        check_capture(row['input'], response, paths)
                        elapsed = time.monotonic()-tick
                        append(stream, dict(id=identifier, component_id=row['component_id'],
                            request=dict(input=row['input'], context=''), response=response, base_paths=paths,
                            elapsed_s=elapsed))
                    timings[identifier] = time.monotonic()-tick
                    if mode == 'probe' or (index+1) % 25 == 0:
                        print(json.dumps(dict(captured=index+1, total=len(ids), elapsed_s=time.monotonic()-started)), flush=True)
            with guard.operation('write_report', update=True):
                write(output / 'capture.complete.json', dict(mode=mode, shard=shard, rows=len(ids),
                    capture_sha256=sha(output / 'capture.jsonl'), labels_opened=False))
                need(all(sha(p) == h for p, h in pins.items()), 'final pins differ')
                projected = estimates(prepared['groups'], prepared['shards'], timings, initialization) if mode == 'probe' else None
                report = dict(completed=True, mode=mode, shard=shard, rows=len(ids), ids=ids, pins=pins,
                    capture_sha256=sha(output / 'capture.jsonl'),
                    closure_sha256=sha(output / 'capture.complete.json'), initialization_s=initialization,
                    timings=timings, projected_shard_seconds=projected,
                    qualified=all(t <= LIMITS['whole_seconds'] for t in projected) if projected else True,
                    version=engine.version, model_artifacts=engine.model_artifacts, limits=LIMITS,
                    train_only=True, labels_opened=False, production_changed=False, neural_training=False)
                write(output / 'report.json', report)
            guard.sample_and_append('after_report')
            with guard.operation('write_completion', update=True):
                write(output / 'completion.json', dict(completed=True, report_sha256=sha(output / 'report.json')))
            guard.sample_and_append('after_completion')
        return {k: report[k] for k in ('completed', 'rows', 'qualified', 'projected_shard_seconds')}
    except BaseException as error:
        write(output / 'failure.json', dict(completed=False, error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'probe', 'capture'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=range(6))
    args = parser.parse_args()
    need((args.mode == 'capture') == (args.shard is not None), 'shard is required only for capture')
    print(json.dumps(prepare(args.output) if args.mode == 'prepare' else capture(args.output, args.mode, args.shard)))
