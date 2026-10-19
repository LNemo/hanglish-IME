"""Capture, replay, and evaluate fixed joint-search arms in separate stages."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_fit_generation_gate import sha

PARENT = ROOT / '.omx/artifacts/ungated-phonetic-fit-v1'
CHECKPOINT = ROOT / 'checkpoints/sentence_ranker'
ARMS = ('original', 'roman_repaired')
LIMITS = {'whole_seconds': 600., 'row_seconds': 30.,
          'rss_bytes': 2 * 1024**3, 'output_bytes': 128 * 1024**2}


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def write(path, value):
    with path.open('x') as stream:
        stream.write(canonical(value) + '\n')


def verify_pins(pins):
    for path, digest in pins.items():
        need(sha(path) == digest, 'input changed: ' + path)


def source_rows(*, repair_mixed=False, prioritize_beams=False):
    report, proof = read(PARENT / 'report.json'), read(PARENT / 'verification.json')
    need(report['completed'] and proof['verified'] and report['rows'] == 13
         and report['labels_opened'] is False
         and proof['report_sha256'] == sha(PARENT / 'report.json')
         and proof['capture_sha256'] == report['capture_sha256'] == sha(PARENT / 'capture.jsonl'),
         'closed parent capture required')
    pins = dict(report['pins'])
    for path in (PARENT / 'report.json', PARENT / 'verification.json', PARENT / 'capture.jsonl',
                 Path(__file__), ROOT / '.omx/plans/ungated-joint-fit-v1.md',
                 ROOT / 'hanglish/core/ungated_joint_proposals.py',
                 ROOT / 'hanglish/core/source_edge_recombination.py',
                 ROOT / 'hanglish/core/source_correspondence.py'):
        pins[str(path.resolve())] = sha(path)
    if repair_mixed:
        plan = ROOT / '.omx/plans/ungated-joint-mixed-fit-v2.md'
        pins[str(plan)] = sha(plan)
    if prioritize_beams:
        plan = ROOT / '.omx/plans/ungated-joint-ranked-fit-v3.md'
        pins[str(plan)] = sha(plan)
    verify_pins(pins)
    with (PARENT / 'capture.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    need(len(rows) == len({row['id'] for row in rows}) == 13, 'source row coverage differs')
    return rows, pins


def generate_row(row, tokenizer, *, repair_mixed=False, prioritize_beams=False):
    from hanglish.core.ungated_joint_proposals import propose_joint
    arms = {}
    for arm in ARMS:
        result = propose_joint(row['request']['input'], row['parents'], row['proposals'],
                               row['generated_by_source'], seed_mode=arm, repair_mixed=repair_mixed,
                               prioritize_beams=prioritize_beams)
        for candidate in result['proposals']:
            text = ' '.join(candidate['text'].strip().split())
            tokens = len(tokenizer(text, add_special_tokens=False, truncation=False)['input_ids'])
            tokens += tokenizer.num_special_tokens_to_add(pair=True)
            candidate['sentence_tokens'] = tokens
            candidate['sentence_support'] = 'supported' if tokens <= 96 else 'unsupported'
        arms[arm] = result
    inventories = [arms[arm]['diagnostics']['edge_inventory'] for arm in ARMS]
    need(inventories[0] == inventories[1], 'search arms must use the same edge inventory')
    digest = hashlib.sha256(canonical(inventories[0]).encode()).hexdigest()
    for arm in ARMS:
        del arms[arm]['diagnostics']['edge_inventory']
        arms[arm]['diagnostics']['edge_inventory_sha256'] = digest
    return {'id': row['id'], 'request': row['request'], 'arms': arms}


def counts(records):
    result = {}
    for arm in ARMS:
        counter = Counter(rows=0, proposals=0, supported=0, unsupported=0)
        for record in records:
            proposed = record['arms'][arm]['proposals']
            need(len(proposed) <= 96 and len({c['text'] for c in proposed}) == len(proposed),
                 'proposal budget or uniqueness differs')
            counter['rows'] += 1
            counter['proposals'] += len(proposed)
            for candidate in proposed:
                need(candidate['sentence_support'] in ('supported', 'unsupported'),
                     'unknown sentence support status')
                counter[candidate['sentence_support']] += 1
        result[arm] = dict(counter)
    return result


def capture(directory, *, repair_mixed=False, prioritize_beams=False):
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        rows, pins = source_rows(repair_mixed=repair_mixed, prioritize_beams=prioritize_beams)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
        need(tokenizer.num_special_tokens_to_add(pair=True) == 3, 'tokenizer contract differs')
        records, observations = [], []
        with (directory / 'capture.jsonl').open('x') as stream:
            for row in rows:
                tick = time.monotonic()
                record = generate_row(row, tokenizer, repair_mixed=repair_mixed,
                                      prioritize_beams=prioritize_beams)
                elapsed = time.monotonic() - started
                row_seconds = time.monotonic() - tick
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss *= 1 if sys.platform == 'darwin' else 1024
                need(elapsed <= LIMITS['whole_seconds'] and row_seconds <= LIMITS['row_seconds']
                     and rss <= LIMITS['rss_bytes'], 'joint search resource limit exceeded')
                payload = canonical(record) + '\n'
                need(stream.tell() + len(payload.encode()) <= LIMITS['output_bytes'], 'output limit exceeded')
                stream.write(payload)
                stream.flush()
                records.append(record)
                observations.append({'id': row['id'], 'seconds': elapsed,
                                     'row_seconds': row_seconds, 'rss_bytes': rss})
                print(json.dumps({'rows': len(records), 'seconds': elapsed}), flush=True)
        verify_pins(pins)
        need(time.monotonic() - started <= LIMITS['whole_seconds'], 'whole capture time exceeded')
        write(directory / 'report.json', {
            'completed': True, 'schema': 'ungated-joint-fit-v1', 'rows': 13,
            'counts': counts(records), 'pins': pins, 'limits': LIMITS,
            'resources': observations, 'elapsed_seconds': time.monotonic() - started,
            'capture_sha256': sha(directory / 'capture.jsonl'),
            'labels_opened': False, 'model_inference': False, 'production_changed': False,
            'repair_mixed': repair_mixed,
            'prioritize_beams': prioritize_beams,
        })
    except Exception as error:
        write(directory / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


def closed(directory):
    report = read(directory / 'report.json')
    need(report['completed'] and report['rows'] == 13 and not report['labels_opened']
         and report['schema'] == 'ungated-joint-fit-v1' and report['limits'] == LIMITS
         and report['model_inference'] is False and report['production_changed'] is False
         and report['capture_sha256'] == sha(directory / 'capture.jsonl'), 'capture closure differs')
    need(type(report.get('repair_mixed', False)) is bool, 'seed recipe must be boolean')
    need(type(report.get('prioritize_beams', False)) is bool, 'beam priority recipe must be boolean')
    verify_pins(report['pins'])
    return report


def verify(directory):
    report = closed(directory)
    rows, pins = source_rows(repair_mixed=report.get('repair_mixed', False),
                             prioritize_beams=report.get('prioritize_beams', False))
    need(pins == report['pins'], 'replay sources differ')
    report_sha = sha(directory / 'report.json')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    with (directory / 'capture.jsonl').open() as stream:
        captured = [json.loads(line) for line in stream]
    need(len(captured) == 13, 'capture row count differs')
    for row, record in zip(rows, captured):
        need(canonical(generate_row(row, tokenizer, repair_mixed=report.get('repair_mixed', False),
                                    prioritize_beams=report.get('prioritize_beams', False)))
             == canonical(record), 'joint search replay differs')
    need(counts(captured) == report['counts'], 'capture count differs')
    need(closed(directory) == report and sha(directory / 'report.json') == report_sha,
         'capture changed during replay')
    write(directory / 'verification.json', {
        'verified': True, 'rows': 13, 'report_sha256': report_sha,
        'capture_sha256': report['capture_sha256'], 'counts': report['counts'],
        'labels_opened': False, 'model_inference': False,
        'scope': 'Deterministic same-helper search replay, not independent search implementation.',
    })
    print(json.dumps({'verified': True, 'counts': report['counts']}), flush=True)


def evaluate(directory):
    report = closed(directory)
    proof = read(directory / 'verification.json')
    need(proof['verified'] and proof['report_sha256'] == sha(directory / 'report.json')
         and proof['capture_sha256'] == report['capture_sha256'], 'verified capture required')
    from scripts.evaluate_ungated_phonetic_fit import LABELS, LABEL_SHA
    need(sha(LABELS) == LABEL_SHA, 'FIT labels changed')
    with LABELS.open() as stream:
        labels = {r['id']: {'input': r['input'], 'expected': r['expected']}
                  for r in map(json.loads, stream)}
    need(len(labels) == 1642, 'FIT label denominator differs')
    rows, _ = source_rows(repair_mixed=report.get('repair_mixed', False),
                          prioritize_beams=report.get('prioritize_beams', False))
    with (directory / 'capture.jsonl').open() as stream:
        records = [json.loads(line) for line in stream]
    outcomes, totals = [], {arm: Counter(rows=0, old_available=0, expanded_available=0,
                                       new_gold_rows=0, new_supported_gold_rows=0) for arm in ARMS}
    for row, record in zip(rows, records):
        need(row['id'] == record['id'] and row['request'] == record['request']
             == {'input': labels[row['id']]['input'], 'context': ''}, 'FIT identity differs')
        gold = set(labels[row['id']]['expected'])
        old = {c['text'] for c in row['fallback_public'] + row['proposals']}
        arm_results = {}
        for arm in ARMS:
            candidates = record['arms'][arm]['proposals']
            new = {c['text'] for c in candidates}
            need(not new & old, 'joint proposal duplicates prior pool')
            supported = {c['text'] for c in candidates if c['sentence_support'] == 'supported'}
            values = {'old_available': bool(gold & old), 'expanded_available': bool(gold & (old | new)),
                      'new_gold_rows': bool(gold & new), 'new_supported_gold_rows': bool(gold & supported)}
            totals[arm]['rows'] += 1
            totals[arm].update(values)
            arm_results[arm] = values
        outcomes.append({'id': row['id'], 'arms': arm_results})
    need(len(outcomes) == 13, 'evaluation row count differs')
    need(closed(directory) == report, 'inputs changed during evaluation')
    write(directory / 'evaluation.json', {
        'completed': True, 'rows': 13, 'counts': {k: dict(v) for k, v in totals.items()},
        'outcomes': outcomes, 'labels_sha256': LABEL_SHA,
        'report_sha256': sha(directory / 'report.json'), 'proof_sha256': sha(directory / 'verification.json'),
        'scope': 'Selected FIT exact candidate coverage, not first/top5 or held-out accuracy.',
        'production_changed': False, 'dev_opened': False, 'test_opened': False,
    })
    print(json.dumps({k: dict(v) for k, v in totals.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('capture', 'verify', 'evaluate'))
    parser.add_argument('directory', type=Path)
    parser.add_argument('--repair-mixed', action='store_true')
    parser.add_argument('--prioritize-beams', action='store_true')
    args = parser.parse_args()
    if args.stage == 'capture':
        capture(args.directory, repair_mixed=args.repair_mixed, prioritize_beams=args.prioritize_beams)
    else:
        need(not args.repair_mixed and not args.prioritize_beams,
             'replay/evaluation read the captured recipe')
        {'verify': verify, 'evaluate': evaluate}[args.stage](args.directory)
