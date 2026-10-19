"""Prepare a lossless source-only FIT projection for span-aligned learning."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import resource
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.source_correspondence import align, compact_paths, contrast, expand_path


DATA = ROOT / '.omx/artifacts/recombination-fresh-runtime-v1'
SOURCES = ROOT / '.omx/artifacts/recombination-fresh-train-sources-v1'
PLAN = ROOT / '.omx/plans/source-correspondence-fit-audit-v1.md'
SCHEMA = 'source-correspondence-fit-audit-v1'
HASHES = {
    DATA / 'report.json': 'd890bad55cbd52f3168e8a8fa7f9caefc88be448dc35af011e4e386e7871212d',
    DATA / 'independent-verification.json': 'd01213d92b20523eb29646cbb4b460342e91bd7c10d782b54ce03ff1ef054ca6',
    DATA / 'runtime-capture.jsonl': '92f7ca44ae9b1302ffe87fec30fda3ae6849284c560530d3c25de21fd66e5a69',
    DATA / 'source-capture.jsonl': 'df9250b8f422f7aecdf255127e677bb30471957a888c74b6e713aab92a7d07ad',
    SOURCES / 'manifest.json': '5680aa160f99d9656b02e52e06d4218f0ca386848d7b733fdf03d2570acfa696',
    SOURCES / 'selection.jsonl': '15a0617e95d5c2fb0ec7ce895230794a9ac9d27f185b288eb3d138e0f91968d2',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def project_fit(selected, assignment, runtime, generated):
    # Gate before touching candidate fields: HOLD representations are not produced.
    require(assignment['split'] == 'fit', 'FIT-only projection required')
    source = selected['input']
    require(set(selected) == {'id', 'input'}
            and selected['id'] == assignment['id'] == runtime['id'] == generated['id']
            and runtime['request'] == generated['request'] == {'input': source, 'context': ''},
            'source identity differs')
    response = runtime['proposed']
    require(response['input'] == source and response['context'] == '', 'runtime source differs')
    candidates = response['pipeline']['pool'] + generated['proposals'] + response['candidates']
    paths = [align(source, c['text'], c['segments']) for c in candidates]
    record = compact_paths(paths)
    reference = response['candidates'][0]
    reference_path = align(source, reference['text'], reference['segments'])
    require(reference['text'] == response['result'], 'reference response differs')
    record.update(id=selected['id'], component_id=assignment['component_id'],
                  reference_path_id=reference_path.path_id)
    # Every saved structural path must round-trip to an actually observed path.
    actual = {p.path_id: p for p in paths}
    restored = [expand_path(record, p) for p in record['paths']]
    require({p.path_id: p for p in restored} == actual, 'compact round-trip differs')
    stats = describe(record, restored)
    stats['counts']['input_path_occurrences'] = len(candidates)
    return record, stats


def describe(record, restored=None):
    paths = restored if restored is not None else [expand_path(record, p) for p in record['paths']]
    reference = next(p for p in paths if p.path_id == record['reference_path_id'])
    by_text = defaultdict(list)
    counts = Counter(rows=1, paths=len(paths), edges=len(record['edges']))
    maxima = {'spans_per_path': 0, 'source_characters': len(record['input']),
              'paths_per_output': 0, 'changed_blocks_per_contrast': 0}
    for path in paths:
        by_text[path.text].append(path)
        maxima['spans_per_path'] = max(maxima['spans_per_path'], len(path.spans))
        counts['span_occurrences'] += len(path.spans)
        copied = sum(s.action == 'copy' for s in path.spans)
        converted = len(path.spans) - copied
        opaque = sum(s.unresolved_ascii_output for s in path.spans)
        counts['copy_span_occurrences'] += copied
        counts['convert_span_occurrences'] += converted
        counts['unresolved_ascii_span_occurrences'] += opaque
        counts['paths_with_unresolved_ascii'] += bool(opaque)
        counts['paths_with_copy_and_convert'] += bool(copied and converted)
        if path.path_id == reference.path_id:
            continue
        blocks = contrast(reference, path)
        require(''.join(b['source'] for b in blocks) == path.source
                and ''.join(b['left']['text'] for b in blocks) == reference.text
                and ''.join(b['right']['text'] for b in blocks) == path.text,
                'contrast round-trip differs')
        changed = [b for b in blocks if b['relation'] == 'output_changed']
        counts['contrasts'] += 1
        counts['raw_contrasts'] += path.text == path.source
        counts['nonraw_contrasts'] += path.text != path.source
        counts['segmentation_only_blocks'] += sum(b['relation'] == 'segmentation_only' for b in blocks)
        counts['output_changed_blocks'] += len(changed)
        counts['unchanged_blocks'] += sum(b['relation'] == 'same' for b in blocks)
        counts['contrasts_with_output_changes'] += bool(changed)
        counts['contrasts_with_multiple_changed_blocks'] += len(changed) > 1
        counts['contrasts_with_localized_output_changes'] += bool(changed) and all(
            b['input_end'] - b['input_start'] < len(path.source) for b in changed)
        counts['changed_source_characters'] += sum(b['input_end'] - b['input_start'] for b in changed)
        counts['compared_source_characters'] += len(path.source)
        maxima['changed_blocks_per_contrast'] = max(maxima['changed_blocks_per_contrast'], len(changed))
    counts['unique_outputs'] = len(by_text)
    counts['outputs_with_multiple_paths'] = sum(len(v) > 1 for v in by_text.values())
    counts['rows_with_multiple_paths_per_output'] = bool(counts['outputs_with_multiple_paths'])
    counts['rows_with_unresolved_ascii'] = bool(counts['paths_with_unresolved_ascii'])
    counts['outputs_with_only_unresolved_ascii_paths'] = sum(
        all(any(s.unresolved_ascii_output for s in p.spans) for p in variants)
        for variants in by_text.values())
    maxima['paths_per_output'] = max(map(len, by_text.values()), default=0)
    return {'counts': dict(counts), 'maxima': maxima}


def run(output):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for path, digest in HASHES.items():
        require(sha(path) == digest, 'frozen source artifact changed: ' + str(path))
    proof = read(DATA / 'independent-verification.json')
    require(proof['verified'] is True and proof['partition_counts'] == {'fit': 1642, 'inner_hold': 406},
            'source proof differs')
    manifest = read(SOURCES / 'manifest.json')
    assignments = manifest['partition']
    require(Counter(x['split'] for x in assignments) == {'fit': 1642, 'inner_hold': 406},
            'global split differs')
    pins = {str(p): h for p, h in HASHES.items()}
    for p in (Path(__file__).resolve(), ROOT / 'hanglish/core/source_correspondence.py', PLAN):
        pins[str(p)] = sha(p)
    totals = Counter(); maxima = {}; seen = set(); skipped = 0
    capture = output / 'fit-paths.jsonl'
    journal = output / 'rows.jsonl'
    with capture.open('x', encoding='utf-8') as stream, journal.open('x', encoding='utf-8') as row_stream:
        for selected, assignment, runtime, generated in zip_longest(
                rows(SOURCES / 'selection.jsonl'), assignments,
                rows(DATA / 'runtime-capture.jsonl'), rows(DATA / 'source-capture.jsonl')):
            require(all(x is not None for x in (selected, assignment, runtime, generated)), 'row denominator differs')
            require(selected['id'] == assignment['id'] == runtime['id'] == generated['id']
                    and selected['id'] not in seen, 'source order or duplicate differs')
            seen.add(selected['id'])
            require(time.monotonic() - started <= 300, 'FIT projection time limit exceeded')
            if assignment['split'] != 'fit':
                skipped += 1
                continue
            record, stats = project_fit(selected, assignment, runtime, generated)
            stream.write(json.dumps(record, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
            row_stream.write(json.dumps(dict(id=selected['id'], **stats), sort_keys=True) + '\n')
            totals.update(stats['counts'])
            for key, value in stats['maxima'].items():
                maxima[key] = max(maxima.get(key, 0), value)
        for target in (stream, row_stream):
            target.flush()
            os.fsync(target.fileno())
    require(len(seen) == 2048 and skipped == 406 and totals['rows'] == 1642, 'final FIT count differs')
    require(all(sha(p) == digest for p, digest in pins.items()), 'inputs changed during projection')
    closure = {'schema': SCHEMA + '-closure', 'rows': 1642,
               'capture_sha256': sha(capture), 'source_only': True, 'targets_joined': False}
    write_json(output / 'capture.complete.json', closure)
    # macOS ru_maxrss is bytes; Linux reports KiB.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    report = {'schema': SCHEMA, 'completed': True, 'pins': pins,
              'counts': dict(totals), 'maxima': maxima, 'skipped_inner_hold_rows': skipped,
              'elapsed_s': time.monotonic() - started, 'peak_rss_bytes': peak,
              'outputs': {p.name: sha(p) for p in (capture, journal, output / 'capture.complete.json')},
              'contracts': {'model_loaded': False, 'tokenizer_loaded': False,
                            'labels_opened': False, 'inner_hold_projected': False,
                            'dev_opened': False, 'test_opened': False,
                            'production_changed': False, 'phonetic_correctness_claimed': False}}
    require(report['elapsed_s'] <= 300, 'FIT projection time limit exceeded')
    write_json(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({k: report[k] for k in ('counts', 'maxima', 'elapsed_s', 'peak_rss_bytes')}))
