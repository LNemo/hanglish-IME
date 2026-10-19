"""Audit source spans outside whole-word phonetic input, without labels/models."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.neural_converter import PROTECTED
from hanglish.core.ungated_phonetic_proposals import _validate_parent
from scripts.audit_fit_generation_gate import sha

AUDIT = ROOT / '.omx/artifacts/fit-generation-gate-audit-v1.json'
PLAN = ROOT / '.omx/plans/ungated-boundary-fit-audit-v1.md'
ELIGIBLE = re.compile(r'[a-z]{2,40}\Z')
LATIN = re.compile(r'[A-Za-z]')


def classify_span(source):
    """Flags overlap and describe syntax; none authorizes a conversion."""
    if not LATIN.search(source) or ELIGIBLE.fullmatch(source):
        return []
    flags = []
    if any(c.isdigit() for c in source): flags.append('digits')
    if re.search(r'[A-Z]', source): flags.append('uppercase')
    if any(not c.isalnum() for c in source): flags.append('punctuation')
    if len(source) > 40: flags.append('over_40_characters')
    if len(source) < 2: flags.append('under_2_characters')
    if re.search(r'[^\x00-\x7f]', source): flags.append('non_ascii')
    if PROTECTED.fullmatch(source): flags.append('parent_protected_pattern')
    if re.match(r'(?:https?://|www\.)', source, re.I): flags.append('url_shape')
    if re.fullmatch(r'\S+@\S+', source): flags.append('email_shape')
    if '`' in source: flags.append('backtick_shape')
    if '_' in source or '/' in source or '\\' in source: flags.append('path_or_identifier_shape')
    return flags


def audit_row(row):
    source = row['request']['input']
    if row['request'].get('context') != '' or not row['id'].startswith('s3-train-'):
        raise ValueError('FIT source contract differs')
    response = row['response']
    if response['input'] != source or type(response['sentence_ranking_applied']) is not bool:
        raise ValueError('parent input/ranking flag differs')
    spans = {}
    parent_count = 0
    for index, candidate in enumerate(response['candidates']):
        segments = candidate['segments']
        if not segments or all(s['kind'] == 'raw' for s in segments):
            continue
        _validate_parent(source, candidate, index)
        parent_count += 1
        for segment in segments:
            flags = classify_span(segment['source'])
            if not flags:
                continue
            key = (segment['input_start'], segment['input_end'])
            item = spans.setdefault(key, {
                'input_start': key[0], 'input_end': key[1], 'source': segment['source'],
                'flags': flags, 'parents': [],
            })
            item['parents'].append({'candidate_index': index, 'text': segment['text'],
                                    'kind': segment['kind'],
                                    'source_copied_exactly': segment['text'] == segment['source']})
    ranked = response['sentence_ranking_applied']
    return {'id': row['id'], 'request': row['request'], 'ranked': ranked,
            'parent_count': parent_count, 'opaque_spans': sorted(spans.values(),
                key=lambda value: (value['input_start'], value['input_end'])),
            'generation_requests': response['phonetic_word_union']['generation_requests']}


def verify_pins(pins):
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('source pin changed: ' + path)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    gate = json.loads(AUDIT.read_text())
    paths = [ROOT / f'.omx/artifacts/current-phonetic-train-shard-{s}-v1/capture.jsonl'
             for s in range(6)]
    pins = {str(path): gate['pins'][str(path)] for path in paths}
    for path in [AUDIT, PLAN, Path(__file__).resolve(),
                 ROOT / 'hanglish/core/ungated_phonetic_proposals.py',
                 ROOT / 'hanglish/core/neural_converter.py']:
        pins[str(path)] = sha(path)
    verify_pins(pins)
    counts, flag_rows, flag_spans = Counter(), Counter(), Counter()
    grouped = {'ranked': Counter(), 'unranked': Counter()}
    identifiers = set()
    with (output / 'capture.jsonl').open('x') as target:
        for shard, path in enumerate(paths):
            with path.open() as stream:
                for index, line in enumerate(stream):
                    row = audit_row(json.loads(line))
                    if row['id'] in identifiers: raise ValueError('duplicate source ID')
                    identifiers.add(row['id'])
                    row.update(source_shard=shard, source_row_index=index)
                    target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    group = 'ranked' if row['ranked'] else 'unranked'
                    summary = {'rows': 1, 'public_nonraw_parents': row['parent_count'],
                               'opaque_spans': len(row['opaque_spans']),
                               'rows_with_opaque_spans': int(bool(row['opaque_spans']))}
                    counts.update(summary); grouped[group].update(summary)
                    observed = set()
                    for span in row['opaque_spans']:
                        flag_spans.update(span['flags']); observed.update(span['flags'])
                    flag_rows.update(observed)
    if len(identifiers) != 1642: raise ValueError('FIT denominator differs')
    verify_pins(pins)
    report = {'completed': True, 'schema': 'opaque-fit-span-audit-v1',
              'counts': dict(counts), 'groups': {k: dict(v) for k,v in grouped.items()},
              'flag_rows': dict(flag_rows), 'flag_spans': dict(flag_spans), 'pins': pins,
              'capture_sha256': sha(output / 'capture.jsonl'),
              'seconds': time.monotonic() - started, 'labels_opened': False,
              'neural_inference': False, 'production_changed': False,
              'limitations': ['Public nonraw parent paths only; reserved raw output excluded.',
                              'Flags overlap; totals are not additive.',
                              'Opaque does not mean wrong, English, or safe to split.',
                              'No gold reachability or accuracy claim.']}
    with (output / 'report.json').open('x') as target:
        json.dump(report, target, ensure_ascii=False, indent=2); target.write('\n')
    print(json.dumps({k: report[k] for k in ['counts','groups','flag_rows','flag_spans','seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
