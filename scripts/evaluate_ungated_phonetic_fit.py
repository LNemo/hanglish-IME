"""Measure generated-candidate coverage after source-only capture closure."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_fit_generation_gate import sha
from scripts.audit_saved_candidate_beam_union import reachable
from hanglish.core.source_correspondence import align

LABELS = ROOT / '.omx/artifacts/current-phonetic-train-features-v1/rows.jsonl'
LABEL_SHA = '4c13063656d9a108316b9ce1ceec394a61e2c45c2ba86e7a8c8361f7480d6cb4'


def graph_reachability(row, gold):
    """Gold-conditioned upper bounds, never an inference-time search result."""
    source = row['request']['input']

    def edges(candidates):
        values = set()
        for candidate in candidates:
            path = align(source, candidate['text'], candidate['segments'])
            values.update((span.input_start, span.input_end, span.text) for span in path.spans)
        return values

    parent = edges(row['parents'])
    expanded = parent | edges(row['proposals'])
    all_beams = set(expanded)
    for candidate in row['parents']:
        for segment in candidate['segments']:
            for replacement, _ in row['generated_by_source'].get(segment['source'], []):
                all_beams.add((segment['input_start'], segment['input_end'], replacement))

    def can_reach(values):
        graph = [{'start': start, 'end': end, 'text': text} for start, end, text in values]
        return any(reachable(target, graph, len(source)) for target in gold)

    return {'parent_edge_reachable': can_reach(parent),
            'expanded_edge_reachable': can_reach(expanded),
            'all_beams_edge_reachable': can_reach(all_beams)}


def compare(row, label):
    if row['id'] != label['id'] or row['request'] != {'input': label['input'], 'context': ''}:
        raise ValueError('FIT identity mismatch')
    gold = set(label['expected'])
    if not gold:
        raise ValueError('missing gold')
    original = {candidate['text'] for candidate in row['fallback_public']}
    proposed = {candidate['text'] for candidate in row['proposals']}
    supported = {candidate['text'] for candidate in row['proposals']
                 if candidate['sentence_support'] == 'supported'}
    return {'id': row['id'], 'original_available': bool(gold & original),
            'expanded_available': bool(gold & (original | proposed)),
            'new_gold': sorted(gold & (proposed - original)),
            'new_supported_gold': sorted(gold & (supported - original)),
            'new_unsupported_gold': sorted(gold & (proposed - supported - original)),
            'generated_candidates': len(proposed)}


def run(capture, output):
    if output.exists():
        raise FileExistsError(output)
    report_path = capture / 'report.json'
    proof_path = capture / 'verification.json'
    report = json.loads(report_path.read_text())
    proof = json.loads(proof_path.read_text())
    if not (report['completed'] and proof['verified'] and report['rows'] == 13
            and proof['report_sha256'] == sha(report_path)
            and proof['capture_sha256'] == report['capture_sha256'] == sha(capture / 'capture.jsonl')
            and report['labels_opened'] is False):
        raise ValueError('closed source-only capture and verified replay required')
    pins = {**report['pins'], str(report_path.resolve()): sha(report_path),
            str(proof_path.resolve()): sha(proof_path),
            str((capture / 'capture.jsonl').resolve()): sha(capture / 'capture.jsonl'),
            str((capture / 'resources.jsonl').resolve()): report['resources_sha256'],
            str(Path(__file__).resolve()): sha(__file__)}
    for relative in ('scripts/audit_saved_candidate_beam_union.py',
                     'hanglish/core/source_correspondence.py'):
        pins[str(ROOT / relative)] = sha(ROOT / relative)
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('capture closure changed')
    # The first label read occurs after capture and replay authentication.
    if sha(LABELS) != LABEL_SHA:
        raise ValueError('FIT labels changed')
    with LABELS.open() as stream:
        labels = {row['id']: row for row in map(json.loads, stream)}
    if len(labels) != 1642:
        raise ValueError('FIT label coverage differs')
    outcomes = []
    with (capture / 'capture.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            label = labels[row['id']]
            outcomes.append({**compare(row, label), **graph_reachability(row, label['expected'])})
    if len(outcomes) != 13 or len({row['id'] for row in outcomes}) != 13:
        raise ValueError('selected row coverage differs')
    totals = Counter(rows=len(outcomes))
    for row in outcomes:
        for key in ('original_available', 'expanded_available', 'generated_candidates',
                    'parent_edge_reachable', 'expanded_edge_reachable', 'all_beams_edge_reachable'):
            totals[key] += int(row[key])
        for key in ('new_gold', 'new_supported_gold', 'new_unsupported_gold'):
            totals[key + '_rows'] += bool(row[key])
        totals['coverage_gain'] += not row['original_available'] and row['expanded_available']
        totals['coverage_loss'] += row['original_available'] and not row['expanded_available']
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('input changed during evaluation')
    result = {'completed': True, 'scope': '13 source-selected length-gated FIT rows only',
              'counts': dict(totals), 'outcomes': outcomes, 'pins': pins,
              'labels_sha256': LABEL_SHA, 'labels_opened_after_verified_capture': True,
              'dev_opened': False, 'test_opened': False, 'production_changed': False,
              'limitations': ['Candidate coverage only; no sentence scores or first/top5 ranking.',
                              'Edge reachability uses the gold output and is only an upper bound, not a deployable decoder.',
                              'Shared-helper replay checks capture consistency; it does not rerun neural beams.',
                              'No unseen-data or whole-corpus improvement claim.']}
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(totals)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.capture, args.output)
