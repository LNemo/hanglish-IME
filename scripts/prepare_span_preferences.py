"""Prepare FIT-only conditional local labels; no model or evaluation labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.models.span_supervision import SCHEMA, digest, extract_preferences
from scripts.train_source_span_pair import training_data
from scripts.preflight_source_span_residual import FIXED, sha, write


def run(output):
    started = time.monotonic()
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    code = [Path(__file__).resolve(), ROOT / 'hanglish/models/span_supervision.py',
            ROOT / 'hanglish/core/source_correspondence.py',
            ROOT / 'scripts/train_source_span_pair.py',
            ROOT / 'scripts/preflight_source_span_residual.py']
    pins = {str(p): sha(p) for p in [*FIXED, *code]}
    data, groups = training_data()
    counts = Counter(); components = set(); supervised = set()
    local = defaultdict(set); contextual = defaultdict(set)
    actions = Counter(); flags = Counter()
    with (output / 'preferences.jsonl').open('x') as stream:
        for row in data:
            pairs, row_counts = extract_preferences(row['input'], row['candidates'],
                row['positive_indices'], groups[row['id']])
            counts.update(row_counts); components.add(row['component_id'])
            if pairs:
                supervised.add(row['component_id'])
            record = {'schema': SCHEMA, **row, 'preferences': pairs, 'counts': row_counts}
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + '\n')
            for pair in pairs:
                ordered = sorted([pair['preferred_text'], pair['dispreferred_text']])
                direction = int(pair['preferred_text'] == ordered[1])
                local[digest([pair['source'], ordered])].add(direction)
                contextual[digest([row['input'], pair['input_start'], pair['input_end'],
                    pair['prefix'], pair['suffix'], ordered])].add(direction)
                for witness in pair['witnesses']:
                    for side in ('preferred', 'dispreferred'):
                        actions.update(f'{side}:{action}' for action in witness[side]['actions'])
                        if witness[side]['unresolved_ascii_output']:
                            flags[f'{side}:unresolved_ascii_witnesses'] += 1
    counts['rows'] = len(data); counts['components'] = len(components)
    counts['supervised_components'] = len(supervised)
    report = {
        'schema': SCHEMA, 'scope': 'authorized_fit_selected_candidates_only',
        'counts': dict(sorted(counts.items())), 'witness_actions': dict(sorted(actions.items())),
        'flags': dict(sorted(flags.items())),
        'local_unordered_pairs': len(local),
        'local_pairs_with_both_preferences': sum(len(v) == 2 for v in local.values()),
        'contextual_unordered_pairs': len(contextual),
        'contextual_pairs_with_both_preferences': sum(len(v) == 2 for v in contextual.values()),
        'labels': 'accepted full candidate preferred in the shared output context',
        'not_claimed': ['phonetic invalidity of dispreferred text',
                        'gold internal segmentation', 'unseen sentence accuracy',
                        'candidate generation improvement'],
        'weighting_contract': 'row mean of distinct contrasts, each mean of distinct path witnesses',
        'pins': pins, 'preferences_sha256': sha(output / 'preferences.jsonl'),
        'elapsed_seconds': time.monotonic() - started,
    }
    if not all(sha(p) == h for p, h in pins.items()):
        raise ValueError('input or producer changed during preparation')
    write(output / 'report.json', report)
    print(json.dumps({k: report[k] for k in ('counts', 'local_pairs_with_both_preferences',
        'contextual_pairs_with_both_preferences', 'elapsed_seconds')}, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
