"""Deterministic FIT projection for shared output-span features (no NN)."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_source_span_pair import training_data
from scripts.preflight_source_span_residual import FIXED, sha, write

SCHEMA = 'shared-edge-fit-inventory-v1'


def project(data, groups):
    targets = sorted({span.text for row in data for group in groups[row['id']]
                      for path in group for span in path.spans})
    target_indices = {text: i for i, text in enumerate(targets)}
    records = []; counts = Counter()
    for row in data:
        source = row['input']; paths = groups[row['id']]
        keys = sorted({(s.input_start, s.input_end, s.text)
                       for group in paths for path in group for s in path.spans})
        indices = {key: i for i, key in enumerate(keys)}
        edges = [{'input_start': a, 'input_end': b, 'text': text,
                  'target_index': target_indices[text]} for a, b, text in keys]
        path_groups = []
        for group in paths:
            values = []
            for path in sorted(group, key=lambda p: p.path_id):
                edge_indices = [indices[(s.input_start, s.input_end, s.text)] for s in path.spans]
                numerators = [s.input_end - s.input_start for s in path.spans]
                if sum(numerators) != len(source):
                    raise ValueError('path source-length mass differs')
                values.append({'path_id': path.path_id, 'edge_indices': edge_indices,
                               'weight_numerators': numerators, 'weight_denominator': len(source)})
            path_groups.append(values)
        records.append({**row, 'edges': edges, 'paths': path_groups})
        counts['rows'] += 1; counts['edges'] += len(edges)
        counts['candidates'] += len(row['candidates'])
        counts['paths'] += sum(map(len, path_groups))
        for a, b, text in keys:
            copied = source[a:b] == text
            counts['copy_edges' if copied else 'convert_edges'] += 1
            counts['unresolved_ascii_edges'] += int(not copied and any(
                'a' <= c <= 'z' or 'A' <= c <= 'Z' or '0' <= c <= '9' for c in text))
    counts['unique_target_texts'] = len(targets)
    counts['max_target_characters'] = max(map(len, targets))
    counts['max_edges_per_row'] = max(len(row['edges']) for row in records)
    return targets, records, dict(sorted(counts.items()))


def run(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    pins = {str(p): sha(p) for p in [*FIXED, Path(__file__).resolve(),
        ROOT / 'scripts/train_source_span_pair.py',
        ROOT / 'scripts/preflight_source_span_residual.py',
        ROOT / 'hanglish/core/source_correspondence.py']}
    data, groups = training_data(); targets, records, counts = project(data, groups)
    write(output / 'targets.json', targets)
    with (output / 'rows.jsonl').open('x') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
    if not all(sha(p) == h for p, h in pins.items()):
        raise ValueError('inventory inputs changed')
    report = {'schema': SCHEMA, 'scope': 'authorized_fixed_fit_only',
              'counts': counts, 'pins': pins,
              'targets_sha256': sha(output / 'targets.json'),
              'rows_sha256': sha(output / 'rows.jsonl'),
              'source_weights': 'codepoint lengths divided by complete source codepoint length',
              'limitations': ['Producer paths are alignment hypotheses, not gold segmentation.',
                              'Output feature extraction and learned edge scores are not run here.',
                              'Constant-score mass conservation does not imply segmentation-invariant scores.']}
    write(output / 'report.json', report); print(json.dumps(counts, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
