"""Conditional local preferences from accepted/rejected complete candidates.

These are context-dependent ranking labels, NOT binary phonetic-validity labels.
Only a single changed block at shared producer boundaries may receive a label.
No alignment is inferred inside a producer edge, and no whole-source contrast is
advertised as local supervision.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

from hanglish.core.source_correspondence import Path, contrast, require


SCHEMA = 'conditional-span-preference-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _side(path, block, name):
    start, end = block['input_start'], block['input_end']
    indices = [i for i, s in enumerate(path.spans)
               if start <= s.input_start and s.input_end <= end]
    require(indices and path.spans[indices[0]].input_start == start
            and path.spans[indices[-1]].input_end == end,
            'shared block does not cover complete producer spans')
    return {'path_id': path.path_id, 'span_indices': indices,
            'output_start': block[name]['output_start'],
            'output_end': block[name]['output_end'],
            'text': block[name]['text'],
            'actions': [path.spans[i].action for i in indices],
            'unresolved_ascii_output': any(
                path.spans[i].unresolved_ascii_output for i in indices)}


def extract_preferences(source, candidates, positive_indices, groups):
    """Return deduplicated text contrasts with every distinct path witness.

    A row should contribute the mean over distinct contrasts, and each contrast
    the mean over its witnesses. More generator paths must not increase a row's
    training weight. Multiple accepted candidates are never compared as rivals.
    """
    require(isinstance(source, str) and bool(source), 'nonempty source required')
    require(isinstance(candidates, (list, tuple)) and len(candidates) >= 2
            and all(isinstance(t, str) for t in candidates)
            and len(set(candidates)) == len(candidates), 'distinct candidates required')
    require(isinstance(positive_indices, (list, tuple)) and positive_indices
            and all(type(i) is int and 0 <= i < len(candidates) for i in positive_indices)
            and len(set(positive_indices)) == len(positive_indices)
            and len(positive_indices) < len(candidates), 'invalid accepted candidate indices')
    require(len(groups) == len(candidates), 'candidate path groups differ')
    for text, group in zip(candidates, groups):
        require(group and all(isinstance(p, Path) and p.source == source
                             and p.text == text for p in group), 'candidate path identity differs')
    positives = set(positive_indices)
    counts = Counter()
    records = {}
    for pi in sorted(positives):
        for ni in range(len(candidates)):
            if ni in positives:
                continue
            left = {p.path_id: p for p in groups[pi]}
            right = {p.path_id: p for p in groups[ni]}
            for pid in sorted(left):
                for nid in sorted(right):
                    counts['path_pairs'] += 1
                    p, n = left[pid], right[nid]
                    blocks = contrast(p, n)
                    changed = [b for b in blocks if b['relation'] == 'output_changed']
                    counts[f'changed_blocks_{len(changed)}'] += 1
                    if len(changed) != 1:
                        counts['excluded_multiple_or_zero_changes'] += 1
                        continue
                    if any(b['relation'] == 'segmentation_only' for b in blocks):
                        counts['excluded_outside_segmentation_changes'] += 1
                        continue
                    block = changed[0]
                    start, end = block['input_start'], block['input_end']
                    if start == 0 and end == len(source):
                        counts['excluded_whole_source'] += 1
                        continue
                    preferred = _side(p, block, 'left')
                    dispreferred = _side(n, block, 'right')
                    prefix = p.text[:preferred['output_start']]
                    suffix = p.text[preferred['output_end']:]
                    require(prefix == n.text[:dispreferred['output_start']]
                            and suffix == n.text[dispreferred['output_end']:],
                            'outside context differs for a local preference')
                    key = digest([source, start, end, pi, ni, prefix, suffix,
                                  preferred['text'], dispreferred['text']])
                    if key not in records:
                        records[key] = {
                            'preference_id': key, 'input_start': start, 'input_end': end,
                            'source': source[start:end], 'preferred_index': pi,
                            'dispreferred_index': ni, 'prefix': prefix, 'suffix': suffix,
                            'preferred_text': preferred['text'],
                            'dispreferred_text': dispreferred['text'], 'witnesses': []}
                    records[key]['witnesses'].append({
                        'preferred': preferred, 'dispreferred': dispreferred})
    intervals = {}
    for pair in records.values():
        key = (pair['preferred_index'], pair['dispreferred_index'])
        intervals.setdefault(key, set()).add((pair['input_start'], pair['input_end']))
    result = []
    for k in sorted(records):
        pair = records[k]
        if len(intervals[(pair['preferred_index'], pair['dispreferred_index'])]) != 1:
            counts['excluded_ambiguous_interval_path_pairs'] += len(pair['witnesses'])
            continue
        result.append(pair)
        counts['eligible_path_pairs'] += len(pair['witnesses'])
    counts['distinct_preferences'] = len(result)
    counts['rows_with_preferences'] = int(bool(result))
    return result, dict(sorted(counts.items()))
