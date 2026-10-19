"""Pure late-label coverage summaries for already verified recombination rows.

The caller must authenticate and close every source-only capture before supplying
FIT references. These helpers neither read files nor generate/select candidates.
"""
from collections import Counter


def need(value, message):
    if not value:
        raise ValueError(message)


def compare_row(capture, reference):
    need(capture.get('id') == reference.get('id')
         and capture.get('component_id') == reference.get('component_id')
         and capture.get('request') == {'input': reference.get('input'), 'context': ''},
         'capture/reference identity differs')
    old = reference.get('candidates')
    expected = reference.get('expected')
    proposed = capture.get('proposals')
    need(isinstance(old, list) and all(isinstance(x, str) for x in old)
         and len(set(old)) == len(old)
         and isinstance(expected, list) and expected
         and all(isinstance(x, str) for x in expected)
         and isinstance(proposed, list), 'candidate/reference domain differs')
    new = [row.get('text') for row in proposed]
    need(all(isinstance(x, str) and x for x in new)
         and len(new) == len(set(new)) and not set(new).intersection(old)
         and reference['input'] not in new, 'proposals are not novel')
    metadata = capture.get('source_reference', {})
    ranked = reference.get('ranked')
    need(type(ranked) is bool and metadata.get('ranked') is ranked
         and metadata.get('pool_rows') == len(old)
         and metadata.get('source_shard') == reference.get('source_shard'),
         'base pool denominator differs')
    need(capture.get('status') == ('generated' if old else 'no_pool')
         and (bool(old) is ranked), 'base pool status differs')
    need(old or not new, 'no-pool row unexpectedly generated proposals')
    fallback = capture.get('fallback_public')
    fallback_texts = []
    if not old:
        need(isinstance(fallback, list)
             and all(isinstance(row, dict) and isinstance(row.get('text'), str)
                     for row in fallback), 'unranked fallback missing')
        fallback_texts = [row['text'] for row in fallback]
        need(fallback_texts == reference.get('public_candidates'), 'unranked fallback changed')
    gold = set(expected)
    old_hit = bool(gold.intersection(old))
    new_hit = bool(gold.intersection(new))
    return dict(id=reference['id'], component_id=reference['component_id'],
                ranked=ranked, old_count=len(old), added_count=len(new),
                old_pool_covered=old_hit, added_pool_covered=new_hit,
                expanded_pool_covered=old_hit or new_hit,
                newly_covered=not old_hit and new_hit,
                fallback_public_covered=(not old and bool(gold.intersection(fallback_texts))))


def summarize(rows):
    need(rows and len(rows) == len({row['id'] for row in rows}), 'empty/duplicate rows')
    result = Counter(rows=len(rows))
    for row in rows:
        for name in ('ranked', 'old_pool_covered', 'added_pool_covered',
                     'expanded_pool_covered', 'newly_covered', 'fallback_public_covered'):
            result[name] += int(row[name])
        result['unranked'] += not row['ranked']
        result['old_candidates'] += row['old_count']
        result['added_candidates'] += row['added_count']
    need(result['expanded_pool_covered'] == result['old_pool_covered'] + result['newly_covered'],
         'union coverage identity differs')
    return dict(result)
