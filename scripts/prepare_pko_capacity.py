"""Target-insensitive, category-balanced FIT selection for the PKO control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_pko_seq2seq as trainer
from scripts import preflight_pko_seq2seq as preflight

SEED = 'pko-capacity-control-v1-20260921'
QUOTAS = {'liaison': 8, 'nasal_liquid_assimilation': 8, 'tense_consonants': 8,
          'vowel_sequences': 8, 'copula_assimilation': 4, 'palatalization': 4,
          'english_particle': 8, 'protected_token': 8,
          'natural_sentence_context': 4, 'natural_sentence_oov_growth': 2,
          'natural_sentence_oov_growth_v3': 2}


def select(metadata, fit_ids, groups, quotas=QUOTAS):
    selected, seen_groups = [], set()
    for category, count in quotas.items():
        candidates = sorted((r for r in metadata if r['id'] in fit_ids
                             and r['category'] == category),
                            key=lambda r: hashlib.sha256((SEED+'\n'+r['id']).encode()).hexdigest())
        chosen = []
        for row in candidates:
            group = groups[row['id']]
            if group in seen_groups:
                continue
            seen_groups.add(group); chosen.append(row['id'])
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError('insufficient independent FIT groups: '+category)
        selected.extend(chosen)
    return selected


def donors(selected):
    ordered = sorted(selected, key=lambda r: (len(r['source']), r['id']))
    result = {}
    for bin_index in range(4):
        bucket = ordered[len(ordered)*bin_index//4:len(ordered)*(bin_index+1)//4]
        offset = next((k for k in range(1, len(bucket)) if all(
            r['source'] != bucket[(i+k)%len(bucket)]['source'] for i, r in enumerate(bucket))), None)
        if offset is None:
            raise ValueError('no distinct-source derangement in length bin')
        for i, row in enumerate(bucket):
            donor = bucket[(i+offset)%len(bucket)]
            result[row['id']] = {'donor_id': donor['id'], 'shuffled_source': donor['source'],
                                 'length_bin': bin_index}
    return result


def prepare(output):
    fit, inner, partition_proof = trainer.load_train_partition()
    fit_by_id = {r['id']: r for r in fit}
    partition = json.loads(trainer.PARTITION.read_text())
    groups = {ident: g['group_sha256'] for g in partition['groups'] for ident in g['ids']}
    metadata = []
    for path in (preflight.BASE/'train.jsonl', preflight.EXTRA/'train.jsonl'):
        metadata.extend(json.loads(line) for line in path.read_bytes().splitlines() if line)
    ids = select(metadata, set(fit_by_id), groups)
    categories = {r['id']: r['category'] for r in metadata}
    # Labels are attached only after category/group/ID selection is fixed.
    selected = [{**fit_by_id[ident], 'group': groups[ident], 'category': categories[ident]}
                for ident in ids]
    donor_by_id = donors(selected)
    rows = [{**r, **donor_by_id[r['id']]} for r in selected]
    assert len(rows) == 64 and len({r['group'] for r in rows}) == 64
    assert not set(ids).intersection(r['id'] for r in inner)
    pins = {preflight.path_key(p): h for p, h in preflight.EXPECTED_FILES.items()
            if p != preflight.BASE/'dev.jsonl'}
    for path in [Path(__file__), Path(trainer.__file__), Path(preflight.__file__)]:
        pins[preflight.path_key(path)] = preflight.sha256_file(path)
    result = {'schema': SEED, 'completed': True, 'seed': SEED, 'quotas': QUOTAS,
              'rows': rows, 'rows_count': 64, 'unique_groups': 64,
              'selected_without_targets_or_model_results': True,
              'fit_only': True, 'partition': partition_proof, 'pins': pins,
              'dev_opened': False, 'test_opened': False, 'model_loaded': False}
    output = Path(output)
    with output.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    result = prepare(parser.parse_args().output)
    print(json.dumps({'rows': result['rows_count'], 'unique_groups': result['unique_groups'],
                      'model_loaded': False, 'dev_opened': False, 'test_opened': False}))
