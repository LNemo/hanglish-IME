"""Publish full-Roman-sentence training/evaluation data without external hints.

Legacy sentence pairs are preserved. New natural references come from pinned
source documents, never from converter predictions. Their Roman inputs are
synthetic and explicitly carry that limitation in the manifest.
"""
from collections import Counter, defaultdict, deque
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.phonology import romanize_word

PARENT = ROOT / 'data/challenge/v2.2'
NATURAL = ROOT / 'data/natural_context/runtime-collision-source-v5'
REVIEW = ROOT / '.omx/artifacts/runtime-collision-source-review-v2.json'
OUTPUT = ROOT / 'data/challenge/sentence-v3'
NATURAL_COUNTS = {'train': 4096, 'dev': 256, 'test': 200}
VERSION = '3.0-sentence-only'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def natural_eligible(row, quarantined):
    text = row['text']
    return (24 <= len(text) <= 160 and len(text.split()) >= 4
            and text.rstrip('”’"\')]').endswith(('.', '?', '!'))
            and not re.search('[○●■□▲△▶▷◆◇☞]', text)
            and not any(word in text for word in quarantined)
            and not re.search(r'[\u1100-\u11ff\u3130-\u318f]', text)
            and all('가' <= c <= '힣' or c.isascii() or unicodedata.category(c)[0] in 'PZS'
                    for c in text))


def merge_sentences(rows, quarantined):
    merged = {}
    for row in rows:
        if not natural_eligible(row, quarantined):
            continue
        if row['text'] not in merged:
            merged[row['text']] = dict(text=row['text'], group_id=row['group_id'],
                                      source_ids=set(), origins=[], collisions=set())
        target = merged[row['text']]
        check(target['group_id'] == row['group_id'], 'duplicate text crosses document components')
        target['source_ids'].update(row['source_ids'])
        target['origins'].append(row['id'])
        target['collisions'].add(row['word_input'])
    return list(merged.values())


def held_partition(group):
    return 'dev' if int(digest('sentence-v3-held\0'+group), 16) % 5 == 0 else 'test'


def select_natural(rows, split, count, used_inputs, used_targets):
    buckets = defaultdict(list)
    for row in rows:
        if split != 'train' and held_partition(row['group_id']) != split:
            continue
        # Balancing uses only source corpus collision metadata, never predictions.
        family = min(row['collisions'], key=lambda s: digest('sentence-v3-family\0'+s))
        buckets[family].append(row)
    queues = {key: deque(sorted(values, key=lambda r: digest('sentence-v3-select\0'+r['text'])))
              for key, values in buckets.items()}
    selected, documents = [], set()
    while len(selected) < count:
        progressed = False
        for family in sorted(queues, key=lambda s: digest('sentence-v3-order\0'+s)):
            queue = queues[family]
            while queue:
                row = queue.popleft()
                if row['group_id'] in documents or row['text'] in used_targets:
                    continue
                source = romanize_word(row['text'])
                if source in used_inputs or any('가' <= c <= '힣' for c in source):
                    continue
                selected.append(dict(input=source, context='', expected=[row['text']],
                    category='natural_sentence_context', family='natural:'+row['group_id'],
                    split=split, provenance='KLUE-source-reference+synthetic-Roman-sentence',
                    context_role='none', boundary='word_spaced', group_id=row['group_id'],
                    source_ids=sorted(row['source_ids']), source_row_ids=sorted(row['origins']),
                    focus_families=sorted(row['collisions']), input_contract='roman_sentence_only',
                    reference_policy='observed_source_sentence; alternative readings not exhaustively annotated'))
                used_inputs.add(source); used_targets.add(row['text']); documents.add(row['group_id'])
                progressed = True
                break
            if len(selected) == count:
                break
        check(progressed, f'not enough disjoint natural {split} documents')
    return selected


def verify_splits(splits):
    inputs, targets, families, documents, sources = {}, {}, {}, {}, {}
    for split, rows in splits.items():
        for row in rows:
            check(row['context'] == '' and row['context_role'] == 'none', 'external context is forbidden')
            check(len(row['input'].split()) >= 2 and not re.search('[가-힣]', row['input']), 'full Roman input required')
            check(all(len(answer.split()) >= 2 for answer in row['expected']), 'whole-sentence targets required')
            check(row['split'] == split, 'split identity differs')
            for values, keys in ((inputs, [row['input']]), (targets, row['expected']),
                                 (families, [row['family']]),
                                 (documents, [row['group_id']]), (sources, row.get('source_ids', []))):
                for key in keys:
                    check(key not in values or values[key] == split, 'cross-split leakage')
                    values[key] = split
        check(len({r['id'] for r in rows}) == len(rows), 'duplicate case IDs')
        check(len({r['input'] for r in rows}) == len(rows), 'duplicate input within split')
    return dict(external_context_rows=0, single_token_rows=0, cross_split_input_overlap=0,
                cross_split_target_overlap=0, cross_split_document_overlap=0,
                cross_split_source_id_overlap=0, cross_split_family_overlap=0)


def build(output=OUTPUT):
    output = Path(output)
    check(not output.exists(), 'preserve existing dataset; use a new output path')
    parent = json.loads((PARENT/'manifest.json').read_text())
    natural = json.loads((NATURAL/'manifest.json').read_text())
    proof = json.loads((NATURAL/'independent-verification.json').read_text())
    check(proof['verified'] and proof['manifest_sha256'] == sha(NATURAL/'manifest.json'), 'source verification differs')
    pins = {}
    for filename, expected in parent['sha256'].items():
        path = PARENT/filename
        check(sha(path) == expected, 'legacy source differs')
        pins[str(path.relative_to(ROOT))] = expected
    for split in ('train', 'hold'):
        path = NATURAL/(split+'.jsonl')
        check(sha(path) == natural['outputs'][split]['sha256'], 'natural source differs')
        pins[str(path.relative_to(ROOT))] = sha(path)
    for path in (PARENT/'manifest.json', NATURAL/'manifest.json', NATURAL/'independent-verification.json',
                 REVIEW, Path(__file__), ROOT/'hanglish/core/phonology.py',
                 ROOT/'hanglish/core/predicate_boundaries.py',
                 ROOT/'hanglish/core/data/predicate_boundary_inventory.json'):
        pins[str(path.relative_to(ROOT))] = sha(path)
    review = json.loads(REVIEW.read_text())
    splits, excluded, used_inputs, used_targets = {}, {}, set(), set()
    for split in ('train', 'dev', 'test'):
        original = read_jsonl(PARENT/(split+'.jsonl'))
        excluded[split] = [r['id'] for r in original if r['context'] or len(r['input'].split()) < 2]
        splits[split] = []
        for row in original:
            if row['id'] in excluded[split]:
                continue
            row = dict(row, legacy_id=row['id'], input_contract='roman_sentence_only',
                       group_id='authored:'+row['family'], source_ids=[], context_role='none')
            splits[split].append(row)
            used_inputs.add(row['input']); used_targets.update(row['expected'])
    raw = {split: merge_sentences(read_jsonl(NATURAL/(split+'.jsonl')), review['quarantined_forms'])
           for split in ('train', 'hold')}
    for key in ('group_id', 'text'):
        check(not {r[key] for r in raw['train']} & {r[key] for r in raw['hold']}, 'raw TRAIN/HOLD overlap')
    check(not {s for r in raw['train'] for s in r['source_ids']} &
          {s for r in raw['hold'] for s in r['source_ids']}, 'raw source ID overlap')
    retained = {split: len(rows) for split, rows in splits.items()}
    for split, count in NATURAL_COUNTS.items():
        splits[split] += select_natural(raw['train' if split == 'train' else 'hold'],
                                       split, count, used_inputs, used_targets)
        for i, row in enumerate(splits[split], 1):
            row['id'] = f's3-{split}-{i:05d}'
    checks = verify_splits(splits)
    check(len(splits['test']) == 1000, 'full-sentence test must contain1000cases')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sentence-v3-', dir=output.parent) as tmp:
        staged = Path(tmp)/'corpus'; staged.mkdir()
        for split, rows in splits.items():
            (staged/(split+'.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        words = Counter(word for r in splits['train'] for answer in r['expected'] for word in re.findall('[가-힣]+', answer))
        lexicon = dict(train_sha256=sha(staged/'train.jsonl'), provenance=dict(source_split='train',
                       fields=['expected'], context_included=False),
                       entries=[dict(text=w, count=n) for w,n in sorted(words.items())])
        (staged/'train_lexicon.json').write_text(json.dumps(lexicon, ensure_ascii=False)+'\n')
        manifest = dict(schema_version=2, corpus_version=VERSION,
            input_contract='roman_sentence_only', model_input_fields=['input'],
            label_fields=['expected'], forbidden_input_fields=['context', 'expected', 'left_context', 'provenance'],
            counts={k:len(v) for k,v in splits.items()}, retained_legacy_sentence_counts=retained,
            added_natural_sentence_counts=NATURAL_COUNTS,
            category_counts={k:dict(Counter(r['category'] for r in v)) for k,v in splits.items()},
            excluded_legacy_hint_ids=excluded, checks=checks, source_hashes=pins,
            sha256={p.name:sha(p) for p in staged.iterdir()},
            selection_policy='source collision-family round robin; hash order; max1natural sentence/document; held document hash split',
            evaluation_policy='whole-output exact match against allowed references; report legacy/natural separately; no supplied Korean context',
            romanization_policy='legacy authored pairs retained; new natural inputs synthesized with pinned forward rules; whitespace and non-Hangul preserved',
            limitations=['Not a fresh blind benchmark: retained legacy cases and some source documents were used in prior experiments.',
                         'Synthetic Roman inputs may contain forward-rule errors; same-function round trips are not independent phonological verification.',
                         'Natural references are observed source text; not all alternative readings or source typos have been expert-audited.',
                         'Natural test is200distinct documents; legacy800cases include controlled template variants.'],
            prior_model_exposure='Do not claim unseen-document generalization for existing checkpoints; future runs must use these fixed splits.',
            license=natural['license'], attribution=natural['attribution'],
            training_run=False, model_evaluation_run=False)
        (staged/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
        (staged/'ATTRIBUTION.md').write_text('# Attribution\n\nNatural sentence references: '+natural['attribution']+'\n\nLicense: '+natural['license']+'. Romanized derivatives retain this attribution and license.\n')
        staged.rename(output)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=OUTPUT)
    result = build(parser.parse_args().output)
    print(json.dumps({k:result[k] for k in ('corpus_version','counts','checks')}, ensure_ascii=False))
