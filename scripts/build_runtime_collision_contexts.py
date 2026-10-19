"""Mine whole-word romanization collisions without using evaluation labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.lattice import CandidateGenerator
from hanglish.core.lexical_data import FILES
from hanglish.core.phonology import romanize_word
from scripts.build_natural_context_corpus import (
    DEFAULT_SOURCE, SPLIT_SALT, TRUSTED_ATTRIBUTION, canonicalize_occurrences,
    extract_dp_text, extract_mrc, extract_nli, guard_memory, sha256_file,
    stable_hash, verify_sources,
)
from scripts.build_natural_homophone_contexts import (
    FROZEN, FROZEN_HASHES, SALT, blocked_text_index, embeds_blocked,
)

OUTPUT = ROOT / 'data/natural_context/runtime-collision-source-v1'
PREVIOUS = ROOT / 'data/natural_context/homophone-source-v1'
PREVIOUS_MANIFEST_SHA = '5c49a872942c126818ce5cbcc42cad434e3b60cdf7098dfe1097b535d35d25f3'


def inventory(indexes):
    """Retain complete pre-pruning families, including unobserved alternatives."""
    families = {}
    for source in sorted(set().union(*(index.keys() for index in indexes))):
        spellings = {word for index in indexes for word in index.get(source, {})
                     if re.fullmatch('[가-힣]{1,12}', word) and romanize_word(word) == source}
        if len(spellings) > 1:
            families[source] = sorted(spellings)
    return families


def document_split(group):
    # Reuse the earlier homophone split: previously held documents remain held.
    return 'hold' if int(stable_hash(SALT, group)[:16], 16) % 5 == 0 else 'train'


def selector_held(group):
    return int(stable_hash('natural-homophone-selector-doc-v1', group), 16) % 5 == 0


def mine(rows, frozen, families, previous_hold=()):
    words = {word: source for source, forms in families.items() for word in forms}
    excluded_groups = {r['group_id'] for r in frozen}
    excluded_sources = {s for r in frozen for s in r['source_ids']}
    blocked = blocked_text_index(frozen)
    held_texts = [{'text': text} for row in previous_hold
                  for text in (row['text'], row['left_context']) if len(text) >= 8]
    held_blocked = blocked_text_index(held_texts)
    records, counts = [], Counter()
    for row in rows:
        if row['group_id'] in excluded_groups or excluded_sources.intersection(row['source_ids']):
            counts['old_document_windows_excluded'] += 1
            continue
        if int(stable_hash(SPLIT_SALT, row['group_id'])[:16], 16) % 10 == 0:
            counts['original_dev_windows_excluded'] += 1
            continue
        if embeds_blocked(row['text'], blocked) or embeds_blocked(row['left_context'], blocked):
            counts['embedded_old_sentence_windows_excluded'] += 1
            continue
        if document_split(row['group_id']) == 'train':
            if selector_held(row['group_id']):
                counts['previous_selector_hold_partition_windows_excluded'] += 1
                continue
            if embeds_blocked(row['text'], held_blocked) or embeds_blocked(row['left_context'], held_blocked):
                counts['embedded_previous_hold_windows_excluded'] += 1
                continue
        for token in re.finditer('[가-힣]+', row['text']):
            word = token.group()
            if word not in words:
                continue
            source = words[word]
            identity = row['text'] + '\0' + str(token.start()) + '\0' + str(token.end())
            records.append({**row, 'id': 'rcc-' + stable_hash('runtime-collision-row-v1', identity)[:20],
                'word': word, 'word_input': source, 'word_start': token.start(), 'word_end': token.end(),
                'contrast_candidates': families[source], 'split': document_split(row['group_id']),
                'match_kind': 'whole_word', 'sense_annotation': False,
                'source_partition': 'upstream_train_and_existing_component_train',
                'candidate_origin': 'pre-pruning runtime index; not validated words or scored candidates'})
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate source token identities')
    return sorted(records, key=lambda r: r['id']), dict(counts)


def load_previous_hold():
    """Held text is used only as an exclusion index, never a mining target."""
    path = PREVIOUS/'manifest.json'
    if sha256_file(path) != PREVIOUS_MANIFEST_SHA:
        raise ValueError('previous corpus manifest changed')
    manifest = json.loads(path.read_text())
    path = PREVIOUS/manifest['outputs']['hold']['file']
    if sha256_file(path) != manifest['outputs']['hold']['sha256']:
        raise ValueError('previous held corpus changed')
    return [json.loads(line) for line in path.read_text().splitlines()]


def select_training(rows, *, limit=2048, per_spelling=8, per_family=32):
    """Round-robin families/forms; TRAIN only, one document per family."""
    if min(limit, per_spelling, per_family) < 1:
        raise ValueError('selection limits must be positive')
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['split'] == 'train':
            grouped[row['word_input']][row['word']].append(row)
    queues = {}
    for source, forms in grouped.items():
        if len(forms) < 2:
            continue
        ordered = sorted(forms, key=lambda w: (len({r['group_id'] for r in forms[w]}), w))
        queues[source] = [(word, deque(sorted(forms[word], key=lambda r: stable_hash('rcc-selection-v1', r['id']))))
                          for word in ordered]
    selected, documents, form_counts, family_counts = [], defaultdict(set), Counter(), Counter()
    sources = sorted(queues, key=lambda s: stable_hash('rcc-family-order-v1', s))
    while len(selected) < limit:
        changed = False
        for source in sources:
            if family_counts[source] >= per_family:
                continue
            # Fewest already selected first; initial tie order favors rare forms.
            forms = sorted(queues[source], key=lambda item: form_counts[source, item[0]])
            for word, queue in forms:
                if form_counts[source, word] >= per_spelling:
                    continue
                while queue and queue[0]['group_id'] in documents[source]:
                    queue.popleft()
                if not queue:
                    continue
                row = queue.popleft()
                selected.append(row)
                documents[source].add(row['group_id'])
                form_counts[source, word] += 1
                family_counts[source] += 1
                changed = True
                break
            if len(selected) == limit:
                break
        if not changed:
            break
    return selected


def validate_previous_split(records):
    """Check split metadata only; do not inspect held text or run a model."""
    manifest_path = PREVIOUS / 'manifest.json'
    if sha256_file(manifest_path) != PREVIOUS_MANIFEST_SHA:
        raise ValueError('previous corpus manifest changed')
    manifest = json.loads(manifest_path.read_text())
    groups, sources = {}, {}
    for split in ('train', 'hold'):
        path = PREVIOUS / manifest['outputs'][split]['file']
        if sha256_file(path) != manifest['outputs'][split]['sha256']:
            raise ValueError('previous corpus data changed')
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['split'] != split or document_split(row['group_id']) != split:
                    raise ValueError('previous split disagrees with reused policy')
                if row['group_id'] in groups and groups[row['group_id']] != split:
                    raise ValueError('previous document crosses splits')
                groups[row['group_id']] = split
                for source in row['source_ids']:
                    if source in sources and sources[source] != split:
                        raise ValueError('previous source crosses splits')
                    sources[source] = split
    overlaps = Counter()
    for row in records:
        prior = {groups[row['group_id']]} if row['group_id'] in groups else set()
        prior.update(sources[s] for s in row['source_ids'] if s in sources)
        if prior and prior != {row['split']}:
            raise ValueError('new corpus crosses previous split')
        if prior:
            overlaps[row['split']] += 1
    return dict(overlaps)


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_manifest, source_hashes, _ = verify_sources(DEFAULT_SOURCE)
    generator = CandidateGenerator()
    families = inventory([generator.index, generator.dictionary_index, generator.inflections])
    external_fingerprint, external_version = generator.external_fingerprint, generator.external_version
    del generator
    frozen = []
    pins = {str((DEFAULT_SOURCE/name).relative_to(ROOT)): spec['sha256'] for name, spec in source_hashes.items()}
    for name, digest in FROZEN_HASHES.items():
        path = FROZEN / name
        if sha256_file(path) != digest:
            raise ValueError('changed frozen natural corpus')
        pins[str(path.relative_to(ROOT))] = digest
        frozen.extend(json.loads(line) for line in path.read_text().splitlines())
    files = [Path(__file__), ROOT/'scripts/build_natural_context_corpus.py',
             ROOT/'scripts/build_natural_homophone_contexts.py', ROOT/'hanglish/core/lattice.py',
             ROOT/'hanglish/core/phonology.py', ROOT/'hanglish/core/lexical_data.py',
             ROOT/'hanglish/core/predicate_boundaries.py',
             ROOT/'hanglish/core/productive_endings.py',
             ROOT/'hanglish/core/data/predicate_boundary_inventory.json',
             PREVIOUS/'manifest.json', PREVIOUS/'train.jsonl', PREVIOUS/'hold.jsonl',
             *[ROOT/'data'/n for n in ('morphology_lexicon.json','workbench_lexicon.json','challenge/train_lexicon.json')],
             *[ROOT/'data/external/mecab-ko-dic'/(name+'.csv') for name in FILES]]
    pins.update({str(path.relative_to(ROOT)): sha256_file(path) for path in files})
    rejects = Counter()
    occurrences = extract_mrc(json.loads((DEFAULT_SOURCE/'klue-mrc-v1.1_train.json').read_text()), rejects)
    occurrences.extend(extract_nli(json.loads((DEFAULT_SOURCE/'klue-nli-v1.1_train.json').read_text()), rejects))
    occurrences.extend(extract_dp_text((DEFAULT_SOURCE/'klue-dp-v1.1_train.tsv').read_text(), rejects))
    rows = canonicalize_occurrences(occurrences)
    del occurrences
    guard_memory('canonical runtime collision source')
    records, exclusions = mine(rows, frozen, families, load_previous_hold())
    previous_overlap = validate_previous_split(records)
    selected = select_training(records)
    inventory_path = output/'inventory.json'
    inventory_path.write_text(json.dumps(families, ensure_ascii=False, indent=2)+'\n')
    outputs = {}
    for role, subset in [('train', [r for r in records if r['split']=='train']),
                         ('hold', [r for r in records if r['split']=='hold']), ('selected_train', selected)]:
        path = output/(role+'.jsonl')
        with path.open('w') as stream:
            for row in subset:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n')
        counts = defaultdict(Counter)
        for row in subset:
            counts[row['word_input']][row['word']] += 1
        outputs[role] = {'file': path.name, 'rows': len(subset), 'sha256': sha256_file(path),
            'documents': len({r['group_id'] for r in subset}), 'families': len(counts),
            'multi_observed_families': sum(len(c)>1 for c in counts.values()),
            'form_counts': {s: dict(sorted(c.items())) for s,c in sorted(counts.items())}}
    for name, digest in pins.items():
        if sha256_file(ROOT/name) != digest:
            raise ValueError('source changed during mining')
    manifest = {'schema': 'runtime-collision-context-source-v1', 'completed': True,
        'training_ready': False, 'model_inference_performed': False, 'production_changed': False,
        'next_required': 'source spelling review and actual runtime candidate capture',
        'sense_annotations': False, 'all_spellings_validated': False,
        'inventory': {'file': inventory_path.name, 'families': len(families), 'sha256': sha256_file(inventory_path)},
        'inventory_scope': 'precomputed lexical and inflection indexes; dynamic ending and compound proposals are not enumerated',
        'generator_external_fingerprint': external_fingerprint, 'wordfreq_version': external_version,
        'pins': pins, 'source_revision': source_manifest['revision'], **TRUSTED_ATTRIBUTION,
        'source_windows': len(rows), 'exclusions': exclusions, 'outputs': outputs,
        'previous_corpus_overlapping_rows': previous_overlap,
        'previous_hold_access': 'document/source split checks and text exclusion index only; no model use or qualitative text review',
        'previous_selector_hold_policy': 'exclude all source-TRAIN documents in natural-homophone-selector-doc-v1 hold partition before selection',
        'hold_policy': 'reuse natural-homophone-document-v1 document hash20%; not all documents are new',
        'selection_policy': 'TRAIN only; require2 observed forms per family; round-robin families/forms; max2048 total,8 per form,32 per family,1 document per family',
        'elapsed_s': time.monotonic()-started, 'peak_rss_bytes': guard_memory('runtime collision source build')}
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: manifest[k] for k in ('completed','source_windows','elapsed_s','peak_rss_bytes')}, ensure_ascii=False))
    print(json.dumps({k: {n:v for n,v in info.items() if n!='form_counts'} for k,info in outputs.items()}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
