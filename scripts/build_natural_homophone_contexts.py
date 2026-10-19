"""Mine source-grounded spelling contrasts from unused KLUE TRAIN components."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.lattice import suffix_allowed
from hanglish.core.phonology import romanize_word
from scripts.build_natural_context_corpus import (
    DEFAULT_SOURCE, SPLIT_SALT, TRUSTED_ATTRIBUTION, canonicalize_occurrences,
    extract_dp_text, extract_mrc, extract_nli, guard_memory, sha256_file,
    stable_hash, verify_sources,
)

FROZEN = ROOT/'data/natural_context/klue-v1.1-pilot-v2'
FROZEN_HASHES = {'train.jsonl': '904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b',
                 'dev.jsonl': '8491eeb77a48d6b66efaf16e6a030badf8de489502be9ea0552729e3dff274c9'}
TRAIN = ROOT/'data/challenge/train.jsonl'
TRAIN_SHA = 'c7336caf348f0c4629c9ad65581d264b53df40a04474e67ae3bce18d501dd9b6'
OUTPUT = ROOT/'data/natural_context/homophone-source-v1'
SALT = 'natural-homophone-document-v1'


def load_inventory():
    if sha256_file(TRAIN) != TRAIN_SHA:
        raise ValueError('changed challenge TRAIN inventory')
    families = defaultdict(set)
    for row in map(json.loads, TRAIN.read_text().splitlines()):
        if row.get('category') == 'homophone_context':
            if row['split'] != 'train':
                raise ValueError('non-TRAIN spelling inventory')
            families[row['input']].update(row['expected'])
    if len(families) != 9 or sum(map(len,families.values())) != 20:
        raise ValueError('expected9 families/20 spellings')
    return {key: sorted(values) for key, values in sorted(families.items())}


class Contrasts:
    def __init__(self, families, suffixes):
        self.replacements = {word: sorted(set(values)-{word}) for values in families.values() for word in values}
        self.pattern = re.compile('|'.join(re.escape(w) for w in sorted(self.replacements,key=lambda w:(-len(w),w))))
        self.simple_forms = set(self.replacements)
        for word in self.replacements:
            # These three are an adverb or a bare predicate stem in the TRAIN inventory.
            if word in {'곧','같이','맞'}:
                continue
            self.simple_forms.update(word+s for s in suffixes if suffix_allowed(word,s))

    @lru_cache(maxsize=65536)
    def token(self, word):
        roman = romanize_word(word)
        candidates, changes = {word}, []
        for span in self.pattern.finditer(word):
            for alternative in self.replacements[span.group()]:
                candidate = word[:span.start()]+alternative+word[span.end():]
                if candidate != word and romanize_word(candidate) == roman:
                    candidates.add(candidate)
                    changes.append({'anchor': span.group(), 'replacement': alternative,
                                    'start': span.start(), 'end': span.end(), 'candidate': candidate})
        return roman, tuple(sorted(candidates)), tuple(changes)


def blocked_text_index(rows):
    index = defaultdict(set)
    for row in rows:
        text = row['text']
        if len(text) < 8:
            raise ValueError('unexpected short frozen sentence')
        index[text[:8]].add(text)
    return index


def embeds_blocked(text, index):
    return any(text.startswith(candidate, offset)
               for offset in range(max(0,len(text)-7))
               for candidate in index.get(text[offset:offset+8], ()))


def mine(rows, frozen, contrasts):
    excluded_groups = {r['group_id'] for r in frozen}
    excluded_sources = {s for r in frozen for s in r['source_ids']}
    blocked = blocked_text_index(frozen)
    records, counts = [], Counter()
    for row in rows:
        if row['group_id'] in excluded_groups or excluded_sources.intersection(row['source_ids']):
            counts['old_document_windows_excluded'] += 1
            continue
        if int(stable_hash(SPLIT_SALT,row['group_id'])[:16],16)%10 == 0:
            counts['original_dev_windows_excluded'] += 1
            continue
        if embeds_blocked(row['text'],blocked) or embeds_blocked(row['left_context'],blocked):
            counts['embedded_old_sentence_windows_excluded'] += 1
            continue
        for token in re.finditer('[가-힣]+',row['text']):
            word = token.group()
            if not contrasts.pattern.search(word):
                continue
            roman, candidates, changes = contrasts.token(word)
            if len(candidates) < 2:
                continue
            identity = row['text']+'\0'+str(token.start())+'\0'+str(token.end())
            split = 'hold' if int(stable_hash(SALT,row['group_id'])[:16],16)%5==0 else 'train'
            records.append({**row, 'id': 'nhc-'+stable_hash('natural-homophone-row-v1',identity)[:20],
                'word': word, 'word_input': roman, 'word_start': token.start(), 'word_end': token.end(),
                'spelling_anchors': sorted({c['anchor'] for c in changes}),
                'contrast_candidates': list(candidates), 'substitutions': list(changes),
                'match_kind': 'simple_form' if word in contrasts.simple_forms else 'compound_substring',
                'split': split, 'source_partition': 'upstream_train_and_existing_component_train',
                'candidate_origin': 'spelling substitutions; not runtime-generated candidates',
                'sense_annotation': False})
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate mined token identities')
    return sorted(records,key=lambda r:r['id']), dict(counts)


def select_training(rows, cap=64):
    queues = defaultdict(list)
    for row in rows:
        if row['split'] != 'train':
            continue
        for anchor in row['spelling_anchors']:
            queues[anchor].append(row)
    for queue in queues.values():
        queue.sort(key=lambda r:(r['match_kind'] != 'simple_form',stable_hash('natural-homophone-selection-v1',r['id'])))
    selected, seen, counts, documents = [], set(), Counter(), defaultdict(set)
    positions = Counter()
    while True:
        added = False
        for anchor in sorted(queues):
            if counts[anchor] >= cap:
                continue
            queue = queues[anchor]
            while positions[anchor] < len(queue):
                row = queue[positions[anchor]]; positions[anchor] += 1
                if row['id'] in seen or any(counts[a]>=cap or row['group_id'] in documents[a] for a in row['spelling_anchors']):
                    continue
                selected.append(row); seen.add(row['id']); added=True
                for a in row['spelling_anchors']:
                    counts[a]+=1; documents[a].add(row['group_id'])
                break
        if not added:
            break
    return selected


def build(output):
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    source_manifest, source_hashes, _=verify_sources(DEFAULT_SOURCE)
    pins={str((DEFAULT_SOURCE/name).relative_to(ROOT)):spec['sha256'] for name,spec in source_hashes.items()}
    frozen=[]
    for name,digest in FROZEN_HASHES.items():
        path=FROZEN/name
        if sha256_file(path)!=digest:
            raise ValueError('changed frozen natural corpus')
        pins[str(path.relative_to(ROOT))]=digest
        frozen.extend(json.loads(line) for line in path.read_text().splitlines())
    families=load_inventory()
    suffix_path=ROOT/'data/morphology_lexicon.json'
    contrasts=Contrasts(families,json.loads(suffix_path.read_text())['suffixes'])
    for path in (Path(__file__),ROOT/'scripts/build_natural_context_corpus.py',
                 ROOT/'hanglish/core/lattice.py',ROOT/'hanglish/core/phonology.py',TRAIN,suffix_path):
        pins[str(path.relative_to(ROOT))]=sha256_file(path)
    rejects=Counter()
    occurrences=extract_mrc(json.loads((DEFAULT_SOURCE/'klue-mrc-v1.1_train.json').read_text()),rejects)
    occurrences.extend(extract_nli(json.loads((DEFAULT_SOURCE/'klue-nli-v1.1_train.json').read_text()),rejects))
    occurrences.extend(extract_dp_text((DEFAULT_SOURCE/'klue-dp-v1.1_train.tsv').read_text(),rejects))
    rows=canonicalize_occurrences(occurrences)
    del occurrences
    records,exclusions=mine(rows,frozen,contrasts)
    selected=select_training(records)
    outputs={}
    for role,subset in [('train',[r for r in records if r['split']=='train']),
                        ('hold',[r for r in records if r['split']=='hold']),('selected_train',selected)]:
        path=output/(role+'.jsonl')
        path.write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in subset))
        counts=Counter(a for row in subset for a in row['spelling_anchors'])
        outputs[role]={'file':path.name,'rows':len(subset),'documents':len({r['group_id'] for r in subset}),
                       'sha256':sha256_file(path),'anchor_counts':dict(sorted(counts.items())),
                       'match_kinds':dict(Counter(r['match_kind'] for r in subset))}
    for name,digest in pins.items():
        if sha256_file(ROOT/name)!=digest:
            raise ValueError('source changed during mining')
    manifest={'schema':'natural-homophone-context-source-v1','completed':True,
        'training_ready':False,'next_required':'actual source-only runtime candidate capture and source-spelling review',
        'model_inference_performed':False,'production_changed':False,'sense_annotations':False,
        'family_inventory':families,'pins':pins,'source_revision':source_manifest['revision'],
        **TRUSTED_ATTRIBUTION,'source_windows':len(rows),'frozen_excluded_groups':len({r['group_id'] for r in frozen}),
        'exclusions':exclusions,'outputs':outputs,'hold_policy':'document hash20%; no model evaluation or training in this build',
        'selection_policy':'TRAIN-only, at most64 per anchor, one document per anchor, simple forms first',
        'elapsed_s':time.monotonic()-started,'peak_rss_bytes':guard_memory('homophone source build')}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUTPUT)
    build(parser.parse_args().output)
