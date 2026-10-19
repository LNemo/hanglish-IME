"""Strict source/target separation for the sentence-only corpus."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

DEFAULT_CORPUS = Path(__file__).resolve().parents[1]/'data/challenge/sentence-v3'


@dataclass(frozen=True)
class SentenceExample:
    id: str
    input: str
    targets: tuple[str, ...]


def model_input(row):
    if row.get('context') or row.get('context_role') != 'none' or row.get('left_context'):
        raise ValueError('Korean hints and external context are forbidden')
    source = row['input']
    if not isinstance(source, str) or len(source.split()) < 2 or re.search('[가-힣]', source):
        raise ValueError('expected a complete Roman sentence')
    if row.get('input_contract') != 'roman_sentence_only':
        raise ValueError('missing sentence-only input contract')
    return source


def read_split(corpus, split, *, allow_test=False):
    if split not in ('train', 'dev', 'test') or (split == 'test' and not allow_test):
        raise ValueError('test access requires explicit evaluation mode')
    corpus = Path(corpus)
    manifest = json.loads((corpus/'manifest.json').read_text())
    if manifest.get('input_contract') != 'roman_sentence_only' or manifest.get('model_input_fields') != ['input']:
        raise ValueError('incorrect corpus input contract')
    name = split+'.jsonl'; raw = (corpus/name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['sha256'][name]:
        raise ValueError('corpus checksum mismatch')
    rows = [json.loads(line) for line in raw.splitlines() if line]
    if len(rows) != manifest['counts'][split] or any(row['split'] != split for row in rows):
        raise ValueError('corpus split/count mismatch')
    for row in rows:
        model_input(row)
        if not row['expected'] or any(not isinstance(t, str) or len(t.split()) < 2 for t in row['expected']):
            raise ValueError('whole-sentence targets required')
    return rows


def load_training_data(corpus=DEFAULT_CORPUS):
    # This path never opens test.jsonl or uses labels as encoder context.
    return {split: [SentenceExample(r['id'], model_input(r), tuple(r['expected']))
                    for r in read_split(corpus, split)] for split in ('train', 'dev')}


def encoder_inputs(examples):
    return [example.input for example in examples]
