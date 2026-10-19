"""Prepare immutable, source-only sentence candidates for a natural-context pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_natural_context_corpus import verify_sources, DEFAULT_SOURCE, split_isolation_assertions
from hanglish.models.training_guard import validate_training_splits
from hanglish.core.phonology import romanize_word
from hanglish.core.neural_converter import file_sha256

CORPUS = ROOT / 'data/natural_context/klue-v1.1-pilot-v2'
OUTPUT = ROOT / 'data/natural_context/ranker-pilot-v1'
PINS = {'train.jsonl': '904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b',
        'dev.jsonl': '8491eeb77a48d6b66efaf16e6a030badf8de489502be9ea0552729e3dff274c9'}
LIMITS = {'natural_train': 512, 'natural_dev': 128, 'retention_train': 512}
SALT = 'natural-ranker-source-selection-v1'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def select_rows(rows, role, limit):
    return sorted(rows, key=lambda row: hashlib.sha256(
        (SALT + '\0' + role + '\0' + row['id']).encode()).hexdigest())[:limit]


def selected_sources():
    verify_sources(DEFAULT_SOURCE)
    natural = {}
    for name, expected in PINS.items():
        if file_sha256(CORPUS / name) != expected:
            raise ValueError('natural source differs from pinned corpus')
        natural[name] = read_rows(CORPUS / name)
    split_isolation_assertions(natural['train.jsonl'], natural['dev.jsonl'])
    challenge = ROOT / 'data/challenge'
    guard = validate_training_splits(challenge / 'train.jsonl', challenge / 'dev.jsonl',
                                     challenge / 'manifest.json')
    selected = {
        'natural_train': select_rows(natural['train.jsonl'], 'natural_train', LIMITS['natural_train']),
        'natural_dev': select_rows(natural['dev.jsonl'], 'natural_dev', LIMITS['natural_dev']),
        'retention_train': select_rows(read_rows(challenge / 'train.jsonl'), 'retention_train', LIMITS['retention_train']),
    }
    return selected, guard


def source_fields(row, role):
    natural = role.startswith('natural_')
    return {'id': row['id'], 'origin': 'natural' if natural else 'retention',
            'split': 'dev' if role == 'natural_dev' else 'train',
            'group_id': row['group_id'] if natural else row['family'],
            'input': romanize_word(row['text']) if natural else row['input'],
            'context': row['left_context'] if natural else row['context'],
            'expected': [row['text']] if natural else row['expected']}


def prepare_row(row, role, propose):
    natural = role.startswith('natural_')
    source = romanize_word(row['text']) if natural else row['input']
    context = row['left_context'] if natural else row['context']
    # Proposer sees only input and preceding context. Read targets afterwards.
    candidates = list(dict.fromkeys(propose(source, context)))
    fields = source_fields(row, role)
    fields.update(candidates=candidates, positive_indices=[
        i for i, candidate in enumerate(candidates) if candidate in fields['expected']])
    return fields


def validate_row(row, source, role):
    expected = source_fields(source, role)
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError('prepared row differs from selected source or split')
    candidates = row.get('candidates')
    if (not isinstance(candidates, list) or not 1 <= len(candidates) <= 48
            or any(not isinstance(c, str) or not c for c in candidates)
            or len(set(candidates)) != len(candidates)):
        raise ValueError('invalid candidate pool')
    positives = row.get('positive_indices')
    if (not isinstance(positives, list) or any(type(i) is not int for i in positives)
            or positives != [i for i, c in enumerate(candidates) if c in expected['expected']]):
        raise ValueError('invalid positive indices')


def frozen_hashes():
    from scripts.train_sentence_ranker import runtime_proposal_artifact_hashes, frozen_source_hashes
    hashes = frozen_source_hashes()
    hashes.update(runtime_proposal_artifact_hashes())
    for path in [Path(__file__).resolve(), ROOT / 'scripts/build_natural_context_corpus.py',
                 ROOT / 'hanglish/models/training_guard.py', CORPUS / 'manifest.json',
                 CORPUS / 'train.jsonl', CORPUS / 'dev.jsonl', DEFAULT_SOURCE / 'SOURCE.json',
                 ROOT / 'data/common_english.txt', ROOT / 'data/it_terms.txt']:
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    return hashes


def validate_prepared(directory, manifest_sha256):
    path = directory / 'manifest.json'
    if file_sha256(path) != manifest_sha256:
        raise ValueError('prepared manifest differs from explicitly pinned hash')
    manifest = json.loads(path.read_text())
    if (manifest.get('schema_version') != 1 or manifest.get('selection_salt') != SALT
            or manifest.get('limits') != LIMITS or manifest.get('no_gold_injection') is not True
            or manifest.get('test_data_read') is not False):
        raise ValueError('invalid preparation policy')
    sources, guard = selected_sources()
    if manifest['frozen_hashes'] != frozen_hashes() or manifest['challenge_guard'] != guard:
        raise ValueError('candidate or source artifacts changed since preparation')
    if set(manifest['outputs']) != set(LIMITS):
        raise ValueError('invalid output inventory')
    result = {}
    for role, selected in sources.items():
        output = directory / (role + '.jsonl')
        expected = manifest['outputs'][role]
        if file_sha256(output) != expected['sha256'] or output.stat().st_size != expected['bytes']:
            raise ValueError('prepared output integrity mismatch')
        rows = read_rows(output)
        if len(rows) != len(selected) or len(rows) != expected['rows']:
            raise ValueError('prepared row count mismatch')
        for row, source in zip(rows, selected):
            validate_row(row, source, role)
        result[role] = rows
    return manifest, result


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    sources, guard = selected_sources()
    before = frozen_hashes()
    from hanglish.core.neural_converter import NeuralConverter
    converter = NeuralConverter(ranker_kind='semantic')
    converter.load_models()
    output.mkdir(parents=True)
    started = time.monotonic()
    outputs = {}
    total = 0
    for role, rows in sources.items():
        covered = 0
        path = output / (role + '.jsonl')
        with path.open('x') as stream:
            for source in rows:
                row = prepare_row(source, role, lambda text, context: [
                    p[0] for p in converter.sentence_proposals(text, context)['paths']])
                validate_row(row, source, role)
                covered += bool(row['positive_indices'])
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
                total += 1
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if sys.platform != 'darwin':
                    rss *= 1024
                if rss > 3 * 1024**3 or time.monotonic() - started > 1800:
                    raise RuntimeError('preparation resource bound exceeded; retain incomplete files for diagnosis')
                if total % 64 == 0:
                    print(json.dumps({'event': 'prepare', 'completed': total,
                                      'elapsed_seconds': round(time.monotonic() - started, 2)}), flush=True)
        outputs[role] = {'rows': len(rows), 'covered': covered, 'bytes': path.stat().st_size,
                         'sha256': file_sha256(path)}
    if before != frozen_hashes():
        raise RuntimeError('sources changed during preparation')
    manifest = {'schema_version': 1, 'selection_salt': SALT, 'limits': LIMITS,
                'no_gold_injection': True, 'test_data_read': False,
                'outputs': outputs, 'challenge_guard': guard, 'frozen_hashes': before,
                'runtime_version': converter.version, 'elapsed_seconds': time.monotonic() - started,
                'peak_rss_bytes': rss,
                'limitations': ['Current bounded forward Romanizer generates synthetic input.',
                                'Natural source spelling is the observed positive; other valid homophones are not exhaustively annotated.',
                                'Uncovered rows are retained and count as evaluation failures.']}
    path = output / 'manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    digest = file_sha256(path)
    validate_prepared(output, digest)
    print(json.dumps({'manifest': str(path), 'sha256': digest, 'outputs': outputs}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    prepare(parser.parse_args().output)
