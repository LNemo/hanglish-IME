"""Capture expanded, source-only word and sentence features from current runtime."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.neural_converter import NeuralConverter, file_sha256
from scripts.prepare_context_training import source_hashes as base_hashes, mixed_split_guard
from scripts.prepare_natural_ranker import (
    selected_sources, CORPUS, read_rows, select_rows, source_fields,
)
from scripts.prepare_natural_word_ranker import Recorder, capture_groups, target_spans
from scripts.train_natural_sentence_ranker import rss_bytes

OUTPUT = ROOT/'data/natural_context/current-context-v1'
RUNTIME_VERSION = 'neural-1-addbecef42dc'
COUNTS = {'natural_train': 2048, 'natural_dev': 128, 'retention_train': 1820}
PROFILE = {'name': 'current-expanded-context-v1', 'counts': COUNTS, 'threads': 2,
           'candidate_limit': 48, 'rss_limit_bytes': 3*1024**3,
           'time_limit_seconds': 10800, 'label_policy': 'all observed positives after actual runtime inference'}


def source_hashes():
    values = base_hashes()
    for name in ('scripts/prepare_current_context_data.py',
                 'scripts/train_natural_sentence_ranker.py'):
        values[name] = file_sha256(ROOT/name)
    return values


def load_sources():
    # Existing loader verifies both full source files, document split isolation,
    # and the challenge train/dev hashes before we expand the natural selection.
    selected, guard = selected_sources()
    selected['natural_train'] = select_rows(read_rows(CORPUS/'train.jsonl'),
                                            'natural_train', COUNTS['natural_train'])
    selected['retention_train'] = read_rows(ROOT/'data/challenge/train.jsonl')
    sources = {role: [source_fields(row, role) for row in selected[role]] for role in COUNTS}
    if any(len(sources[role]) != count for role, count in COUNTS.items()):
        raise ValueError('expanded source selection is incomplete')
    split_guard = mixed_split_guard(sources)
    return sources, guard, split_guard


def selection_digest(sources):
    return hashlib.sha256(json.dumps(sources, ensure_ascii=False,
                                     sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class SentenceRecorder:
    def __init__(self, delegate):
        self.delegate, self.candidates, self.scores = delegate, None, None

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def score_pairs(self, contexts, candidates):
        if self.candidates is not None:
            raise ValueError('unexpected repeated sentence scoring call')
        self.candidates = list(candidates)
        self.scores = list(map(float, self.delegate.score_pairs(contexts, candidates)))
        return self.scores


class CaptureConverter(NeuralConverter):
    def _prepare_candidates(self, *args, **kwargs):
        result = super()._prepare_candidates(*args, **kwargs)
        self.static_pieces = result[1]
        return result


def align_groups(source, groups):
    targets = target_spans(source['input'], source['expected'])
    result = []
    for group in groups:
        expected = sorted(targets.get((group['start'], group['end']), ()))
        result.append({**group, 'id': f"{source['id']}:{group['start']}:{group['end']}",
                       'expected': expected, 'label_aligned': bool(expected),
                       'positive_indices': [i for i, c in enumerate(group['candidates']) if c in expected]})
    return result


def visible_candidates(ranked, original):
    values = ranked[:5]
    return values if original in values else values[:4] + [original]


def validate_groups(source, groups):
    if not isinstance(groups, list):
        raise ValueError('invalid word group collection')
    seen = set()
    for group in groups:
        start, end = group['start'], group['end']
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(source['input'])
                or (start, end) in seen or source['input'][start:end] != group['source']):
            raise ValueError('invalid or duplicated word span')
        seen.add((start, end))
        candidates = group['candidates']
        if (not isinstance(candidates, list) or not candidates
                or any(not isinstance(c, str) or not c for c in candidates)
                or len(set(candidates)) != len(candidates)):
            raise ValueError('invalid word candidates')
        for key in ('base_scores', 'teacher_scores', 'kinds'):
            if not isinstance(group[key], list) or len(group[key]) != len(candidates):
                raise ValueError('word feature length mismatch')
        if any(type(n) not in (int, float) or not math.isfinite(n)
               for key in ('base_scores', 'teacher_scores') for n in group[key]):
            raise ValueError('nonfinite word feature')
        if any(not isinstance(kind, str) or not kind for kind in group['kinds']):
            raise ValueError('invalid word kind')
        if not isinstance(group['context'], str) or '[MASK]' not in group['context']:
            raise ValueError('missing active word context')
        if type(group['label_aligned']) is not bool:
            raise ValueError('word alignment flag must be boolean')
    rebuilt = align_groups(source, [{k:v for k,v in group.items()
               if k not in ('id', 'expected', 'label_aligned', 'positive_indices')} for group in groups])
    if groups != rebuilt:
        raise ValueError('word labels differ from source alignment')


def validate_case(row, source):
    if any(row.get(k) != value for k, value in source.items()):
        raise ValueError('captured row differs from selected source')
    candidates, scores = row['candidates'], row['teacher_scores']
    if (not 1 <= len(candidates) <= 48 or len(set(candidates)) != len(candidates)
            or row['parent_candidates'] != candidates
            or row['positive_indices'] != [i for i, c in enumerate(candidates) if c in source['expected']]):
        raise ValueError('invalid captured sentence pool or labels')
    if row['sentence_eligible']:
        if len(scores) != len(candidates) or any(not math.isfinite(n) for n in scores):
            raise ValueError('invalid teacher score shape or values')
        ranked = [c for c, _ in sorted(zip(candidates, scores), key=lambda item: (-item[1], item[0]))]
        expected_visible = visible_candidates(ranked, source['input'])
    else:
        if scores:
            raise ValueError('fallback must not carry sentence scores')
        expected_visible = candidates
    if row['actual_candidates'] != expected_visible or row['result'] != expected_visible[0]:
        raise ValueError('captured sentence scores do not reproduce actual runtime output')
    validate_groups(source, row['groups'])


def capture(source, converter, word_recorder, sentence_recorder):
    word_recorder.records = []
    sentence_recorder.candidates = sentence_recorder.scores = None
    response = converter.convert(source['input'], source['context'])
    groups = capture_groups(converter.static_pieces, word_recorder.records, source['context'])
    actual = [c['text'] for c in response['candidates']]
    eligible = sentence_recorder.candidates is not None
    if response['sentence_ranking_applied'] != eligible:
        raise ValueError('actual sentence-stage flag differs from recorded call')
    candidates = sentence_recorder.candidates if eligible else actual
    # Labels affect only the prepared training targets, never proposal generation.
    row = {**source, 'groups': align_groups(source, groups),
           'candidates': candidates, 'parent_candidates': candidates,
           'teacher_scores': sentence_recorder.scores if eligible else [],
           'positive_indices': [i for i,c in enumerate(candidates) if c in source['expected']],
           'sentence_eligible': eligible, 'actual_candidates': actual, 'result': response['result'],
           'engine_ms': response['engine_ms']}
    validate_case(row, source)
    return row


def counts(rows):
    return {'rows': len(rows), 'correct': sum(row['result'] in row['expected'] for row in rows),
            'covered': sum(bool(row['positive_indices']) for row in rows),
            'sentence_eligible': sum(row['sentence_eligible'] for row in rows),
            'word_groups': sum(len(row['groups']) for row in rows),
            'unaligned_word_groups': sum(not g['label_aligned'] for row in rows for g in row['groups'])}


def validate_data(directory, manifest_sha256):
    path = directory/'manifest.json'
    if file_sha256(path) != manifest_sha256:
        raise ValueError('manifest differs from explicit pin')
    manifest = json.loads(path.read_text())
    sources, guard, split_guard = load_sources()
    if (manifest.get('profile') != PROFILE or manifest.get('runtime_version') != RUNTIME_VERSION
            or manifest.get('source_hashes') != source_hashes()
            or manifest.get('selection_digest') != selection_digest(sources)
            or manifest.get('challenge_guard') != guard or manifest.get('mixed_split_guard') != split_guard
            or manifest.get('no_gold_injection') is not True or manifest.get('test_data_read') is not False
            or set(manifest.get('outputs', {})) != set(COUNTS)):
        raise ValueError('current preparation provenance differs')
    for name in ('selection.json', 'preparation-source.py'):
        if file_sha256(directory/name) != manifest.get('audit_sha256', {}).get(name):
            raise ValueError('preparation audit copy changed')
    if json.loads((directory/'selection.json').read_text()) != sources:
        raise ValueError('audit selection differs from fixed sources')
    if file_sha256(directory/'preparation-source.py') != manifest['source_hashes']['scripts/prepare_current_context_data.py']:
        raise ValueError('archived preparation code differs')
    data = {}
    for role, count in COUNTS.items():
        path, spec = directory/(role+'.jsonl'), manifest['outputs'][role]
        if file_sha256(path) != spec['sha256'] or path.stat().st_size != spec['bytes']:
            raise ValueError('prepared file changed')
        rows = read_rows(path)
        if len(rows) != count or counts(rows) != spec['counts']:
            raise ValueError('prepared row counts disagree')
        for row, source in zip(rows, sources[role]):
            validate_case(row, source)
        data[role] = rows
    return manifest, data


def prepare(directory):
    if directory.exists():
        raise FileExistsError(directory)
    sources, guard, split_guard = load_sources()
    before = source_hashes()
    converter = CaptureConverter(ranker_kind='hybrid')
    converter.load_models()
    if converter.version != RUNTIME_VERSION:
        raise ValueError('runtime differs from pinned preparation version')
    words, sentences = Recorder(converter.ranker), SentenceRecorder(converter.sentence_ranker)
    converter.ranker, converter.sentence_ranker = words, sentences
    directory.mkdir(parents=True)
    (directory/'preparation-source.py').write_bytes(Path(__file__).read_bytes())
    (directory/'selection.json').write_text(json.dumps(sources, ensure_ascii=False)+'\n')
    started, outputs = time.monotonic(), {}
    for role, selected in sources.items():
        path, rows = directory/(role+'.jsonl'), []
        with path.open('x') as stream:
            for index, source in enumerate(selected):
                row = capture(source, converter, words, sentences)
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush()
                if rss_bytes() > PROFILE['rss_limit_bytes'] or time.monotonic()-started > PROFILE['time_limit_seconds']:
                    raise RuntimeError('preparation resource bound exceeded')
                if (index+1) % 64 == 0:
                    print(json.dumps({'role': role, 'rows': index+1, 'seconds': round(time.monotonic()-started, 1)}), flush=True)
        outputs[role] = {'counts': counts(rows), 'bytes': path.stat().st_size, 'sha256': file_sha256(path)}
        print(json.dumps({'role': role, **counts(rows)}), flush=True)
    if before != source_hashes():
        raise ValueError('preparation sources changed during capture')
    manifest = {'profile': PROFILE, 'runtime_version': RUNTIME_VERSION, 'source_hashes': before,
                'selection_digest': selection_digest(sources), 'challenge_guard': guard,
                'mixed_split_guard': split_guard, 'outputs': outputs,
                'audit_sha256': {name:file_sha256(directory/name) for name in ('selection.json', 'preparation-source.py')},
                'no_gold_injection': True, 'test_data_read': False,
                'seconds': time.monotonic()-started, 'peak_rss_bytes': rss_bytes()}
    path = directory/'manifest.json';path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    digest = file_sha256(path)
    validate_data(directory, digest)
    print(json.dumps({'manifest_sha256': digest, 'outputs': outputs}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--output', type=Path, default=OUTPUT)
    prepare(parser.parse_args().output)
