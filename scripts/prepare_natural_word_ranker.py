"""Capture actual word-ranker features before aligning source-only positives."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_natural_ranker import OUTPUT as SENTENCE_DATA, validate_prepared, frozen_hashes
from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.core.phonology import romanize_word
from hanglish.core.lexical_data import FILES

SOURCE_MANIFEST = '92a37bbc699af4a4981bbe9a29f60c48d71c9901b5113a230791aa7e9f769503'
OUTPUT = ROOT / 'data/natural_context/word-ranker-pilot-v1'
ROLES = ('natural_train', 'natural_dev', 'retention_train')
SPAN = re.compile(r'[A-Za-z가-힣]+|[^A-Za-z가-힣]+')


class Recorder:
    def __init__(self, delegate):
        self.delegate = delegate
        self.records = []

    def score_pairs(self, contexts, candidates):
        scores = self.delegate.score_pairs(contexts, candidates)
        self.records.extend(zip(contexts, candidates, scores))
        return scores


def target_spans(text, accepted):
    targets = {}
    for expected in accepted:
        if romanize_word(expected) != text:
            raise ValueError('source spelling cannot be aligned by exact forward Romanization')
        offset = 0
        for match in SPAN.finditer(expected):
            roman = romanize_word(match[0])
            if re.fullmatch(r'[A-Za-z가-힣]+', match[0]):
                targets.setdefault((offset, offset + len(roman)), set()).add(match[0])
            offset += len(roman)
        if offset != len(text):
            raise ValueError('forward span length mismatch')
    return targets


def capture_groups(pieces, records, context):
    """Reconstruct group boundaries and assert exact runtime call parity."""
    preliminary = [options[0].text for _, _, options in pieces]
    groups = []
    cursor = 0
    for i, (source, offset, options) in enumerate(pieces):
        if len(options) <= 1:
            continue
        lexical = any(o.score >= 5.5 and o.kind != 'raw' for o in options)
        generated = any(o.kind == 'neural' for o in options)
        if not lexical and not generated:
            continue
        chosen = [o for o in options if not (o.kind == 'raw' and o.score < 6)]
        surrounding = context + ' ' + ''.join(preliminary[:i]) + ' [MASK] ' + ''.join(preliminary[i + 1:])
        observed = records[cursor:cursor + len(chosen)]
        if [(c, t) for c, t, _ in observed] != [(surrounding, o.text) for o in chosen]:
            raise ValueError('captured word-ranker inputs differ from runtime grouping')
        groups.append({'source': source, 'start': offset, 'end': offset + len(source),
            'context': surrounding, 'candidates': [o.text for o in chosen],
            'base_scores': [o.score for o in chosen], 'kinds': [o.kind for o in chosen],
            'teacher_scores': [float(score) for _, _, score in observed]})
        cursor += len(chosen)
    if cursor != len(records):
        raise ValueError('unconsumed runtime score calls')
    return groups


def add_targets(source, groups):
    targets = target_spans(source['input'], source['expected'])
    output = []
    for group in groups:
        accepted = targets.get((group['start'], group['end']))
        if not accepted:
            raise ValueError('ranked source span has no exact target alignment')
        output.append({**group, 'id': f"{source['id']}:{group['start']}:{group['end']}",
                       'expected': sorted(accepted),
                       'positive_indices': [i for i, c in enumerate(group['candidates']) if c in accepted]})
    return output


def source_hashes():
    hashes = frozen_hashes()
    paths = [Path(__file__).resolve(), SENTENCE_DATA / 'manifest.json']
    paths.extend(SENTENCE_DATA / (role + '.jsonl') for role in ROLES)
    dictionary = ROOT / 'data/external/mecab-ko-dic'
    paths.extend(dictionary / (name + '.csv') for name in FILES)
    paths.append(dictionary / 'SOURCE.json')
    paths.extend((ROOT / 'checkpoints/sentence_ranker').iterdir())
    for path in paths:
        if path.is_file():
            hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    return hashes


def validate_case(case, source):
    if any(case.get(k) != source[k] for k in ('id', 'origin', 'split', 'group_id', 'input', 'context', 'expected')):
        raise ValueError('word case differs from pinned source/split')
    if not isinstance(case.get('groups'), list):
        raise ValueError('invalid group collection')
    spans = target_spans(source['input'], source['expected'])
    seen = set()
    for group in case['groups']:
        start, end = group['start'], group['end']
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source['input']):
            raise ValueError('invalid group span')
        if (start, end) in seen or source['input'][start:end] != group['source']:
            raise ValueError('duplicate or incorrect group span')
        seen.add((start, end))
        if group['id'] != f"{source['id']}:{start}:{end}" or group['expected'] != sorted(spans.get((start, end), set())):
            raise ValueError('group targets differ from source')
        candidates = group['candidates']
        if (not isinstance(candidates, list) or not candidates
                or any(not isinstance(c, str) or not c for c in candidates)
                or len(set(candidates)) != len(candidates)):
            raise ValueError('invalid candidates')
        for key in ('base_scores', 'teacher_scores', 'kinds'):
            if len(group[key]) != len(candidates):
                raise ValueError('group feature count mismatch')
        if any(type(n) not in (int, float) or not math.isfinite(n)
               for key in ('base_scores', 'teacher_scores') for n in group[key]):
            raise ValueError('nonfinite group score')
        positives = group['positive_indices']
        if not isinstance(positives, list) or any(type(i) is not int for i in positives) or positives != [i for i,c in enumerate(candidates) if c in group['expected']]:
            raise ValueError('incorrect positives')
        if not isinstance(group['context'], str) or '[MASK]' not in group['context']:
            raise ValueError('missing active candidate context')


def validate_word_data(directory, manifest_hash):
    path = directory / 'manifest.json'
    if file_sha256(path) != manifest_hash:
        raise ValueError('word manifest differs from explicitly pinned hash')
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1 or manifest.get('source_manifest_sha256') != SOURCE_MANIFEST:
        raise ValueError('unapproved word-data source')
    if manifest.get('no_gold_injection') is not True or manifest.get('test_data_read') is not False:
        raise ValueError('invalid word-data policy')
    _, sources = validate_prepared(SENTENCE_DATA, SOURCE_MANIFEST)
    if manifest['source_hashes'] != source_hashes() or set(manifest['outputs']) != set(ROLES):
        raise ValueError('word-data generation sources changed')
    result = {}
    for role in ROLES:
        path = directory / (role + '.jsonl')
        spec = manifest['outputs'][role]
        if file_sha256(path) != spec['sha256'] or path.stat().st_size != spec['bytes']:
            raise ValueError('word-data file integrity mismatch')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(rows) != len(sources[role]) or len(rows) != spec['rows']:
            raise ValueError('word-data row count mismatch')
        for case, source in zip(rows, sources[role]):
            validate_case(case, source)
        result[role] = rows
    return manifest, result


def prepare(directory):
    if directory.exists():
        raise FileExistsError(directory)
    _, sources = validate_prepared(SENTENCE_DATA, SOURCE_MANIFEST)
    before = source_hashes()
    converter = NeuralConverter(ranker_kind='semantic')
    converter.load_models()
    recorder = Recorder(converter.ranker)
    converter.ranker = recorder
    directory.mkdir(parents=True)
    started = time.monotonic(); done = 0; outputs = {}
    for role in ROLES:
        counts = {'rows': len(sources[role]), 'scored_groups': 0, 'ambiguous_groups': 0, 'covered_ambiguous_groups': 0}
        path = directory / (role + '.jsonl')
        with path.open('x') as stream:
            for source in sources[role]:
                recorder.records = []
                _, pieces, _, _, _ = converter._prepare_candidates(source['input'], source['context'], learned=True, raw=False)
                # Build/capture features first; only then consult target spelling.
                groups = capture_groups(pieces, recorder.records, source['context'])
                groups = add_targets(source, groups)
                case = {k: source[k] for k in ('id', 'origin', 'split', 'group_id', 'input', 'context', 'expected')}
                case['groups'] = groups
                validate_case(case, source)
                stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + '\n')
                counts['scored_groups'] += len(groups)
                counts['ambiguous_groups'] += sum(len(g['candidates']) > 1 for g in groups)
                counts['covered_ambiguous_groups'] += sum(len(g['candidates']) > 1 and bool(g['positive_indices']) for g in groups)
                done += 1
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if sys.platform != 'darwin': rss *= 1024
                if rss > 3 * 1024**3 or time.monotonic() - started > 1800:
                    raise RuntimeError('word preparation resource limit exceeded')
                if done % 64 == 0:
                    print(json.dumps({'event': 'prepare_words', 'cases': done, 'elapsed_seconds': round(time.monotonic()-started, 2)}), flush=True)
        outputs[role] = {**counts, 'bytes': path.stat().st_size, 'sha256': file_sha256(path)}
    if before != source_hashes(): raise ValueError('sources changed during word capture')
    manifest = {'schema_version': 1, 'source_manifest_sha256': SOURCE_MANIFEST,
        'source_hashes': before, 'runtime_version': converter.version, 'outputs': outputs,
        'no_gold_injection': True, 'test_data_read': False,
        'feature_context': 'Exact live word-ranker call inputs, including preliminary neighbors, before target alignment',
        'elapsed_seconds': time.monotonic()-started, 'peak_rss_bytes': rss}
    path = directory / 'manifest.json'; path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    digest = file_sha256(path); validate_word_data(directory, digest)
    print(json.dumps({'manifest_sha256': digest, 'outputs': outputs, 'peak_rss_bytes': rss}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=OUTPUT)
    prepare(parser.parse_args().output)
