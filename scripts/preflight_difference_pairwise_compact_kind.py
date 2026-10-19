"""Qualify lossless kind serialization without loading model weights.

This isolated preflight runs before either Stage 1 constructor changes.
Its receipt is evidence for the independent verifier, not training data.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_difference_pairwise_lora_data as producer
from scripts import verify_difference_pairwise_lora as independent

INTENT = ROOT / '.omx/artifacts/difference-pairwise-lora-compact-kind-intent-v1.json'
ARCHIVE = ROOT / '.omx/artifacts/difference-pairwise-lora-data-v1-precompact-archive/manifest.json'
CODEBOOK = {'L': 'lexicon', 'X': 'mixed', 'M': 'morphology', 'N': 'neural',
            'D': 'numeric', 'R': 'raw', 'S': 'syllable'}
SYSTEM_TEXT = ('로마자 원문과 문맥을 보고 CURRENT를 유지하면 1, ALTERNATIVE로 바꾸면 2를 답하세요. '
               'K.c는 구간별 kind 코드열(L=lexicon,X=mixed,M=morphology,N=neural,D=numeric,R=raw,S=syllable), '
               'K.a는 CURRENT와 다른 ALTERNATIVE의 0기준 index=code 목록입니다. 숫자 하나만 답하세요.')
VERSION = 'shared-alpha-kind-edits-v1'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    return independent.file_digest(path)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def encode_payload(semantic):
    need(isinstance(semantic, dict) and 'K' not in semantic, 'reserved K collision')
    compact = deepcopy(semantic)
    left = compact['CURRENT'].pop('kinds')
    right = compact['ALTERNATIVE'].pop('kinds')
    need(isinstance(left, list) and isinstance(right, list) and left
         and len(left) == len(right), 'kind sequence lengths differ or empty')
    reverse = {value: key for key, value in CODEBOOK.items()}
    need(all(isinstance(kind, str) and kind in reverse for kind in left + right),
         'unknown kind')
    compact['K'] = {'c': ''.join(reverse[kind] for kind in left),
                    'a': ','.join(f'{i}={reverse[b]}' for i, (a, b)
                                  in enumerate(zip(left, right)) if a != b)}
    return compact


def decode_payload(compact):
    semantic = deepcopy(compact)
    need('kinds' not in semantic['CURRENT'] and 'kinds' not in semantic['ALTERNATIVE'],
         'compact payload still contains full kinds')
    encoded = semantic.pop('K')
    need(isinstance(encoded, dict) and set(encoded) == {'c', 'a'}, 'malformed K')
    codes, edits = encoded['c'], encoded['a']
    need(isinstance(codes, str) and codes and all(code in CODEBOOK for code in codes),
         'invalid current codes')
    need(isinstance(edits, str), 'edits must be a string')
    current = [CODEBOOK[code] for code in codes]
    alternative = current.copy()
    previous = -1
    for edit in edits.split(',') if edits else []:
        need(re.fullmatch(r'(0|[1-9][0-9]*)=[LXMNDRS]', edit) is not None,
             'malformed edit')
        raw_index, code = edit.split('=')
        index = int(raw_index)
        need(previous < index < len(current), 'edit index out of order or range')
        need(current[index] != CODEBOOK[code], 'redundant edit')
        alternative[index] = CODEBOOK[code]
        previous = index
    semantic['CURRENT']['kinds'] = current
    semantic['ALTERNATIVE']['kinds'] = alternative
    return semantic


def capture_sources(intent):
    paths = set(intent['files']) | {str(INTENT.relative_to(ROOT)),
                                   str(ARCHIVE.relative_to(ROOT)),
                                   str(Path(__file__).relative_to(ROOT)),
                                   'scripts/prepare_difference_pairwise_lora_data.py',
                                   'scripts/verify_difference_pairwise_lora.py'}
    for directory in (producer.NATURAL_FIT, producer.CHALLENGE_FIT):
        paths.update(str((directory / name).relative_to(ROOT))
                     for name in ('manifest.json', 'cases.jsonl', 'selection.jsonl'))
    paths.add(str(independent.TRAIN640.relative_to(ROOT)))
    return {name: file_sha(ROOT / name) for name in sorted(paths)}


def preflight():
    intent = json.loads(INTENT.read_bytes())
    need(intent['schema'] == 'difference-pairwise-lora-compact-kind-intent-v1'
         and intent['representation_version'] == VERSION
         and intent['codebook'] == CODEBOOK and intent['system_text'] == SYSTEM_TEXT,
         'intent codec differs')
    for name, expected in intent['files'].items():
        need(file_sha(ROOT / name) == expected, f'intent pin differs: {name}')
    archive = json.loads(ARCHIVE.read_bytes())
    for entry in archive['files']:
        need(file_sha(ROOT / entry['archive']) == entry['sha256'], 'archive pin differs')
        need(file_sha(ROOT / entry['source']) == entry['sha256'], 'pre-edit source differs')
    versions = {name: importlib.metadata.version(name) for name in intent['versions']}
    need(versions == intent['versions'], 'tokenizer package versions differ')
    before = capture_sources(intent)
    roles, _ = independent._load_fit_rows()
    all_pairs, repairs, _ = independent._allowed_pairs(roles)
    units = independent._expected_units(all_pairs, repairs)
    scopes, mechanism, _ = independent._expected_scopes(all_pairs, units, roles)
    counts = {'pairs': len(scopes), 'units': len(units),
              'references': sum(len(unit['references']) for unit in units),
              'updates': math.ceil(len(units) / 4), 'final_update_units': len(units) % 4 or 4,
              'mechanism_rows': len(mechanism),
              'E_mechanism': sum('mechanism' in value for value in scopes.values()),
              'E_train640': sum('train640' in value for value in scopes.values())}
    need(counts == {'pairs': 1928, 'units': 59, 'references': 232, 'updates': 15,
                    'final_update_units': 3, 'mechanism_rows': 175,
                    'E_mechanism': 481, 'E_train640': 1556}, 'required set differs')
    index = {(role, row['id']): row for role, rows in roles.items() for row in rows}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(producer.MODEL_DIR), local_files_only=True)
    observations = []
    for identity in sorted(scopes):
        pair = all_pairs[identity]
        row = index[pair['role'], pair['row_id']]
        original, _ = producer.build_pair(pair['role'], row, pair['alternative_source_index'])
        need(original['pair_id'] == identity, 'pair identity differs')
        semantic = producer.prompt_payload(original)
        compact = encode_payload(semantic)
        decoded = decode_payload(compact)
        need(canonical(decoded) == canonical(semantic), 'semantic round trip differs')
        messages = [{'role': 'system', 'content': SYSTEM_TEXT},
                    {'role': 'user', 'content': canonical(compact).decode()}]
        kwargs = dict(add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        need(tokenizer.encode(rendered, add_special_tokens=False) == tokens,
             'rendered prefix differs')
        digit_ids = {}
        for digit in ('1', '2'):
            completed = tokenizer.encode(rendered + digit, add_special_tokens=False)
            need(len(completed) == len(tokens) + 1 and completed[:-1] == tokens,
                 'digit boundary differs')
            digit_ids[digit] = completed[-1]
        need(digit_ids == {'1': 16, '2': 17}, 'digit continuation IDs differ')
        observations.append({'pair_id': identity, 'semantic_sha256': digest(semantic),
                             'compact_sha256': digest(compact), 'decoded_sha256': digest(decoded),
                             'tokens_sha256': digest(tokens), 'token_count': len(tokens),
                             'digit_continuation_ids': digit_ids})
    lengths = sorted(item['token_count'] for item in observations)
    after = capture_sources(intent)
    need(before == after, 'source pins changed during preflight')
    overflow = sum(length > 1024 for length in lengths)
    return {'schema': 'compact-kind-preflight-v1', 'representation_version': VERSION,
            'intent': {'path': str(INTENT.relative_to(ROOT)), 'sha256': file_sha(INTENT)},
            'archive_manifest': {'path': str(ARCHIVE.relative_to(ROOT)), 'sha256': file_sha(ARCHIVE)},
            'preflight_source': {'path': str(Path(__file__).relative_to(ROOT)),
                                 'sha256': file_sha(Path(__file__))},
            'source_hashes_before': before, 'source_hashes_after': after,
            'codebook': CODEBOOK, 'system_text': SYSTEM_TEXT, 'versions': versions,
            'counts': counts, 'ordered_pair_ids_sha256': digest(sorted(scopes)),
            'units_sha256': digest(units), 'round_trips': len(observations),
            'statistics': {'min': lengths[0], 'median': statistics.median(lengths),
                           'p95': lengths[math.ceil(.95 * len(lengths)) - 1],
                           'max': lengths[-1], 'count': len(lengths), 'ceiling': 1024,
                           'overflow': overflow},
            'ordered_token_digest': digest([item['tokens_sha256'] for item in observations]),
            'observations': observations, 'model_weights_loaded': False,
            'data_qualified': False, 'representation_passed': overflow == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        need(not args.output.exists(), 'output already exists')
        report = preflight()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({'passed': False, 'error': str(error)}))
        return 1
    print(json.dumps({'output': str(args.output), 'counts': report['counts'],
                      'statistics': report['statistics']}))
    return 0 if report['representation_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
