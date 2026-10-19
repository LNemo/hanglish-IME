"""Diagnose generation gates from frozen FIT traces without labels or inference."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
LENGTH_WARNING = "문장 모델의 길이 범위를 넘어 단어별 문맥 순위를 사용했습니다."


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def classify(record):
    response = record['response']
    trace = response['phonetic_word_union']
    ranked = response['sentence_ranking_applied']
    if type(ranked) is not bool:
        raise ValueError('ranking flag must be boolean')
    return {
        'ranked': ranked,
        'length_warning': LENGTH_WARNING in response['warnings'],
        'empty_recorded_pool': not record['base_paths'],
        'generation_requests': trace['generation_requests'],
        'generated_extras': len(trace['generated_extras']),
    }


def run(output):
    from transformers import AutoTokenizer

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    pin_path = ROOT / '.omx/artifacts/current-fit-recombination-scoring-shard-0-v1/pins.json'
    pins = json.loads(pin_path.read_text())
    sources = [ROOT / f'.omx/artifacts/current-phonetic-train-shard-{s}-v1/capture.jsonl'
               for s in range(6)]
    bound = {str(p): pins[str(p)] for p in sources}
    checkpoint = ROOT / 'checkpoints/sentence_ranker'
    metadata_path = checkpoint / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    bound[str(metadata_path)] = sha(metadata_path)
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt'):
        bound[str(checkpoint / name)] = metadata['artifact_hashes'][name]
    bound[str(pin_path)] = sha(pin_path)
    bound[str(Path(__file__).resolve())] = sha(__file__)
    for path, expected in bound.items():
        if sha(path) != expected:
            raise ValueError(f'input changed: {path}')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    reserved = tokenizer.num_special_tokens_to_add(pair=True)
    counts, failures, seen = Counter(), [], set()
    for shard, source in enumerate(sources):
        with source.open() as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                if row['id'] in seen or not row['id'].startswith('s3-train-'):
                    raise ValueError('FIT identity differs')
                seen.add(row['id'])
                result = classify(row)
                counts['rows'] += 1
                group = 'ranked' if result['ranked'] else 'unranked'
                counts[group] += 1
                for key in ('length_warning', 'empty_recorded_pool'):
                    counts[f'{group}_{key}'] += int(result[key])
                counts[f'{group}_zero_generation_requests'] += int(result['generation_requests'] == 0)
                if result['ranked']:
                    continue
                lengths = []
                for candidate in row['response']['candidates']:
                    # The reserved original-input slot is not a sentence-model proposal.
                    if all(s['kind'] == 'raw' for s in candidate['segments']):
                        continue
                    text = ' '.join(candidate['text'].strip().split())
                    lengths.append(len(tokenizer(text, add_special_tokens=False,
                                                 truncation=False)['input_ids']) + reserved)
                failures.append({
                    'id': row['id'], 'shard': shard, 'row_index': index,
                    'source_chars': len(row['request']['input']), **result,
                    'public_nonraw_token_lengths_including_specials': lengths,
                    'public_nonraw_over_budget': sum(n > metadata['max_length'] for n in lengths),
                })
    if len(seen) != 1642:
        raise ValueError('FIT coverage differs')
    for path, expected in bound.items():
        if sha(path) != expected:
            raise ValueError(f'input changed during audit: {path}')
    report = {
        'schema': 'fit-generation-gate-audit-v1', 'counts': dict(counts),
        'unranked_rows': failures, 'max_length': metadata['max_length'],
        'reserved_pair_tokens': reserved, 'pins': bound,
        'elapsed_seconds': time.monotonic() - started,
        'labels_opened': False, 'neural_inference_run': False,
        'limitations': [
            'Saved traces describe the frozen FIT capture, not a new runtime execution.',
            'Only public fallback token lengths are available; the rejected complete proposal set was not saved.',
            'An empty recorded ranking pool does not mean no fallback conversion candidates existed.',
            'No claim about gold recovery after decoupling generation is made.',
        ],
    }
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return {k: report[k] for k in ('counts', 'elapsed_seconds')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False))
