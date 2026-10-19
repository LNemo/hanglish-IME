"""Generate phonetic alternatives on all frozen length-gated FIT rows."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from scripts.audit_fit_generation_gate import sha, classify
from scripts.train_character_edge_full_fit import ResourceGuard

AUDIT = ROOT / '.omx/artifacts/fit-generation-gate-audit-v1.json'
MODEL = ROOT / '.omx/artifacts/observed-phonetic-augmented-training-v1/phonetic_transducer.pt'
MODEL_SHA = 'f06b0a7cd6c5ccbdfaa376cc29f59e90b3fcf0ae3a157fd2c4f7efaa73b4180d'
CHECKPOINT = ROOT / 'checkpoints/sentence_ranker'
LIMITS = {'whole_seconds': 900., 'row_seconds': 120.,
          'rss_bytes': 2 * 1024**3, 'swap_growth_bytes': 256 * 1024**2,
          'output_bytes': 64 * 1024**2}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def support(candidate, tokenizer, maximum):
    text = ' '.join(candidate['text'].strip().split())
    tokens = len(tokenizer(text, add_special_tokens=False, truncation=False)['input_ids'])
    tokens += tokenizer.num_special_tokens_to_add(pair=True)
    return {**candidate, 'sentence_tokens': tokens,
            'sentence_support': 'supported' if tokens <= maximum else 'unsupported'}


def nonraw_parents(public):
    return [{'text': c['text'], 'segments': c['segments']} for c in public
            if c['segments'] and not all(s['kind'] == 'raw' for s in c['segments'])]


def load_rows():
    audit = json.loads(AUDIT.read_text())
    need(audit['counts']['rows'] == 1642 and len(audit['unranked_rows']) == 13
         and audit['labels_opened'] is False, 'source audit differs')
    pins = dict(audit['pins'])
    pins[str(AUDIT)] = sha(AUDIT)
    wanted = {row['id']: row for row in audit['unranked_rows']}
    need(len(wanted) == 13, 'duplicate audit IDs')
    paths = [ROOT / f'.omx/artifacts/current-phonetic-train-shard-{s}-v1/capture.jsonl'
             for s in range(6)]
    for path, expected in pins.items():
        need(sha(path) == expected, 'audit input changed: ' + path)
    rows, seen = [], set()
    for shard, path in enumerate(paths):
        with path.open() as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                need(row['id'] not in seen, 'duplicate FIT source')
                seen.add(row['id'])
                flags = classify(row)
                need((not flags['ranked']) == (row['id'] in wanted), 'selection differs')
                if row['id'] not in wanted:
                    continue
                need(wanted[row['id']]['shard'] == shard
                     and wanted[row['id']]['row_index'] == index
                     and flags['length_warning'] and flags['empty_recorded_pool']
                     and flags['generation_requests'] == 0
                     and row['request']['context'] == '', 'source-gated trace differs')
                rows.append({'id': row['id'], 'request': row['request'],
                             'source_shard': shard, 'source_row_index': index,
                             'fallback_public': row['response']['candidates']})
    need(len(seen) == 1642 and len(rows) == 13, 'FIT selection coverage differs')
    return rows, pins


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    guard = ResourceGuard(output)

    def observe(stage):
        observed = guard.observe(stage)
        need(observed['seconds'] <= LIMITS['whole_seconds']
             and observed['rss_bytes'] <= LIMITS['rss_bytes']
             and observed['swap_growth_bytes'] <= LIMITS['swap_growth_bytes'],
             'ungated capture resource limit exceeded')
        need(sum(p.stat().st_size for p in output.iterdir() if p.is_file())
             <= LIMITS['output_bytes'], 'output size exceeded')
        return observed

    try:
        rows, pins = load_rows()
        need(sha(MODEL) == MODEL_SHA, 'phonetic checkpoint changed')
        metadata = json.loads((CHECKPOINT / 'metadata.json').read_text())
        dependencies = [Path(__file__), ROOT / '.omx/plans/ungated-phonetic-fit-v1.md',
                        ROOT / 'hanglish/core/ungated_phonetic_proposals.py',
                        ROOT / 'hanglish/core/phonetic_word_union_converter.py',
                        ROOT / 'hanglish/core/phonology.py',
                        ROOT / 'hanglish/models/phonetic_transducer.py',
                        ROOT / 'hanglish/models/batched_phonetic_decode.py',
                        ROOT / 'scripts/train_character_edge_full_fit.py',
                        ROOT / 'data/common_english.txt', ROOT / 'data/it_terms.txt',
                        MODEL, MODEL.with_suffix('.json')]
        for path in dependencies:
            pins[str(path.resolve())] = sha(path)
        for name, expected in metadata['artifact_hashes'].items():
            if name != 'model.safetensors':
                need(sha(CHECKPOINT / name) == expected, 'sentence tokenizer changed')
                pins[str(CHECKPOINT / name)] = expected
        observe('authenticated')

        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        from transformers import AutoTokenizer
        from hanglish.models.phonetic_transducer import PhoneticTransducer
        from hanglish.models.batched_phonetic_decode import generate_batched
        from hanglish.core.ungated_phonetic_proposals import generate_proposals

        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
        model = PhoneticTransducer.load(MODEL, device='cpu')
        english = set((ROOT / 'data/common_english.txt').read_text().lower().split())
        english.update((ROOT / 'data/it_terms.txt').read_text().lower().split())
        english.update({'test', 'api', 'hello', 'world', 'openai'})
        observe('model-loaded')
        totals = Counter(rows=0, parents=0, proposals=0, supported_proposals=0,
                         unsupported_proposals=0, generation_requests=0)

        with (output / 'capture.jsonl').open('x') as stream:
            for row in rows:
                started = time.monotonic()

                def generate(roman, beam_size=8):
                    value = generate_batched(model, roman, beam_size)
                    observe('decode-' + row['id'])
                    need(time.monotonic() - started <= LIMITS['row_seconds'], 'row budget exceeded')
                    return value

                generated = generate_proposals(row['request']['input'],
                    nonraw_parents(row['fallback_public']), generate, known_english=english)
                for key in ('parents', 'proposals'):
                    generated[key] = [support(c, tokenizer, metadata['max_length'])
                                      for c in generated[key]]
                result = {**row, **generated}
                encoded = json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n'
                need(sum(p.stat().st_size for p in output.iterdir() if p.is_file())
                     + len(encoded.encode()) <= LIMITS['output_bytes'], 'row output exceeds budget')
                stream.write(encoded)
                stream.flush()
                totals['rows'] += 1
                totals['parents'] += len(generated['parents'])
                totals['proposals'] += len(generated['proposals'])
                totals['generation_requests'] += generated['generation_requests']
                for candidate in generated['proposals']:
                    totals[candidate['sentence_support'] + '_proposals'] += 1
                observe('row-' + row['id'])
                need(time.monotonic() - started <= LIMITS['row_seconds'], 'row budget exceeded')
                print(json.dumps(dict(totals)), flush=True)
        for path, expected in pins.items():
            need(sha(path) == expected, 'input changed during capture: ' + path)
        final = observe('complete')
        write(output / 'report.json', {
            'schema': 'ungated-phonetic-fit-v1', 'completed': True, 'rows': 13,
            'counts': dict(totals), 'pins': pins, 'limits': LIMITS,
            'capture_sha256': sha(output / 'capture.jsonl'),
            'resources_sha256': sha(output / 'resources.jsonl'), 'resources': final,
            'max_rss_bytes': guard.peak_rss, 'max_swap_growth_bytes': guard.peak_swap_growth,
            'known_english': sorted(english), 'max_sentence_tokens': metadata['max_length'],
            'beam_size': 8, 'max_sources': 96, 'max_extras': 96,
            'labels_opened': False, 'sentence_model_inference': False,
            'production_changed': False,
            'scope': 'All13 source-selected unranked FIT rows; public fallback parents only.',
        })
    except Exception as error:
        write(output / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
