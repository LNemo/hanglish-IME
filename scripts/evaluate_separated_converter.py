"""DEV-only paired evaluation of candidate generation/ranking separation."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_sentence_only import sha, summarize

REFERENCE = ROOT / '.omx/artifacts/ctc-union-runtime-dev-v1'
REFERENCE_REPORT_SHA = '05c15eb14ebfd5cad2381d38898fd10f4ac07578789509bf4e14ea528ab6e9f0'
REFERENCE_CAPTURE_SHA = 'ff8b9e2a8708654637d2129a348cd6cb25745bd47871c4939121664a2db4cada'
PLAN = ROOT / '.omx/plans/separated-converter-v1.md'
LIMITS = {'whole_seconds': 1800, 'update_seconds': 120,
          'rss_bytes': 3 * 1024**3, 'swap_growth_bytes': 512 * 1024**2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def source_rows(references, expected_rows=316):
    require(len(references) == expected_rows, 'reference denominator differs')
    result = []
    for row in references:
        request = row.get('request', {})
        require(request.get('context') == '', 'reference context must be empty')
        text = request.get('input')
        require(isinstance(text, str) and len(text.split()) >= 2
                and not re.search('[가-힣ᄀ-ᇿ㄰-㆏]', text), 'reference source differs')
        require(isinstance(row.get('id'), str), 'reference id differs')
        result.append({'id': row['id'], 'input': text})
    require(len({row['id'] for row in result}) == len(result), 'duplicate reference id')
    return result


def compare_public(before, after):
    require(before['result'] == after['result']
            and before['candidates'] == after['candidates'], 'legacy public parity failed')


def compare_legacy_response(before, after):
    def stable(response):
        value = deepcopy(response)
        value.pop('engine_ms', None)
        for name in ('ctc_word', 'ctc_union'):
            if name in value:
                value[name].pop('generation_ms', None)
        return value
    require(stable(before) == stable(after), 'legacy response parity failed')


def pool_for(row, name):
    pipeline = row['legacy_pipeline'] if name == 'legacy' else row[name]['pipeline']
    return pipeline['pool']


def measure(captures, labels):
    require([row['id'] for row in captures] == [row['id'] for row in labels],
            'capture/label identities differ')
    require(all(row.get('request') == {'input': label['input'], 'context': ''}
                for row, label in zip(captures, labels)),
            'capture/label sources differ')
    summaries = {name: summarize([{'id': row['id'], 'response': row[name]}
                                 for row in captures], labels)
                 for name in ('legacy', 'independent')}
    changes = {key: [] for key in ('corrections', 'regressions', 'top5_losses',
                                   'pool_gains', 'pool_losses')}
    coverage = {'legacy': 0, 'independent': 0}
    for row, label, old, new in zip(captures, labels,
                                   summaries['legacy']['outcomes'],
                                   summaries['independent']['outcomes']):
        hits = {}
        for name in coverage:
            texts = {item['text'] for item in pool_for(row, name)}
            hits[name] = bool(texts.intersection(label['expected']))
            coverage[name] += int(hits[name])
        flags = {'corrections': not old['top1'] and new['top1'],
                 'regressions': old['top1'] and not new['top1'],
                 'top5_losses': old['top5'] and not new['top5'],
                 'pool_gains': not hits['legacy'] and hits['independent'],
                 'pool_losses': hits['legacy'] and not hits['independent']}
        for name, flag in flags.items():
            if flag:
                changes[name].append(row['id'])
    return {**summaries, **changes, 'pool_coverage': coverage,
            'quality_improvement_gate': bool(changes['corrections'])
            and not changes['regressions'] and not changes['top5_losses']}


def load_reference():
    require(sha(REFERENCE/'report.json') == REFERENCE_REPORT_SHA
            and sha(REFERENCE/'capture.jsonl') == REFERENCE_CAPTURE_SHA,
            'frozen reference identity differs')
    report = json.loads((REFERENCE/'report.json').read_text())
    require(report['completed'] and report['proposed']['counts'] == {
        'rows': 316, 'top1': 262, 'top5': 282, 'errors': 0}, 'reference counts differ')
    for name, digest in report['pins'].items():
        require(sha(Path(name)) == digest, f'production source/model changed: {name}')
    references = read_rows(REFERENCE/'capture.jsonl')
    return source_rows(references), references, report


def diagnose(captures):
    result = {'rows_without_parent_sentence_ranking': [], 'expanded_pool_rows': [],
              'new_ctc_word_request_rows': [], 'new_ctc_growth_request_rows': [],
              'pool_path_counts': {}, 'pool_unique_text_counts': {}, 'latency': {}}
    for name in ('legacy', 'independent'):
        pools = [pool_for(row, name) for row in captures]
        result['pool_path_counts'][name] = sum(map(len, pools))
        result['pool_unique_text_counts'][name] = sum(
            len({item['text'] for item in pool}) for pool in pools)
        values = sorted(row[name+'_elapsed_ms'] for row in captures)
        result['latency'][name] = {
            'p50_ms': values[math.ceil(len(values)*.5)-1],
            'p95_ms': values[math.ceil(len(values)*.95)-1], 'max_ms': values[-1]}
    reasons = Counter()
    for row in captures:
        if not row['legacy']['sentence_ranking_applied']:
            result['rows_without_parent_sentence_ranking'].append(row['id'])
        old = {item['text'] for item in pool_for(row, 'legacy')}
        new = {item['text'] for item in pool_for(row, 'independent')}
        if new-old:
            result['expanded_pool_rows'].append(row['id'])
        for response_key, result_key in (
                ('ctc_word', 'new_ctc_word_request_rows'),
                ('ctc_union', 'new_ctc_growth_request_rows')):
            if (row['independent'][response_key]['generation_requests']
                    > row['legacy'][response_key]['generation_requests']):
                result[result_key].append(row['id'])
        reasons.update(item['reason'] for item in pool_for(row, 'independent')
                       if not item['evaluated'])
    result['unscored_reasons'] = dict(reasons)
    result['timing_scope'] = 'serial warm convert calls including adapter capture overhead'
    return result


def run(output):
    import torch
    from hanglish.core.separated_converter import SeparatedConverter
    from scripts.preflight_pko_seq2seq import ResourceWatchdog, system_swap_bytes

    inputs, references, reference_report = load_reference()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tracked = [Path(__file__), ROOT/'hanglish/core/separated_converter.py', PLAN,
               ROOT/'scripts/preflight_pko_seq2seq.py',
               REFERENCE/'report.json', REFERENCE/'capture.jsonl']
    pins = {**reference_report['pins'], **{str(p): sha(p) for p in tracked}}
    (output/'pins.json').write_text(json.dumps(pins, indent=2)+'\n')
    started = time.monotonic()
    baseline_swap = system_swap_bytes()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    captures = []
    try:
        with ResourceWatchdog(started, baseline_swap, output/'resources.jsonl',
                              interval_s=5, limits=LIMITS) as watchdog:
            with watchdog.operation('load_models', update=True):
                converter = SeparatedConverter(generation_mode='legacy')
                converter.load_models()
                converter.load_word_model()
                converter.load_growth_model()
            with (output/'capture.jsonl').open('x') as stream:
                for item, reference in zip(inputs, references):
                    record = {'id': item['id'],
                              'request': {'input': item['input'], 'context': ''}}
                    for name in ('legacy', 'independent'):
                        with watchdog.operation(f"{item['id']}:{name}", update=True):
                            request_started = time.monotonic()
                            response = converter.convert(
                                item['input'], context='', engine='neural',
                                mode='batch', raw=False, generation_mode=name)
                            elapsed_ms = (time.monotonic()-request_started)*1000
                        require(response.get('input') == item['input']
                                and response.get('context') == ''
                                and not response.get('external_context_used', False),
                                'whole-source predictor contract differs')
                        if name == 'legacy':
                            try:
                                compare_public(reference['proposed'], response)
                                compare_legacy_response(reference['proposed'], response)
                            except ValueError:
                                (output/'parity-failure.json').write_text(json.dumps({
                                    'id': item['id'], 'mode': name,
                                    'reference': reference['proposed'],
                                    'observed': response}, ensure_ascii=False, indent=2)+'\n')
                                raise
                        record[name] = deepcopy(response)
                        record[name+'_elapsed_ms'] = elapsed_ms
                        if name == 'legacy':
                            record['legacy_pipeline'] = deepcopy(converter.last_pipeline)
                        require(isinstance(pool_for(record, name), list)
                                and pool_for(record, name),
                                'full pipeline capture missing')
                    # The independent mode deliberately retains public selection.
                    try:
                        compare_public(record['legacy'], record['independent'])
                    except ValueError:
                        (output/'parity-failure.json').write_text(
                            json.dumps(record, ensure_ascii=False, indent=2)+'\n')
                        raise
                    captures.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False)+'\n')
                    stream.flush()
                    if len(captures) % 25 == 0:
                        print(json.dumps({'captured': len(captures), 'rows': len(inputs),
                                          'elapsed_s': time.monotonic()-started}), flush=True)
                os.fsync(stream.fileno())
            capture_sha = sha(output/'capture.jsonl')
            closure = {'rows': len(captures), 'capture_sha256': capture_sha,
                       'source_only': True, 'targets_joined': False}
            (output/'capture.complete.json').write_text(json.dumps(closure, indent=2)+'\n')
            # Inference and complete pool capture have finished before label joining.
            labels = read_rows(ROOT/'data/challenge/sentence-v3/dev.jsonl')
            metrics = measure(captures, labels)
            diagnostics = diagnose(captures)
            for path, digest in pins.items():
                require(sha(Path(path)) == digest, f'pinned file changed: {path}')
            final_resources = watchdog._sample('final')
        report = {
            'schema': 'separated-converter-dev-v1', 'completed': True,
            'split': 'dev', 'input_contract': 'roman_sentence_only',
            'gold_not_passed_to_predictor': True,
            'targets_joined_after_closed_capture': True,
            'legacy_frozen_public_parity': True,
            'legacy_full_response_parity_except_timing': True,
            'legacy_timing_exclusions': ['engine_ms', 'ctc_word.generation_ms',
                                         'ctc_union.generation_ms'],
            'independent_public_parity': True,
            'capture_sha256': capture_sha,
            'closure_sha256': sha(output/'capture.complete.json'),
            'pins': pins, **metrics, 'diagnostics': diagnostics,
            'resources': {'limits': LIMITS, 'baseline_swap_bytes': baseline_swap,
                          'final': final_resources,
                          'trace_sha256': sha(output/'resources.jsonl')},
            'elapsed_s': time.monotonic()-started,
            'production_changed': False, 'test_opened': False,
            'runtime_promotion': False,
        }
    except BaseException as error:
        existing = output/'report.json'
        prior = json.loads(existing.read_text()) if existing.exists() else None
        report = {'schema': 'separated-converter-dev-v1', 'completed': False,
                  'error': f'{type(error).__name__}: {error}',
                  'captured_rows': len(captures), 'watchdog_report': prior,
                  'production_changed': False, 'test_opened': False}
        existing.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        raise
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in (
        'completed', 'pool_coverage', 'pool_gains', 'pool_losses', 'corrections',
        'regressions', 'elapsed_s')}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
