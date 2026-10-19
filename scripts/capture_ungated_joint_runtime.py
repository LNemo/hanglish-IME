"""Fresh parent/generation/ranking replay for the closed 13-row FIT experiment."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from scripts.score_ungated_joint_fit import (
    PARENT, authenticate_scoring, need, read, sha, verify_pins, write,
)

JOINT = ROOT / '.omx/artifacts/ungated-joint-ranked-fit-v3'
SCORES = ROOT / '.omx/artifacts/ungated-joint-sentence-score-fit-v1'
MODEL = ROOT / '.omx/artifacts/observed-phonetic-augmented-training-v1/phonetic_transducer.pt'
PLAN = ROOT / '.omx/plans/ungated-joint-runtime-v1.md'
LIMITS = {'whole_seconds': 900., 'row_seconds': 120., 'rss_bytes': 3 * 1024**3,
          'swap_growth_bytes': 512 * 1024**2, 'output_bytes': 128 * 1024**2}


def normalized(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def compare_runtime(parent, scored, response, trace):
    """Check real runtime against the already closed generation/score capture."""
    need(parent['id'] == scored['id'] and parent['request'] == scored['request'],
         'reference source identity differs')
    need(response['input'] == parent['request']['input'] and response['context'] == ''
         and not response['external_context_used'], 'runtime request differs')
    baseline = trace['baseline']
    need(not baseline['sentence_ranking_applied']
         and baseline['candidates'] == parent['fallback_public'],
         'fresh parent public candidates differ')
    generated = trace['generated']
    need(normalized(generated['generated_by_source']) == parent['generated_by_source'],
         'fresh neural beams differ')
    for key in ('parents', 'proposals'):
        actual = [(c['text'], c['segments']) for c in generated[key]]
        fixed = [(c['text'], c['segments']) for c in parent[key]]
        need(actual == fixed, 'fresh ' + key + ' paths differ')
    expected = scored['arms']['roman_repaired']
    need([r['text'] for r in trace['pool']] == [r['text'] for r in expected['pool']],
         'fresh combined pool differs')
    scores = {row['text']: row['score'] for row in scored['supported_scores']}
    max_delta = 0.
    for actual, fixed in zip(trace['pool'], expected['pool']):
        need(actual['supported'] == (fixed['support'] == 'supported'),
             'fresh support status differs')
        if actual['supported']:
            delta = abs(actual['score'] - scores[actual['text']])
            need(math.isfinite(delta) and delta <= 2e-5,
                 'fresh sentence score differs beyond batch tolerance')
            max_delta = max(max_delta, delta)
        else:
            need(actual['score'] is None, 'unsupported candidate acquired score')
    need([r['text'] for r in response['candidates']]
         == [r['text'] for r in expected['public']], 'fresh public ordering differs')
    need(trace['applied'] == (not expected['fallback_used']), 'fallback dispatch differs')
    return {'parent_public_exact': True, 'fresh_neural_beams_exact': True,
            'generated_paths_exact': True, 'pool_exact': True, 'public_exact': True,
            'max_sentence_score_delta': max_delta}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    from scripts.train_character_edge_full_fit import ResourceGuard
    guard = ResourceGuard(output)

    def observe(stage):
        value = guard.observe(stage)
        need(value['seconds'] <= LIMITS['whole_seconds']
             and value['rss_bytes'] <= LIMITS['rss_bytes']
             and value['swap_growth_bytes'] <= LIMITS['swap_growth_bytes'],
             'runtime resources exceeded')
        need(sum(p.stat().st_size for p in output.iterdir() if p.is_file())
             <= LIMITS['output_bytes'], 'runtime output size exceeded')
        return value

    try:
        score_report, closure = authenticate_scoring(JOINT, SCORES)
        pins = dict(score_report['pins'])
        for path in [Path(__file__), PLAN, SCORES / 'report.json',
                     SCORES / 'capture.complete.json', SCORES / 'capture.jsonl']:
            pins[str(path.resolve())] = sha(path)
        import torch
        from hanglish.core.score_scale_converter import ScoreScaleConverter
        from hanglish.core.ungated_joint_converter import UngatedJointConverter
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        observe('authenticated')
        parent_engine = ScoreScaleConverter(phonetic_checkpoint=MODEL)
        parent_engine.load_models()
        parent_engine.load_word_model()
        parent_engine.load_growth_model()
        parent_engine.load_phonetic_model()
        need(parent_engine.version == 'experimental-score-scale-1-f6976e1b269d',
             'current production parent version differs')
        inventory = read(PARENT / 'report.json')['known_english']
        wrapper = UngatedJointConverter(parent_engine, known_english=inventory)
        for name, digest in {**parent_engine.model_artifacts, **wrapper.artifacts}.items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            need(sha(path) == digest, 'runtime dependency differs: ' + str(path))
            pins[str(path.resolve())] = digest
        verify_pins(pins)
        observe('models-loaded')
        timings, ids, comparisons = [], [], []
        with (PARENT / 'capture.jsonl').open() as parents, \
                (SCORES / 'capture.jsonl').open() as scores, \
                (output / 'capture.jsonl').open('x') as target:
            for index in range(13):
                parent_line, score_line = next(parents, None), next(scores, None)
                need(parent_line is not None and score_line is not None, 'missing reference row')
                fixed_parent, fixed_score = json.loads(parent_line), json.loads(score_line)
                started = time.monotonic()
                response = wrapper.convert(fixed_parent['request']['input'])
                elapsed = time.monotonic() - started
                trace = wrapper.last_trace
                comparison = compare_runtime(fixed_parent, fixed_score, response, trace)
                need(elapsed <= LIMITS['row_seconds'], 'runtime row time exceeded')
                record = {'id': fixed_parent['id'], 'request': fixed_parent['request'],
                          'response': response, 'trace': trace,
                          'elapsed_seconds': elapsed, 'comparison': comparison}
                target.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                target.flush()
                ids.append(record['id'])
                timings.append(elapsed)
                comparisons.append(comparison)
                observe('row-' + record['id'])
                print(json.dumps({'completed': index + 1, 'rows': 13,
                                  'elapsed_seconds': elapsed}), flush=True)
            need(next(parents, None) is None and next(scores, None) is None,
                 'extra reference rows')
        verify_pins(pins)
        final = observe('complete')
        report = {'completed': True, 'rows': len(ids), 'ids': ids, 'pins': pins,
                  'limits': LIMITS, 'resources': final, 'max_rss_bytes': guard.peak_rss,
                  'max_swap_growth_bytes': guard.peak_swap_growth,
                  'timings_seconds': timings, 'comparisons': comparisons,
                  'capture_sha256': sha(output / 'capture.jsonl'),
                  'resources_sha256': sha(output / 'resources.jsonl'),
                  'labels_opened': False, 'production_changed': False,
                  'fresh_parent_inference': True, 'fresh_generation_inference': True,
                  'fresh_sentence_inference': True, 'dev_opened': False, 'test_opened': False}
        write(output / 'report.json', report)
        write(output / 'capture.complete.json', {
            'completed': True, 'rows': len(ids), 'report_sha256': sha(output / 'report.json'),
            'capture_sha256': report['capture_sha256'], 'labels_opened': False})
    except Exception as error:
        write(output / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
