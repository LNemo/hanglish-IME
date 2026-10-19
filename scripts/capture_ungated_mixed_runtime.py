"""Fresh FIT runtime replay for the closed joint + mixed scoring comparison."""
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
from scripts.score_ungated_joint_fit import PARENT, need, read, sha, verify_pins, write
from scripts.evaluate_ungated_mixed_union_fit import authenticate
from scripts.capture_ungated_joint_runtime import MODEL, normalized

SCORES = ROOT / '.omx/artifacts/ungated-mixed-union-sentence-score-fit-v2'
MIXED = ROOT / '.omx/artifacts/ungated-mixed-boundary-fit-v1'
JOINT = ROOT / '.omx/artifacts/ungated-joint-ranked-fit-v3'
PLAN = ROOT / '.omx/plans/ungated-mixed-runtime-v1.md'
LIMITS = {'whole_seconds': 900., 'row_seconds': 120., 'rss_bytes': 3 * 1024**3,
          'swap_growth_bytes': 512 * 1024**2, 'output_bytes': 128 * 1024**2}


def compare_runtime(parent, scored, mixed, response, trace, *, joint_fixed=None, score_calls=None):
    need(parent['id'] == scored['id'] and parent['request'] == scored['request'],
         'reference identity differs')
    need(response['input'] == parent['request']['input'] and response['context'] == ''
         and not response['external_context_used'], 'runtime input differs')
    baseline = trace['baseline']
    need(not baseline['sentence_ranking_applied']
         and baseline['candidates'] == parent['fallback_public'], 'fresh production parent differs')
    joint = trace['joint_trace']
    generated = joint['generated']
    need(normalized(generated['generated_by_source']) == parent['generated_by_source'],
         'fresh phonetic beams differ')
    for key in ('parents', 'proposals'):
        need([(c['text'], c['segments']) for c in generated[key]]
             == [(c['text'], c['segments']) for c in parent[key]], 'fresh word paths differ')
    if joint_fixed is not None:
        need([(c['text'], c['segments']) for c in joint['joint']['proposals']]
             == [(c['text'], c['segments']) for c in joint_fixed['arms']['roman_repaired']['proposals']],
             'fresh joint paths differ')
    expected_mixed = mixed['proposals'] if mixed else []
    need([(c['text'], c['segments']) for c in trace['mixed_proposals']]
         == [(c['text'], c['segments']) for c in expected_mixed], 'fresh mixed paths differ')
    expected = scored['arms']['mixed_union']
    pool = trace['union_pool']
    need([r['text'] for r in pool] == [r['text'] for r in expected['pool']],
         'fresh union pool differs')
    scores = {r['text']: r['score'] for r in scored['supported_scores']}
    max_delta = 0.
    for actual, fixed in zip(pool, expected['pool']):
        if joint_fixed is not None:
            names = {'joint': 'joint_roman_repaired', 'mixed': 'mixed_boundary'}
            need([names.get(name, name) for name in actual['origins']] == fixed['provenance'],
                 'fresh candidate provenance differs')
        need(actual['supported'] == (fixed['support'] == 'supported'), 'fresh support differs')
        if actual['supported']:
            delta = abs(actual['score'] - scores[actual['text']])
            need(math.isfinite(delta) and delta <= 2e-5, 'fresh sentence score differs')
            max_delta = max(max_delta, delta)
        else:
            need(actual['score'] is None, 'unsupported scored')
    need([r['text'] for r in response['candidates']]
         == [r['text'] for r in expected['public']], 'fresh public order differs')
    need(response['candidates'][-1] == parent['fallback_public'][-1], 'raw reserve changed')
    if score_calls is not None:
        old_texts = {r['text'] for r in joint['pool']}
        groups = [[r['text'] for r in joint['pool'] if r['supported']],
                  [r['text'] for r in pool if r['supported'] and r['text'] not in old_texts]]
        calls = [{'contexts': [''] * len(texts), 'texts': texts} for texts in groups if texts]
        need(score_calls == calls, 'runtime scored other than old pool then new supported candidates')
    if trace.get('applied'):
        info = response.get('ungated_mixed', {})
        old_texts = {r['text'] for r in joint['pool']}
        need(info.get('pool_size') == len(pool)
             and info.get('supported') == sum(r['supported'] for r in pool)
             and info.get('unsupported') == sum(not r['supported'] for r in pool)
             and info.get('mixed_proposals') == len(trace['mixed_proposals'])
             and info.get('new_supported_scored') == sum(r['supported'] and r['text'] not in old_texts for r in pool),
             'runtime mixed metadata differs')
        need('ungated_joint' not in response and 'phonetic_choice' not in response
             and 'phonetic_score_scale' not in response, 'stale selection metadata retained')
        need('문장 모델의 길이 범위를 넘어 단어별 문맥 순위를 사용했습니다.' not in response['warnings'],
             'stale fallback warning retained')
    return {'parent_public_exact': True, 'fresh_phonetic_beams_exact': True,
            'generated_paths_exact': True, 'mixed_paths_exact': True,
            'union_pool_exact': True, 'public_exact': True,
            'max_sentence_score_delta': max_delta}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    from scripts.train_character_edge_full_fit import ResourceGuard
    guard = ResourceGuard(output)

    def observe(stage):
        value = guard.observe(stage)
        need(value['seconds'] <= LIMITS['whole_seconds']
             and value['rss_bytes'] <= LIMITS['rss_bytes']
             and value['swap_growth_bytes'] <= LIMITS['swap_growth_bytes'], 'runtime resource cap')
        need(sum(p.stat().st_size for p in output.iterdir() if p.is_file())
             <= LIMITS['output_bytes'], 'runtime output cap')
        return value

    try:
        score_report = authenticate(SCORES)
        pins = dict(score_report['pins'])
        for path in (Path(__file__), PLAN, ROOT/'scripts/evaluate_ungated_mixed_union_fit.py',
                     ROOT/'scripts/capture_ungated_joint_runtime.py',
                     ROOT/'scripts/train_character_edge_full_fit.py',
                     SCORES/'report.json', SCORES/'capture.complete.json',
                     SCORES/'capture.jsonl', SCORES/'independent-verification.json'):
            pins[str(path.resolve())] = sha(path)
        import torch
        from hanglish.core.score_scale_converter import ScoreScaleConverter
        from hanglish.core.ungated_mixed_converter import UngatedMixedConverter
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        observe('authenticated')
        parent_engine = ScoreScaleConverter(phonetic_checkpoint=MODEL)
        parent_engine.load_models(); parent_engine.load_word_model()
        parent_engine.load_growth_model(); parent_engine.load_phonetic_model()
        need(parent_engine.version == 'experimental-score-scale-1-f6976e1b269d', 'parent version differs')
        wrapper = UngatedMixedConverter(parent_engine, known_english=read(PARENT/'report.json')['known_english'])
        for name, digest in {**parent_engine.model_artifacts, **wrapper.artifacts}.items():
            path = Path(name) if Path(name).is_absolute() else ROOT/name
            need(sha(path) == digest, 'runtime dependency differs')
            pins[str(path.resolve())] = digest
        verify_pins(pins); observe('models-loaded')
        mixed = {r['id']: r for r in map(json.loads, (MIXED/'capture.jsonl').open())}
        joints = {r['id']: r for r in map(json.loads, (JOINT/'capture.jsonl').open())}
        parents = [json.loads(line) for line in (PARENT/'capture.jsonl').open()]
        scores = [json.loads(line) for line in (SCORES/'capture.jsonl').open()]
        need(len(parents) == len(scores) == 13, 'reference row count differs')
        timings, comparisons, ids = [], [], []
        with (output/'capture.jsonl').open('x') as stream:
            for parent, scored in zip(parents, scores):
                started = time.monotonic()
                score_calls = []
                original_score_pairs = parent_engine.sentence_ranker.score_pairs
                def record_scores(contexts, texts):
                    score_calls.append({'contexts': list(contexts), 'texts': list(texts)})
                    return original_score_pairs(contexts, texts)
                parent_engine.sentence_ranker.score_pairs = record_scores
                try:
                    response = wrapper.convert(parent['request']['input'])
                finally:
                    parent_engine.sentence_ranker.score_pairs = original_score_pairs
                elapsed = time.monotonic() - started
                trace = wrapper.last_trace
                comparison = compare_runtime(parent, scored, mixed.get(parent['id']), response, trace,
                    joint_fixed=joints[parent['id']], score_calls=score_calls)
                if trace['applied']:
                    need(response['version'] == wrapper._version(trace['baseline']['version'])
                         and all(response['model_artifacts'].get(name) == digest
                                 for name, digest in wrapper.artifacts.items()), 'runtime version/artifacts differ')
                need(elapsed <= LIMITS['row_seconds'], 'runtime row cap')
                stream.write(json.dumps({'id': parent['id'], 'request': parent['request'],
                    'response': response, 'trace': trace, 'elapsed_seconds': elapsed,
                    'comparison': comparison, 'sentence_score_calls': score_calls}, ensure_ascii=False, allow_nan=False)+'\n')
                stream.flush(); timings.append(elapsed); comparisons.append(comparison); ids.append(parent['id'])
                observe('row-'+parent['id'])
                print(json.dumps({'completed': len(ids), 'rows': 13}), flush=True)
        verify_pins(pins); final = observe('complete')
        report = {'completed': True, 'rows': 13, 'ids': ids, 'pins': pins, 'limits': LIMITS,
                  'resources': final, 'max_rss_bytes': guard.peak_rss,
                  'max_swap_growth_bytes': guard.peak_swap_growth, 'timings_seconds': timings,
                  'comparisons': comparisons, 'capture_sha256': sha(output/'capture.jsonl'),
                  'resources_sha256': sha(output/'resources.jsonl'), 'labels_opened': False,
                  'fresh_parent_inference': True, 'fresh_phonetic_inference': True,
                  'fresh_mixed_word_inference': True, 'fresh_sentence_inference': True,
                  'production_changed': False, 'dev_opened': False, 'test_opened': False}
        write(output/'report.json', report)
        write(output/'capture.complete.json', {'completed': True, 'rows': 13,
              'report_sha256': sha(output/'report.json'), 'capture_sha256': report['capture_sha256'],
              'labels_opened': False})
    except Exception as error:
        write(output/'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
