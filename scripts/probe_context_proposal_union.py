"""Train-split-only probe of a bounded parent-preserving sentence proposal union."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.models.semantic_ranker import SemanticRanker
from scripts.prepare_context_training import OUTPUT, validate_context_data, source_hashes, RUNTIME_VERSION
from scripts.train_natural_sentence_ranker import rss_bytes
from scripts.context_proposal_union import propose, choose

MANIFEST = '7c1c03205dbc9c8983c2b3dee37e4a6bcabb04d45a265e9e1699460595275b10'
STUDENT = ROOT / '.omx/artifacts/preserved-context-ranker-v1/checkpoint'
STUDENT_WEIGHT = '8d6c439c115d2ff04a9ef5c5271d20ce65ebcf2810435bea1232e5fa072a83e2'
PROFILE = {'natural_train_rows': 512, 'parent_word_beam': 24, 'static_beam': 24,
           'student_word_beam': 24, 'max_pool': 72, 'parent_runtime_parity_rows': 16,
           'rss_limit_bytes': 3 * 1024**3, 'seconds_limit': 1800,
           'selection': 'All fixed natural_train rows; no target-dependent selection'}


def hashes():
    result = source_hashes()
    paths = [Path(__file__), ROOT / 'scripts/context_proposal_union.py',
             ROOT / 'scripts/train_natural_sentence_ranker.py']
    paths.extend(STUDENT.iterdir())
    paths.extend(OUTPUT / name for name in ('manifest.json', 'natural_train.jsonl',
                                           'natural_dev.jsonl', 'retention_train.jsonl'))
    for path in paths:
        if path.is_file():
            result[str(path.relative_to(ROOT))] = file_sha256(path)
    return result


def visible_candidates(paths, original):
    selected = [path[0] for path in paths[:5]]
    if original not in selected:
        selected = selected[:4] + [original]
    return selected


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    _, data = validate_context_data(OUTPUT, MANIFEST)
    rows = data['natural_train']
    if len(rows) != PROFILE['natural_train_rows'] or file_sha256(STUDENT / 'model.safetensors') != STUDENT_WEIGHT:
        raise ValueError('unexpected source selection or student')
    before = hashes()
    converter = NeuralConverter(ranker_kind='hybrid')
    converter.load_models()
    if converter.version != RUNTIME_VERSION:
        raise ValueError('runtime differs from pinned parent')
    student = SemanticRanker.load(STUDENT)
    if student.metadata['parent_runtime_version'] != RUNTIME_VERSION:
        raise ValueError('student parent differs')
    args.output.mkdir(parents=True)
    started = time.monotonic()
    results = []
    for index, row in enumerate(rows):
        tick = time.perf_counter()
        proposal = propose(converter, student, row['input'], row['context'])
        proposal_ms = 1000 * (time.perf_counter() - tick)
        parent_pool = [path[0] for path in proposal['parent_paths']]
        union_pool = [path[0] for path in proposal['union_paths']]
        if parent_pool != row['candidates']:
            raise ValueError(f'parent source-only pool differs from frozen capture: {row["id"]}')
        if union_pool[:len(parent_pool)] != parent_pool or len(union_pool) > PROFILE['max_pool']:
            raise ValueError('parent candidate preservation or bound violated')
        tick = time.perf_counter()
        parent = choose({**proposal, 'union_paths': proposal['parent_paths']}, converter.sentence_ranker, row['context'])
        parent_score_ms = 1000 * (time.perf_counter() - tick)
        tick = time.perf_counter()
        union = choose(proposal, converter.sentence_ranker, row['context'])
        union_score_ms = 1000 * (time.perf_counter() - tick)
        parent_visible = visible_candidates(parent['paths'], row['input'])
        union_visible = visible_candidates(union['paths'], row['input'])
        parity = None
        if index < PROFILE['parent_runtime_parity_rows']:
            actual = converter.convert(row['input'], row['context'])
            parity = (parent_visible == [c['text'] for c in actual['candidates']]
                      and parent['sentence_ranking_applied'] == actual['sentence_ranking_applied'])
            if not parity:
                raise ValueError('experimental parent replay differs from actual hybrid')
        # Labels are consulted only after both proposal sets and their orders exist.
        expected = row['expected']
        result = {'id': row['id'], 'input': row['input'], 'context': row['context'], 'expected': expected,
                  'parent_candidates': parent_pool, 'union_candidates': union_pool,
                  'parent_top5': parent_visible, 'union_top5': union_visible,
                  'parent_correct': parent_visible[0] in expected,
                  'union_correct': union_visible[0] in expected,
                  'parent_covered': any(p in expected for p in parent_pool),
                  'union_covered': any(p in expected for p in union_pool),
                  'sentence_eligible': proposal['sentence_eligible'],
                  'additions': proposal['additions'], 'filtered_long': proposal['filtered_long'],
                  'parent_runtime_parity': parity, 'proposal_ms': proposal_ms,
                  'parent_score_ms': parent_score_ms, 'union_score_ms': union_score_ms}
        results.append(result)
        with (args.output / 'cases.jsonl').open('a') as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
        elapsed = time.monotonic() - started
        if rss_bytes() > PROFILE['rss_limit_bytes'] or elapsed > PROFILE['seconds_limit']:
            raise RuntimeError('proposal probe resource bound exceeded')
        if (index + 1) % 64 == 0:
            print(json.dumps({'cases': index + 1, 'elapsed_seconds': round(elapsed, 2)}), flush=True)
    validate_context_data(OUTPUT, MANIFEST)
    if before != hashes():
        raise ValueError('probe sources changed')
    metrics = {key: sum(row[key] for row in results) for key in
               ('parent_correct', 'union_correct', 'parent_covered', 'union_covered')}
    corrections = [r['id'] for r in results if r['union_correct'] and not r['parent_correct']]
    regressions = [r['id'] for r in results if r['parent_correct'] and not r['union_correct']]
    new_coverage = [r['id'] for r in results if r['union_covered'] and not r['parent_covered']]
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'profile': PROFILE,
              'parent_version': RUNTIME_VERSION, 'student_weight_sha256': STUDENT_WEIGHT,
              'source_hashes': before, 'preparation_manifest_sha256': MANIFEST,
              'metrics': metrics, 'corrections': corrections, 'regressions': regressions,
              'new_coverage': new_coverage, 'all_parent_candidates_preserved': True,
              'parent_runtime_parity_rows': PROFILE['parent_runtime_parity_rows'],
              'median_pool_size': {kind: statistics.median(len(r[kind+'_candidates']) for r in results)
                                   for kind in ('parent', 'union')},
              'median_ms': {kind: statistics.median(r[kind+'_ms'] for r in results)
                            for kind in ('proposal', 'parent_score', 'union_score')},
              'timing_policy': 'Sequential paired scoring, parent first; descriptive timings with shared caches.',
              'elapsed_seconds': time.monotonic()-started, 'peak_rss_bytes': rss_bytes(),
              'development_inference_run': False, 'test_data_read': False, 'deployed': False,
              'eligible_for_joint_training': bool(new_coverage),
              'cases_sha256': file_sha256(args.output / 'cases.jsonl')}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'probe-source.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'union-source.py').write_bytes((ROOT / 'scripts/context_proposal_union.py').read_bytes())
    print(json.dumps({key: report[key] for key in ('metrics', 'corrections', 'regressions', 'new_coverage',
          'median_pool_size', 'median_ms', 'elapsed_seconds', 'peak_rss_bytes', 'eligible_for_joint_training')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / '.omx/artifacts/context-proposal-union-v1')
    run(parser.parse_args())
