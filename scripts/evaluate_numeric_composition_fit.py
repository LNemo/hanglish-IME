"""Capture all source-only numeric-composition FIT predictions before scoring.

This is an isolated experiment, never a production installation. The first 512
natural FIT rows are fixed by the previously verified candidate-stage audit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUDIT = ROOT / '.omx/artifacts/final-choice-candidate-stages-v1'
AUDIT_REPORT_SHA = '528f4fbc1ebb2bbc9e22e18b85b4af6e0a205a9a5fe1a64bbe266b52ba46b615'
AUDIT_CASES_SHA = '860da6746310eee896008280a8ad6eac72736b35db16a289aa02a9a5f45264c6'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(',', ':')).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def without_timing(response):
    return {key: value for key, value in response.items() if key != 'engine_ms'}


def summarize(rows, captures):
    """Labels enter only after every capture has been persisted and closed."""
    if len(rows) != len(captures) or not rows:
        raise ValueError('all FIT rows must be captured')
    ids = set()
    counters = dict.fromkeys(('rows', 'before_top1', 'top1', 'before_top5', 'top5',
                             'before_pool', 'pool', 'eligible_rows', 'new_path_rows'), 0)
    corrections, regressions, top5_gains, top5_losses = [], [], [], []
    for row, capture in zip(rows, captures):
        if row['id'] != capture['id'] or row['id'] in ids:
            raise ValueError('FIT identities differ or repeat')
        ids.add(row['id'])
        for key in ('input', 'context'):
            if capture[key] != row[key]:
                raise ValueError('source identity changed')
        parent, child, trace = (capture[name] for name in ('parent', 'child', 'trace'))
        if parent['result'] != row['result'] or parent['candidates'] != row['candidates']:
            raise ValueError('actual production response differs from frozen FIT')
        if capture['parent_raw'] != capture['child_raw']:
            raise ValueError('raw response changed')
        parent_pool = capture['parent_pool']
        if trace['base'] != capture['parent_base']:
            raise ValueError('original base pool objects changed')
        if capture['parent_sentence_scored']:
            preserved = trace['base'] + trace['accepted_old_extras']
            if preserved != parent_pool or trace['scoring_pool'][:len(preserved)] != preserved:
                raise ValueError('original full pool objects changed')
        if capture['child_sentence_scored']:
            if capture['child_score_inputs'] != [p['text'] for p in trace['scoring_pool']]:
                raise ValueError('trace differs from actual scorer input')
        elif capture['child_score_inputs'] is not None:
            raise ValueError('unexpected sentence scorer call')
        expected = row['expected']
        if not isinstance(expected, list) or not expected or any(not isinstance(v, str) for v in expected):
            raise ValueError('accepted answers missing')
        before, after = parent['result'] in expected, child['result'] in expected
        before5 = any(c['text'] in expected for c in parent['candidates'])
        after5 = any(c['text'] in expected for c in child['candidates'])
        child_pool = trace['scoring_pool'] if capture['child_sentence_scored'] else child['candidates']
        values = (1, before, after, before5, after5,
                  any(p['text'] in expected for p in parent_pool),
                  any(p['text'] in expected for p in child_pool),
                  trace['eligible_piece_count'] > 0, bool(trace['accepted_new_extras']))
        for key, value in zip(counters, values):
            counters[key] += int(value)
        for condition, collection in ((after and not before, corrections),
                (before and not after, regressions), (after5 and not before5, top5_gains),
                (before5 and not after5, top5_losses)):
            if condition:
                collection.append(row['id'])
    return {'counts': counters, 'correction_ids': corrections, 'regression_ids': regressions,
            'top5_gain_ids': top5_gains, 'top5_loss_ids': top5_losses,
            'gate': {'passed': len(rows) == 512 and not regressions and not top5_losses and bool(top5_gains),
                     'original_pool_preserved': True, 'raw_preserved': True,
                     'zero_first_choice_losses': not regressions,
                     'zero_top5_losses': not top5_losses,
                     'at_least_one_new_top5': bool(top5_gains)}}


def run(output):
    import torch
    from scripts import audit_final_choice_candidate_stages as prior
    from scripts.numeric_composition_runtime import NumericCompositionRuntime
    from scripts.final_choice_capture import _response_flags, validate_response
    from scripts.train_final_choice_selector import load_fit_role

    if output.exists():
        raise FileExistsError(output)
    if sha(AUDIT / 'report.json') != AUDIT_REPORT_SHA or sha(AUDIT / 'cases.jsonl') != AUDIT_CASES_SHA:
        raise ValueError('frozen FIT cohort audit changed')
    loaded = load_fit_role(prior.FIT, prior.FIT_MANIFEST_SHA256, 'natural_fit')
    rows = loaded['rows'][:512]
    audit_ids = [json.loads(line)['id'] for line in (AUDIT / 'cases.jsonl').read_bytes().splitlines()]
    if len(rows) != 512 or [r['id'] for r in rows] != audit_ids:
        raise ValueError('FIT cohort differs')
    hashes = dict(json.loads((AUDIT / 'report.json').read_bytes())['source_hashes'])
    for name in ('scripts/numeric_composition_runtime.py', 'hanglish/core/numeric_composition.py',
                 'scripts/evaluate_numeric_composition_fit.py',
                 '.omx/plans/numeric-composition-successor-handoff-v1.json'):
        hashes[name] = sha(ROOT / name)
    prior.verify_hashes(hashes)
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {'schema': 'numeric-composition-fit-v1', 'completed': False, 'deployed': False,
              'scope': 'first512 natural FIT only', 'source_hashes_before': hashes}
    try:
        torch.set_num_threads(2)
        parent = prior.StageConverter(ranker_kind='hybrid', external_lexicon=True)
        parent.load_models()
        if parent.version != prior.RUNTIME_VERSION or parent.model_artifacts != loaded['manifest']['model_artifact_hashes']:
            raise ValueError('production model identity differs')
        recorder = prior.SentenceRecorder(parent.sentence_ranker)
        parent.sentence_ranker = recorder
        child = NumericCompositionRuntime.__new__(NumericCompositionRuntime)
        child.__dict__ = parent.__dict__.copy()
        sources = [{k: row[k] for k in ('id', 'input', 'context')} for row in rows]
        (output / 'sources.jsonl').write_bytes(b''.join(canonical(row) + b'\n' for row in sources))
        captures = []
        with (output / 'captures.jsonl').open('xb') as stream:
            for source in sources:
                text, context = source['input'], source['context']
                recorder.reset()
                before = parent.convert(text, context)
                parent_scored = before['sentence_ranking_applied']
                learned = prior.beam_paths(parent.audit_pieces)
                static = prior.beam_paths(parent.audit_static_pieces)
                pool_probe = len(learned) == 1 or (len(learned) > 1 and len(text.split()) > 1)
                base = dict((p[0], p) for p in learned)
                if pool_probe:
                    for path in static:
                        base.setdefault(path[0], path)
                parent_base = [{'text': p[0], 'score': p[1], 'segments': deepcopy(p[2])}
                               for p in base.values()]
                if parent_scored:
                    _, _, full, _, _ = prior.reconstruct_pool(parent, parent.audit_pieces,
                        parent.audit_static_pieces, sentence_possible=True)
                    parent_pool = [{'text': p[0], 'score': p[1], 'segments': deepcopy(p[2])} for p in full]
                    if recorder.candidates != [p['text'] for p in parent_pool]:
                        raise ValueError('parent pool reconstruction differs from actual scorer')
                else:
                    parent_pool = [{'text': p[0], 'score': p[1], 'segments': deepcopy(p[2])}
                                   for p in learned]
                recorder.reset()
                after = child.convert(text, context)
                trace = deepcopy(child.numeric_composition_trace)
                child_inputs = deepcopy(recorder.candidates)
                parent_raw = parent.convert(text, context, raw=True)
                child_raw = child.convert(text, context, raw=True)
                for response in (before, after, parent_raw, child_raw):
                    validate_response(response, text, context, prior.RUNTIME_VERSION,
                                      parent.model_artifacts, raw=response['raw'])
                capture = {**source, 'parent': without_timing(before), 'child': without_timing(after),
                    'parent_raw': without_timing(parent_raw), 'child_raw': without_timing(child_raw),
                    'parent_pool': parent_pool, 'parent_base': parent_base, 'trace': trace,
                    'parent_sentence_scored': parent_scored,
                    'child_sentence_scored': after['sentence_ranking_applied'],
                    'child_score_inputs': child_inputs,
                    'parent_ms': before['engine_ms'], 'child_ms': after['engine_ms']}
                stream.write(canonical(capture) + b'\n'); stream.flush()
                captures.append(capture)
                prior.require_resources(started)
                if len(captures) % 32 == 0:
                    print(json.dumps({'stage': 'source_capture', 'rows': len(captures)}), flush=True)
        # No outcome/expected fields were passed to either conversion. All512
        # pools are now frozen on disk before the first accuracy calculation.
        report['captures_sha256'] = sha(output / 'captures.jsonl')
        for row, capture in zip(rows, captures):
            if _response_flags(capture['parent']) != row['response_flags']:
                raise ValueError('production flags differ from frozen FIT')
        result = summarize(rows, captures)
        prior.verify_hashes(hashes)
        prior.require_resources(started)
        timings = sorted(c['child_ms'] for c in captures[1:])
        report.update(result)
        report.update({'completed': True, 'runtime_version': parent.version,
            'source_hashes_after': hashes, 'rows_captured_before_accuracy': 512,
            'first_child_ms': captures[0]['child_ms'],
            'warm_child_p95_ms': timings[math.ceil(.95 * len(timings)) - 1],
            'latency_scope': 'engine calls after model load; not browser or cold model loading',
            'sources_sha256': sha(output / 'sources.jsonl')})
    except Exception as error:
        report['failure'] = f'{type(error).__name__}: {error}'
    report['elapsed_seconds'] = time.monotonic() - started
    report['created_at'] = datetime.now(timezone.utc).isoformat()
    (output / 'report.json').write_bytes(canonical(report) + b'\n')
    print(json.dumps(report.get('gate', {'failure': report.get('failure')})), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.omx/artifacts/numeric-composition-fit-v1')
    result = run(parser.parse_args().output)
    raise SystemExit(0 if result['completed'] else 1)
