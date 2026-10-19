"""Experimental runtime recovery for sentences skipped by the parent ranker."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from hanglish.core.ungated_joint_proposals import propose_joint
from hanglish.core.ungated_phonetic_proposals import generate_proposals


def recover_unranked(source, baseline, generate, ranker, *, known_english=()):
    """Return a response and a complete diagnostic pool without changing inputs.

    The caller owns dispatch, model locking and versioning. Unsupported paths
    remain in the diagnostic pool without a numeric score. The old public
    response is returned intact when there is no supported replacement path.
    """
    if baseline.get('input') != source:
        raise ValueError('baseline input differs from requested source')
    parents = [dict(text=row['text'], segments=deepcopy(row['segments']))
               for row in baseline['candidates']
               if row['text'] != source and row['segments']
               and not all(segment['kind'] == 'raw' for segment in row['segments'])][:4]
    trace = {'applied': False, 'baseline': deepcopy(baseline), 'pool': [],
             'generation_requests': 0, 'reason': 'no_nonraw_parents'}
    if not parents:
        return baseline, trace
    generated = generate_proposals(source, parents, generate,
                                   known_english=known_english)
    joint = propose_joint(source, generated['parents'], generated['proposals'],
                          generated['generated_by_source'],
                          seed_mode='roman_repaired', repair_mixed=True,
                          prioritize_beams=True)
    pool_by_text = {}
    for origin, rows in [('parent', generated['parents']),
                         ('single', generated['proposals']),
                         ('joint', joint['proposals'])]:
        for row in rows:
            candidate = pool_by_text.setdefault(row['text'], {
                'text': row['text'], 'segments': deepcopy(row['segments']),
                'origins': [], 'score': None,
            })
            candidate['origins'].append(origin)
    pool = list(pool_by_text.values())
    for candidate in pool:
        candidate['supported'] = ranker.can_score(candidate['text'])
    supported = [row for row in pool if row['supported']]
    trace.update(pool=pool, generated=generated, joint=joint,
                 generation_requests=generated['generation_requests'],
                 reason='no_supported_candidates')
    if not supported:
        return baseline, trace
    scores = ranker.score_pairs([''] * len(supported),
                                [row['text'] for row in supported])
    if len(scores) != len(supported) or any(
            isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) for score in scores):
        raise ValueError('sentence ranker returned invalid scores')
    for candidate, score in zip(supported, scores):
        candidate['score'] = float(score)
    ranked = sorted(supported, key=lambda row: (-row['score'], row['text']))
    raw = next((deepcopy(row) for row in baseline['candidates']
                if row['text'] == source), None)
    if raw is None:
        raw = {'text': source, 'score': None, 'segments': [{
            'input_start': 0, 'input_end': len(source), 'source': source,
            'text': source, 'kind': 'raw'}]}
    candidates = [{'text': row['text'], 'score': round(row['score'], 4),
                   'segments': deepcopy(row['segments'])} for row in ranked[:4]]
    candidates.append(raw)
    response = deepcopy(baseline)
    response.update(candidates=candidates, result=candidates[0]['text'],
                    ranking_applied=True, sentence_ranking_applied=True,
                    context_used=False, external_context_used=False)
    # The saved baseline retains these parent-only diagnostics. They do not
    # describe the replacement public ordering and must not masquerade as such.
    for name in ('phonetic_choice', 'phonetic_score_scale'):
        response.pop(name, None)
    response['ungated_joint'] = {
        'experimental': True, 'parent_version': baseline.get('version'),
        'selection': 'sentence_ranker_unscaled',
        'generation_requests': generated['generation_requests'],
        'pool_size': len(pool), 'supported': len(supported),
        'unsupported': len(pool) - len(supported),
        'parent_warnings': list(baseline.get('warnings', [])),
    }
    old_length_warning = '문장 모델의 길이 범위를 넘어 단어별 문맥 순위를 사용했습니다.'
    response['warnings'] = [warning for warning in baseline.get('warnings', [])
                            if warning != old_length_warning]
    if len(supported) < len(pool):
        response['warnings'].append('문장 모델이 읽을 수 없는 후보는 진단 목록에 보존했습니다.')
    trace.update(applied=True, reason='recovered',
                 supported_order=[row['text'] for row in ranked])
    return response, trace


class UngatedJointConverter:
    """Wrap an existing converter; keep all already-ranked responses unchanged."""

    def __init__(self, parent, *, known_english=()):
        self.parent = parent
        self.known_english = frozenset(known_english)
        self.lock = getattr(parent, 'lock', threading.RLock())
        self.last_trace = {}
        paths = [Path(__file__), Path(__file__).with_name('ungated_joint_proposals.py'),
                 Path(__file__).with_name('ungated_phonetic_proposals.py'),
                 Path(__file__).with_name('source_edge_recombination.py'),
                 Path(__file__).with_name('source_correspondence.py')]
        self.artifacts = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in paths}

    def convert(self, text, context='', engine='neural', mode='batch', raw=False):
        started = time.perf_counter()
        with self.lock:
            self.last_trace = {'applied': False, 'reason': 'parent_passthrough'}
            baseline = self.parent.convert(text, context, engine, mode, raw)
            if (engine != 'neural' or raw or context.strip()
                    or baseline.get('sentence_ranking_applied')):
                return baseline
            ranker = getattr(self.parent, 'sentence_ranker', None)
            if ranker is None:
                self.last_trace['reason'] = 'no_sentence_ranker'
                return baseline

            def generate(word, beam_size=8):
                return self.parent.load_phonetic_model().generate(word, beam_size=beam_size)

            response, self.last_trace = recover_unranked(
                text, baseline, generate, ranker, known_english=self.known_english)
            if not self.last_trace['applied']:
                return baseline
            identity = {'parent': baseline['version'], 'artifacts': self.artifacts,
                        'known_english': sorted(self.known_english)}
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
            response['version'] = 'experimental-ungated-joint-1-' + digest
            response['model_artifacts'] = {
                **response.get('model_artifacts', {}), **self.artifacts}
            response['engine_ms'] = round((time.perf_counter() - started) * 1000, 3)
            return response


__all__ = ['UngatedJointConverter', 'recover_unranked']
