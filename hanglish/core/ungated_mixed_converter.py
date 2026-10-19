"""Experimental mixed-boundary recovery layered over ungated joint recovery."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from hanglish.core import mixed_span_proposals
from hanglish.core.ungated_joint_converter import UngatedJointConverter


def _raw_reserve(source, baseline):
    raw = next((deepcopy(row) for row in baseline.get('candidates', [])
                if row.get('text') == source and row.get('segments')
                and all(segment.get('kind') == 'raw' for segment in row['segments'])), None)
    return raw or {'text': source, 'score': None, 'segments': [{
        'input_start': 0, 'input_end': len(source), 'source': source,
        'text': source, 'kind': 'raw'}]}


def recover_mixed(source, joint_response, joint_trace, word_options, ranker):
    """Add mixed-span proposals while retaining every joint diagnostic candidate."""
    baseline = joint_trace.get('baseline')
    if not isinstance(baseline, dict) or baseline.get('input') != source:
        raise ValueError('original production baseline missing')
    parents = [(row['text'], 0.0, deepcopy(row['segments']))
               for row in baseline.get('candidates', [])
               if row.get('text') != source and row.get('segments')
               and not all(segment.get('kind') == 'raw' for segment in row['segments'])][:4]
    old_pool = deepcopy(joint_trace.get('pool', []))
    by_text = {}
    for row in old_pool:
        if (not isinstance(row.get('text'), str) or not row['text']
                or type(row.get('supported')) is not bool
                or (row['supported'] and (isinstance(row.get('score'), bool)
                    or not isinstance(row.get('score'), (int, float))
                    or not math.isfinite(row['score'])))
                or (not row['supported'] and row.get('score') is not None)):
            raise ValueError('joint pool candidate differs')
        row['origins'] = list(row.get('origins', []))
        if row['text'] in by_text:
            raise ValueError('joint pool contains duplicate text')
        by_text[row['text']] = row
    old_ranked = sorted((row for row in old_pool if row['supported']),
                        key=lambda row: (-row['score'], row['text']))
    trace = {'applied': False, 'reason': 'no_mixed_proposals',
             'baseline': deepcopy(baseline), 'joint_trace': deepcopy(joint_trace),
             'mixed_proposals': [], 'union_pool': old_pool,
             'supported_order': [row['text'] for row in old_ranked]}
    if not parents:
        trace['reason'] = 'no_nonraw_parents'
        return joint_response, trace
    extras = mixed_span_proposals.sentence_extras(
        source, parents, word_options, limit=96)
    trace['mixed_proposals'] = [
        {'text': text, 'score': float(score), 'segments': deepcopy(segments)}
        for text, score, segments in extras]
    if not extras:
        return joint_response, trace

    new_supported = []
    for text, _, segments in extras:
        supported = bool(ranker.can_score(text))
        previous = by_text.get(text)
        if previous is not None:
            if previous['supported'] != supported:
                raise ValueError('duplicate candidate support differs')
            if 'mixed' not in previous['origins']:
                previous['origins'].append('mixed')
            continue
        candidate = {'text': text, 'segments': deepcopy(segments),
                     'origins': ['mixed'], 'supported': supported, 'score': None}
        old_pool.append(candidate); by_text[text] = candidate
        if supported:
            new_supported.append(candidate)
    trace['union_pool'] = old_pool
    if not any(row['text'] not in {old['text'] for old in joint_trace.get('pool', [])}
               for row in old_pool):
        trace['reason'] = 'no_new_candidates'
        return joint_response, trace
    scores = ranker.score_pairs([''] * len(new_supported),
                                [row['text'] for row in new_supported]) if new_supported else []
    if (len(scores) != len(new_supported)
            or any(isinstance(score, bool) or not isinstance(score, (int, float))
                   or not math.isfinite(score) for score in scores)):
        raise ValueError('sentence ranker returned invalid mixed scores')
    for candidate, score in zip(new_supported, scores):
        candidate['score'] = float(score)
    supported = [row for row in old_pool if row['supported']]
    if any(not isinstance(row.get('score'), (int, float))
           or not math.isfinite(row['score']) for row in supported):
        raise ValueError('supported union candidate lacks a finite score')
    ranked = sorted(supported, key=lambda row: (-row['score'], row['text']))
    trace['supported_order'] = [row['text'] for row in ranked]
    if not ranked:
        trace['reason'] = 'no_supported_candidates'
        return joint_response, trace
    public = [{'text': row['text'], 'score': round(row['score'], 4),
               'segments': deepcopy(row['segments'])} for row in ranked[:4]]
    public.append(_raw_reserve(source, baseline))
    if [row['text'] for row in public] == [row.get('text') for row in joint_response['candidates']]:
        trace['reason'] = 'public_unchanged'
        return joint_response, trace
    response = deepcopy(joint_response)
    response.update(candidates=public, result=public[0]['text'], ranking_applied=True,
                    sentence_ranking_applied=True, context_used=False,
                    external_context_used=False)
    for name in ('phonetic_choice', 'phonetic_score_scale'):
        response.pop(name, None)
    old_length_warning = '문장 모델의 길이 범위를 넘어 단어별 문맥 순위를 사용했습니다.'
    warnings = [warning for warning in baseline.get('warnings', [])
                if warning != old_length_warning]
    if any(not row['supported'] for row in old_pool):
        warning = '문장 모델이 읽을 수 없는 후보는 진단 목록에 보존했습니다.'
        if warning not in warnings:
            warnings.append(warning)
    response['warnings'] = warnings
    joint_parent = response.pop('ungated_joint', None)
    response['ungated_mixed'] = {
        'experimental': True, 'selection': 'sentence_ranker_unscaled',
        'pool_size': len(old_pool), 'supported': len(ranked),
        'unsupported': len(old_pool) - len(ranked),
        'mixed_proposals': len(extras), 'new_supported_scored': len(new_supported),
        'parent_warnings': list(baseline.get('warnings', [])),
        'joint_parent': deepcopy(joint_parent),
    }
    trace.update(applied=True, reason='recovered')
    return response, trace


class UngatedMixedConverter:
    """Compose mixed-boundary recovery over the existing joint adapter."""

    def __init__(self, parent, *, known_english=()):
        self.parent = parent
        self.joint = UngatedJointConverter(parent, known_english=known_english)
        self.lock = getattr(parent, 'lock', threading.RLock())
        self.last_trace = {}
        paths = [Path(__file__), Path(mixed_span_proposals.__file__)]
        self.artifacts = {**self.joint.artifacts,
                          **{str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in paths}}

    @property
    def models_loaded(self):
        return getattr(self.parent, 'models_loaded', False)

    def load_models(self):
        return self.parent.load_models()

    def _version(self, parent_version):
        identity = {'parent': parent_version, 'artifacts': self.artifacts,
                    'known_english': sorted(self.joint.known_english)}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        return 'experimental-ungated-mixed-1-' + digest

    @property
    def version(self):
        return self._version(getattr(self.parent, 'version', None))

    def convert(self, text, context='', engine='neural', mode='batch', raw=False):
        started = time.perf_counter()
        with self.lock:
            response = self.joint.convert(text, context, engine, mode, raw)
            joint_trace = self.joint.last_trace
            self.last_trace = {'applied': False, 'reason': 'joint_passthrough',
                               'joint_trace': deepcopy(joint_trace)}
            if engine != 'neural' or raw or context.strip():
                return response
            baseline = joint_trace.get('baseline')
            if not isinstance(baseline, dict) or baseline.get('sentence_ranking_applied'):
                return response
            ranker = getattr(self.parent, 'sentence_ranker', None)
            word_options = getattr(self.parent, 'word_options', None)
            if ranker is None or not callable(word_options):
                self.last_trace['reason'] = 'missing_parent_capability'
                return response
            response, self.last_trace = recover_mixed(
                text, response, joint_trace, word_options, ranker)
            if not self.last_trace['applied']:
                return response
            response['version'] = self._version(baseline.get('version'))
            response['model_artifacts'] = {
                **response.get('model_artifacts', {}), **self.artifacts}
            response['engine_ms'] = round((time.perf_counter() - started) * 1000, 3)
            return response


__all__ = ['UngatedMixedConverter', 'recover_mixed']
