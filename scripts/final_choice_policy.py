"""Frozen source-only action policy for the experimental final-choice selector.

This module changes no runtime path and reads no labels. Training and inference
must use the same finite candidate indices and candidate-object score fields.
"""
from __future__ import annotations

import math
import re

from hanglish.core.phonology import romanize_word

ALLOWED_KINDS = frozenset({'lexicon', 'morphology', 'syllable', 'neural'})
FEATURE_NAMES = (
    'parent_gap', 'proposal_parent_original_delta',
    'proposal_parent_residual_delta', 'residual_gap',
    'proposal_original_index', 'external_context_length',
)


def finite_candidate_indices(candidates):
    indices = []
    for index, candidate in enumerate(candidates):
        score = candidate.get('score')
        if score is None:
            continue
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError('candidate score must be finite or None')
        indices.append(index)
    return indices


def _segment_spans(source, candidate):
    segments = candidate.get('segments')
    if not isinstance(segments, list) or not segments:
        return None
    cursor, spans, outputs = 0, [], []
    for segment in segments:
        if not isinstance(segment, dict):
            return None
        start, end = segment.get('input_start'), segment.get('input_end')
        if (type(start) is not int or type(end) is not int
                or start != cursor or not start < end <= len(source)
                or segment.get('source') != source[start:end]
                or not isinstance(segment.get('text'), str)):
            return None
        spans.append((start, end, segment['source']))
        outputs.append(segment['text'])
        cursor = end
    if cursor != len(source) or candidate.get('text') != ''.join(outputs):
        return None
    return spans


def proposal_allowed(source, parent, proposal):
    parent_spans = _segment_spans(source, parent)
    proposal_spans = _segment_spans(source, proposal)
    if parent_spans is None or parent_spans != proposal_spans:
        return False
    changed = False
    for left, right in zip(parent['segments'], proposal['segments']):
        if left['text'] == right['text']:
            continue
        changed = True
        if (left.get('kind') not in ALLOWED_KINDS
                or right.get('kind') not in ALLOWED_KINDS
                or re.fullmatch('[a-z]+', right['source']) is None
                or re.fullmatch('[가-힣]+', right['text']) is None
                or romanize_word(right['text']) != right['source']):
            return False
    return changed


def decision_features(source, context, candidates, residual_scores, *, raw=False):
    """Return a proposed index and six features, or an explicit stay reason.

The caller must also establish that every candidate fits the frozen sentence
model. No label, target string, or candidate-pool score is accepted here.
"""
    if raw:
        return {'reason': 'raw_mode', 'proposal_index': 0}
    if (not isinstance(candidates, list) or not candidates
            or len(residual_scores) != len(candidates)
            or any(type(score) not in (int, float) or not math.isfinite(score)
                   for score in residual_scores)):
        raise ValueError('residual scores must match the complete candidate list')
    indices = finite_candidate_indices(candidates)
    if len(indices) < 2 or 0 not in indices:
        return {'reason': 'insufficient_finite_candidates', 'proposal_index': 0}
    ranked = sorted(indices, key=lambda index: (-residual_scores[index], index))
    proposal_indices = [index for index in ranked if candidates[index]['text'] != source]
    if not proposal_indices:
        return {'reason': 'no_non_original_candidate', 'proposal_index': 0}
    proposed = proposal_indices[0]
    if proposed == 0:
        return {'reason': 'parent_selected', 'proposal_index': 0}
    if not proposal_allowed(source, candidates[0], candidates[proposed]):
        return {'reason': 'protected_or_unaligned', 'proposal_index': 0,
                'rejected_proposal_index': proposed}
    original = [candidate['score'] for candidate in candidates]
    features = [
        original[indices[0]] - original[indices[1]],
        original[proposed] - original[0],
        residual_scores[proposed] - residual_scores[0],
        residual_scores[ranked[0]] - residual_scores[ranked[1]],
        float(proposed),
        min(len(context.strip()), 4096) / 4096,
    ]
    if any(not math.isfinite(value) for value in features):
        raise ValueError('gate feature overflow')
    return {'reason': 'eligible_disagreement', 'proposal_index': proposed,
            'finite_indices': indices, 'features': features}


def reorder_candidates(candidates, chosen_index):
    if type(chosen_index) is not int or not 0 <= chosen_index < len(candidates):
        raise ValueError('chosen candidate index is out of bounds')
    return [candidates[chosen_index], *candidates[:chosen_index],
            *candidates[chosen_index + 1:]]
