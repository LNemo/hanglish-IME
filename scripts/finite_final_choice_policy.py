"""Experimental candidate selection with the unscored original kept outside ML.

Only candidates with finite parent scores consume the model's token budget.
The complete returned objects, including the original, remain available to users.
This module does not modify the deployed converter or read target labels.
"""
import math

from scripts.final_choice_policy import (
    finite_candidate_indices, proposal_allowed, reorder_candidates,
)


def candidate_view(ranker, source, candidates):
    """Return original indices and source-only eligibility shared with training."""
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('a complete nonempty candidate list is required')
    omitted = 0
    for candidate in candidates:
        if not isinstance(candidate, dict) or 'score' not in candidate:
            raise ValueError('candidate requires an explicit parent score')
        if candidate['score'] is None:
            omitted += 1
            if omitted > 1 or candidate != {
                    'text': source, 'score': None, 'segments': [
                        {'input_start': 0, 'input_end': len(source),
                         'source': source, 'text': source, 'kind': 'raw'},
                    ]}:
                raise ValueError('only the single complete raw original may be unscored')
    indices = finite_candidate_indices(candidates)
    if 0 not in indices or len(indices) < 2:
        return indices, 'insufficient_finite_candidates'
    if not all(ranker.can_score(candidates[index]['text']) for index in indices):
        return indices, 'candidate_budget'
    return indices, None


def select_candidates(ranker, source, context, candidates, *, raw=False):
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('a complete nonempty candidate list is required')
    chosen, indices, scores = 0, [], []
    if raw:
        reason = 'raw_mode'
    else:
        indices, reason = candidate_view(ranker, source, candidates)
        if reason is None:
            scores = list(ranker.score_pairs(
                [context] * len(indices), [candidates[index]['text'] for index in indices],
            ))
            if len(scores) != len(indices) or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in scores):
                raise ValueError('selector scores must cover every finite candidate')
            ranked = sorted(range(len(indices)), key=lambda position: (-scores[position], indices[position]))
            proposals = [indices[position] for position in ranked
                         if candidates[indices[position]]['text'] != source]
            if not proposals:
                reason = 'no_non_original_candidate'
            elif proposals[0] == 0:
                reason = 'parent_selected'
            elif not proposal_allowed(source, candidates[0], candidates[proposals[0]]):
                reason = 'protected_or_unaligned'
            else:
                chosen, reason = proposals[0], 'eligible_disagreement'
    ordered = reorder_candidates(candidates, chosen) if chosen else candidates
    return {'chosen_index': chosen, 'reason': reason,
            'scored_source_indices': indices if scores else [],
            'selector_scores': scores, 'selector_scored': bool(scores),
            'candidates': ordered, 'result': ordered[0]['text']}
