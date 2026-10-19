"""Experimental direct selector; no learned gate, labels, or runtime mutation."""
from scripts.final_choice_policy import (
    decision_features, finite_candidate_indices, reorder_candidates,
)


def select_candidates(ranker, source, context, candidates, *, raw=False):
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("a complete nonempty candidate list is required")
    scores = []
    if raw:
        decision = decision_features(source, context, candidates, scores, raw=True)
    elif 0 not in (indices := finite_candidate_indices(candidates)) or len(indices) < 2:
        decision = {"reason": "insufficient_finite_candidates", "proposal_index": 0}
    elif not all(ranker.can_score(candidate["text"]) for candidate in candidates):
        decision = {"reason": "candidate_budget", "proposal_index": 0}
    else:
        # None in a parent's score field does not change residual-vector indexing.
        scores = ranker.score_pairs([context] * len(candidates),
                                    [candidate["text"] for candidate in candidates])
        decision = decision_features(source, context, candidates, scores)
    chosen = decision["proposal_index"]
    reordered = reorder_candidates(candidates, chosen) if chosen else candidates
    return {"chosen_index": chosen, "decision": decision,
            "selector_scores": list(scores), "selector_scored": bool(scores),
            "candidates": reordered, "result": reordered[0]["text"]}
