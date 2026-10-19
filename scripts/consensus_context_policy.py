"""Source-only consensus policy for two independently frozen selectors."""
from __future__ import annotations

import hashlib
import json
import random

from scripts.final_choice_policy import proposal_allowed, reorder_candidates


MAX_CHOICES = 5


def _validate_source(source):
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a nonempty string")


def choices(source, candidates):
    """Return permitted ``(source_index, text)`` choices in parent order."""
    _validate_source(source)
    if (not isinstance(candidates, list) or not candidates
            or len(candidates) > MAX_CHOICES):
        raise ValueError("candidates must contain one to five objects")
    result = []
    seen = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError("candidate must be an object")
        text = candidate.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("candidate text must be nonempty")
        if text in seen:
            raise ValueError("candidate texts must be unique")
        seen.add(text)
        result.append((index, text))
    if len(result) > 1 and result[0][1] != source:
        result = [item for item in result if item[1] != source]
    if not result:
        raise ValueError("candidate filtering removed every choice")
    return result


def prompt_choices(source, context, candidates):
    """Return the historical deterministic prompt permutation."""
    if not isinstance(context, str):
        raise ValueError("context must be a string")
    result = list(choices(source, candidates))
    seed = hashlib.sha256(
        json.dumps([source, context], ensure_ascii=False).encode()
    ).hexdigest()
    random.Random(seed).shuffle(result)
    return result


def _validate_prediction(index, permitted, label):
    if type(index) is not int or index not in permitted:
        raise ValueError(f"{label} prediction is not a permitted source index")


def decide(source, candidates, base_index, trained_index, raw=False):
    """Apply a protected non-parent proposal only under exact selector consensus."""
    if raw:
        return {"chosen_index": 0, "result": source, "candidates": candidates,
                "reason": "raw_mode"}
    permitted_choices = choices(source, candidates)
    permitted = {index for index, _ in permitted_choices}
    _validate_prediction(base_index, permitted, "base")
    _validate_prediction(trained_index, permitted, "trained")
    if base_index != trained_index:
        return {"chosen_index": 0, "result": candidates[0]["text"],
                "candidates": candidates, "reason": "selector_disagreement"}
    if base_index == 0:
        return {"chosen_index": 0, "result": candidates[0]["text"],
                "candidates": candidates, "reason": "parent_consensus"}
    if not proposal_allowed(source, candidates[0], candidates[base_index]):
        return {"chosen_index": 0, "result": candidates[0]["text"],
                "candidates": candidates, "reason": "protected_or_unaligned"}
    reordered = reorder_candidates(candidates, base_index)
    return {"chosen_index": base_index, "result": reordered[0]["text"],
            "candidates": reordered, "reason": "consensus_proposal"}


__all__ = ["choices", "prompt_choices", "decide"]
