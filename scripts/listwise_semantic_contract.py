"""Pure serialization, objective, and decision contract for listwise selection."""
from __future__ import annotations

import json
import math
from collections.abc import Sequence

import torch

from scripts.semantic_pair_contract import bound_context


def serialize_candidate(roman: str, context: str, candidate_text: str) -> str:
    """Serialize source text and one candidate without action/provenance metadata."""
    if not isinstance(roman, str):
        raise TypeError("roman must be a string")
    if not isinstance(candidate_text, str):
        raise TypeError("candidate_text must be a string")
    payload = {
        "candidate_text": candidate_text,
        "context": bound_context(context),
        "roman": roman,
    }
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _validated_rows(scores: torch.Tensor, row_sizes, positive_indices):
    if not isinstance(scores, torch.Tensor) or scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional tensor")
    if not scores.is_floating_point() or not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite floating point values")
    if (not isinstance(row_sizes, Sequence) or isinstance(row_sizes, (str, bytes))
            or not row_sizes):
        raise ValueError("row_sizes must contain nonempty rows")
    sizes = tuple(row_sizes)
    if any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError("row sizes must be positive integers")
    if sum(sizes) != scores.numel():
        raise ValueError("row sizes do not close over scores")
    if (not isinstance(positive_indices, Sequence)
            or isinstance(positive_indices, (str, bytes))
            or len(positive_indices) != len(sizes)):
        raise ValueError("one positive-index sequence is required per row")
    positives = []
    for size, indices in zip(sizes, positive_indices):
        if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
            raise ValueError("positive indices must be sequences")
        indices = tuple(indices)
        if any(type(index) is not int or not 0 <= index < size for index in indices):
            raise ValueError("positive index is not an in-range integer")
        if len(set(indices)) != len(indices):
            raise ValueError("positive indices must be unique")
        positives.append(indices)
    return sizes, tuple(positives)


def listwise_none_loss(
    scores: torch.Tensor,
    row_sizes: Sequence[int],
    positive_indices: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Equal-row mean likelihood loss with a fixed zero-logit NONE class.

    Accepted candidates form the numerator when present. An all-wrong row uses
    NONE as its sole accepted class, so such rows train candidate scores down
    instead of disappearing from the objective.
    """
    sizes, positives = _validated_rows(scores, row_sizes, positive_indices)
    none = scores.new_zeros(1)
    losses = []
    offset = 0
    for size, accepted in zip(sizes, positives):
        row = scores[offset:offset + size]
        denominator = torch.logsumexp(torch.cat((row, none)), dim=0)
        if accepted:
            index = torch.tensor(accepted, dtype=torch.long, device=scores.device)
            numerator = torch.logsumexp(row.index_select(0, index), dim=0)
        else:
            numerator = none[0]
        losses.append(denominator - numerator)
        offset += size
    loss = torch.stack(losses).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("listwise NONE loss is non-finite")
    return loss


def decision(scores: list[float], current_index: int) -> dict:
    """Select one positive-scoring candidate or explicitly abstain to KEEP."""
    if not isinstance(scores, list) or not scores:
        raise ValueError("scores must be a nonempty list")
    if any(type(score) not in (int, float) or not math.isfinite(score)
           for score in scores):
        raise ValueError("scores must contain finite numbers")
    if type(current_index) is not int or not 0 <= current_index < len(scores):
        raise ValueError("current_index must identify a candidate")
    maximum = max(scores)
    if maximum <= 0.0:
        return {"selected_index": current_index, "abstained": True, "reason": "none"}
    winners = [index for index, score in enumerate(scores) if score == maximum]
    if len(winners) != 1:
        return {"selected_index": current_index, "abstained": True, "reason": "tie"}
    return {"selected_index": winners[0], "abstained": False, "reason": "candidate"}


__all__ = ["decision", "listwise_none_loss", "serialize_candidate"]
