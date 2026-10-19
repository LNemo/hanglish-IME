"""Role-free serialization and loss contract for semantic candidate pairs."""
from __future__ import annotations

from enum import Enum
import json
import unicodedata

import torch
from torch.nn import functional as F

from scripts.symmetric_pair_head import SymmetricPairHead

CONTEXT_EDGE = 64
CONTEXT_LIMIT = 2 * CONTEXT_EDGE
CONTEXT_OMISSION_MARKER = "\n<CONTEXT_OMITTED>\n"


class LabelState(Enum):
    A_BETTER = 1
    B_BETTER = -1
    TIE = 0
    UNKNOWN = None


def bound_context(context: str) -> str:
    """Lossy context view; the marker is plain model text, never parsed syntax."""
    if not isinstance(context, str):
        raise TypeError("context must be a string")
    if len(context) <= CONTEXT_LIMIT:
        return context
    return (context[:CONTEXT_EDGE] + CONTEXT_OMISSION_MARKER
            + context[-CONTEXT_EDGE:])


def canonical_candidates(candidate_a_text: str, candidate_b_text: str):
    """Return deterministic A/B text order and whether inputs were swapped."""
    if not isinstance(candidate_a_text, str) or not isinstance(candidate_b_text, str):
        raise TypeError("candidate texts must be strings")
    keyed = sorted(
        ((unicodedata.normalize("NFC", text), text, index)
         for index, text in enumerate((candidate_a_text, candidate_b_text))),
        key=lambda item: (item[0], item[1], item[2]),
    )
    first, second = keyed
    return first[1], second[1], first[2] == 1


def serialize_pair(roman: str, context: str, candidate_a_text: str,
                   candidate_b_text: str) -> bytes:
    """Serialize only source text and the supplied, ordered candidate pair."""
    if not isinstance(roman, str):
        raise TypeError("roman must be a string")
    if not isinstance(candidate_a_text, str) or not isinstance(candidate_b_text, str):
        raise TypeError("candidate texts must be strings")
    payload = {
        "candidate_a_text": candidate_a_text,
        "candidate_b_text": candidate_b_text,
        "context": bound_context(context),
        "roman": roman,
    }
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def ordered_views(roman: str, context: str, candidate_a_text: str,
                  candidate_b_text: str):
    """Return the A/B and B/A serialization views in that order."""
    return (
        serialize_pair(roman, context, candidate_a_text, candidate_b_text),
        serialize_pair(roman, context, candidate_b_text, candidate_a_text),
    )


def swap_label(label: LabelState) -> LabelState:
    if not isinstance(label, LabelState):
        raise TypeError("label must be a LabelState")
    if label is LabelState.A_BETTER:
        return LabelState.B_BETTER
    if label is LabelState.B_BETTER:
        return LabelState.A_BETTER
    return label


def canonical_label(label: LabelState, inputs_swapped: bool) -> LabelState:
    if not isinstance(label, LabelState):
        raise TypeError("label must be a LabelState")
    if type(inputs_swapped) is not bool:
        raise TypeError("inputs_swapped must be bool")
    return swap_label(label) if inputs_swapped else label


def orient_margin(canonical_margin, inputs_swapped: bool):
    """Restore a canonical A-minus-B margin to the caller's original order."""
    if type(inputs_swapped) is not bool:
        raise TypeError("inputs_swapped must be bool")
    return -canonical_margin if inputs_swapped else canonical_margin


def semantic_pair_loss(margins: torch.Tensor, labels, row_ids: torch.Tensor):
    """Average informative pair loss per row, then equally across rows.

    The first recipe masks both ``TIE`` and ``UNKNOWN``. Only A/B preference
    states contribute ``softplus(-y * margin)``.
    """
    if not isinstance(margins, torch.Tensor) or not isinstance(row_ids, torch.Tensor):
        raise TypeError("margins and row_ids must be torch tensors")
    if margins.ndim != 1 or row_ids.ndim != 1 or margins.shape != row_ids.shape:
        raise ValueError("margins and row_ids must be equal one-dimensional tensors")
    if not margins.is_floating_point() or not bool(torch.isfinite(margins).all()):
        raise ValueError("margins must be finite floating point values")
    if (row_ids.dtype == torch.bool or row_ids.is_floating_point()
            or row_ids.is_complex() or row_ids.device != margins.device):
        raise ValueError("row_ids must be integer values on the margin device")
    labels = tuple(labels)
    if len(labels) != margins.numel() or any(
            not isinstance(label, LabelState) for label in labels):
        raise ValueError("labels must contain one LabelState per margin")

    informative_indices = [index for index, label in enumerate(labels)
                           if label in (LabelState.A_BETTER, LabelState.B_BETTER)]
    if not informative_indices:
        return margins[:0].sum()
    indices = torch.tensor(informative_indices, dtype=torch.long,
                           device=margins.device)
    selected_margins = margins.index_select(0, indices)
    selected_rows = row_ids.index_select(0, indices)
    relations = torch.tensor(
        [float(labels[index].value) for index in informative_indices],
        dtype=margins.dtype, device=margins.device,
    )
    pair_losses = F.softplus(-relations * selected_margins)
    _, inverse = torch.unique(selected_rows, sorted=True, return_inverse=True)
    row_count = int(inverse.max().item()) + 1
    sums = torch.zeros(row_count, dtype=margins.dtype,
                       device=margins.device).scatter_add_(0, inverse, pair_losses)
    counts = torch.zeros(row_count, dtype=margins.dtype,
                         device=margins.device).scatter_add_(
                             0, inverse, torch.ones_like(pair_losses))
    loss = (sums / counts).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("semantic pair loss is non-finite")
    return loss


__all__ = [
    "CONTEXT_OMISSION_MARKER", "LabelState", "SymmetricPairHead",
    "bound_context", "canonical_candidates", "canonical_label", "ordered_views",
    "orient_margin", "semantic_pair_loss", "serialize_pair", "swap_label",
]
