"""Small antisymmetric head for ordered candidate-pair representations."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SymmetricPairHead(nn.Module):
    """Score alternative-vs-current pairs with exact swap antisymmetry."""

    def __init__(self, hidden_size: int):
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        self.hidden_size = hidden_size
        self.projection = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.projection.weight)

    def _validate_features(self, features: torch.Tensor, name: str) -> None:
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not features.is_floating_point():
            raise ValueError(f"{name} must be floating point")
        if features.ndim < 1 or features.shape[-1] != self.hidden_size:
            raise ValueError(
                f"{name} must have final dimension {self.hidden_size}"
            )
        if features.dtype != self.projection.weight.dtype:
            raise ValueError(f"{name} dtype must match the head dtype")
        if features.device != self.projection.weight.device:
            raise ValueError(f"{name} device must match the head device")
        if not bool(torch.isfinite(features).all()):
            raise ValueError(f"{name} contains a non-finite value")

    def score_features(self, delta: torch.Tensor) -> torch.Tensor:
        """Score a precomputed half-difference feature vector."""
        self._validate_features(delta, "delta")
        margins = self.projection(delta).squeeze(-1)
        if not bool(torch.isfinite(margins).all()):
            raise ValueError("pair head produced a non-finite margin")
        return margins

    def forward(self, alternative_first: torch.Tensor,
                current_first: torch.Tensor) -> torch.Tensor:
        """Return a margin where positive means the alternative wins."""
        self._validate_features(alternative_first, "alternative_first")
        self._validate_features(current_first, "current_first")
        if alternative_first.shape != current_first.shape:
            raise ValueError("ordered pair feature shapes must match")
        return self.score_features(0.5 * (alternative_first - current_first))


def relation_loss(margins: torch.Tensor, relations: torch.Tensor,
                  row_ids: torch.Tensor) -> torch.Tensor:
    """Average relation loss within each row, then equally across rows.

    Relations are ``+1`` when only the alternative is accepted, ``-1`` when
    only the current candidate is accepted, and ``0`` when both or neither are
    accepted.  Neutral pairs only penalize a false alternative preference.
    """
    for value, name in (
        (margins, "margins"),
        (relations, "relations"),
        (row_ids, "row_ids"),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    if margins.numel() == 0:
        raise ValueError("relation loss requires at least one pair")
    if margins.shape != relations.shape or margins.shape != row_ids.shape:
        raise ValueError("margins, relations, and row_ids must have equal shapes")
    if not margins.is_floating_point():
        raise ValueError("margins must be floating point")
    if relations.dtype == torch.bool or relations.is_complex():
        raise ValueError("relations must contain only -1, 0, or +1")
    if row_ids.dtype == torch.bool or row_ids.is_floating_point() or row_ids.is_complex():
        raise ValueError("row_ids must use an integer dtype")
    if relations.device != margins.device or row_ids.device != margins.device:
        raise ValueError("all loss inputs must be on the same device")
    if not bool(torch.isfinite(margins).all()):
        raise ValueError("margins contain a non-finite value")
    relation_values = relations.to(dtype=margins.dtype)
    if not bool(torch.isfinite(relation_values).all()):
        raise ValueError("relations contain a non-finite value")
    if not bool(((relation_values == -1) | (relation_values == 0)
                 | (relation_values == 1)).all()):
        raise ValueError("relations must contain only -1, 0, or +1")

    informative = relation_values != 0
    pair_losses = torch.where(
        informative,
        F.softplus(-relation_values * margins),
        F.relu(margins),
    )
    _, inverse = torch.unique(row_ids, sorted=True, return_inverse=True)
    row_count = int(inverse.max().item()) + 1
    row_sums = torch.zeros(
        row_count, dtype=pair_losses.dtype, device=pair_losses.device
    ).scatter_add_(0, inverse, pair_losses)
    counts = torch.zeros(
        row_count, dtype=pair_losses.dtype, device=pair_losses.device
    ).scatter_add_(0, inverse, torch.ones_like(pair_losses))
    return (row_sums / counts).mean()
