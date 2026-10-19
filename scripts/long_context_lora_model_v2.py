"""Grouped finite predicates for the unchanged long-context LoRA arithmetic."""
from __future__ import annotations

import math
from typing import Any, Mapping

from scripts.long_context_lora_model import (
    CLIP_NORM, flat_tree, require_matching_gradient_tree,
    require_trainable_contract,
)


def _count(counters: dict[str, int] | None, field: str) -> None:
    if counters is not None:
        counters[field] = counters.get(field, 0) + 1


def grouped_finite_predicate(mx: Any, arrays: Mapping[str, Any], *,
                             counters: dict[str, int] | None = None) -> Any:
    """Reduce every array separately, stack only scalars, and read one host bool."""
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("finite predicate group must be nonempty")
    predicates = []
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise ValueError("finite predicate names must be nonempty strings")
        predicates.append(mx.all(mx.isfinite(value)))
    return mx.all(mx.stack(predicates))


def require_grouped_finite(mx: Any, arrays: Mapping[str, Any], *, message: str,
                           counters: dict[str, int] | None = None,
                           evaluate_with: tuple[Any, ...] = ()) -> None:
    predicate = grouped_finite_predicate(mx, arrays, counters=counters)
    mx.eval(*evaluate_with, predicate)
    finite = bool(predicate.item())
    _count(counters, "predicate_host_reads")
    if not finite:
        raise FloatingPointError(message)


def read_numeric_scalar(value: Any, *, counters: dict[str, int] | None = None) -> float:
    """Read a report scalar without counting it as a finite-predicate host read."""
    result = float(value.item())
    _count(counters, "numeric_host_reads")
    if not math.isfinite(result):
        raise FloatingPointError("reported scalar is nonfinite")
    return result


def materialize_microbatch(mx: Any, model: Any, loss: Any, gradients: Any, *,
                           counters: dict[str, int] | None = None) -> dict[str, Any]:
    flat = require_matching_gradient_tree(model, gradients)
    require_grouped_finite(
        mx, {"loss": loss, **flat}, message="nonfinite loss or gradient",
        counters=counters, evaluate_with=(loss, gradients))
    return flat


def average_and_clip_gradients(mx: Any, gradients: Any, microbatch_count: int,
                               max_norm: float = CLIP_NORM, *,
                               counters: dict[str, int] | None = None):
    """Run the frozen averaging/clipping arithmetic with one grouped finite read."""
    from mlx.utils import tree_map

    if type(microbatch_count) is not int or microbatch_count < 1:
        raise ValueError("microbatch_count must be positive")
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be finite and positive")
    averaged = tree_map(lambda value: value / microbatch_count, gradients)
    squared = sum((mx.sum(value.astype(mx.float32) ** 2)
                   for value in flat_tree(averaged).values()), mx.array(0.0, mx.float32))
    norm = mx.sqrt(squared)
    scale = mx.minimum(mx.array(1.0, mx.float32), max_norm / mx.maximum(norm, 1e-12))
    clipped = tree_map(lambda value: value * scale.astype(value.dtype), averaged)
    require_grouped_finite(
        mx, {"norm": norm, **flat_tree(clipped)},
        message="averaged/clipped gradients are nonfinite", counters=counters,
        evaluate_with=(norm, clipped))
    return clipped, norm


def apply_materialized_update(mx: Any, model: Any, optimizer: Any, gradients: Any, *,
                              counters: dict[str, int] | None = None) -> None:
    optimizer.update(model, gradients)
    adapter = require_trainable_contract(model)
    require_grouped_finite(
        mx, adapter, message="updated adapter parameters are nonfinite",
        counters=counters, evaluate_with=(model.state, optimizer.state))


__all__ = [
    "apply_materialized_update", "average_and_clip_gradients",
    "grouped_finite_predicate", "materialize_microbatch",
    "read_numeric_scalar", "require_grouped_finite",
]
