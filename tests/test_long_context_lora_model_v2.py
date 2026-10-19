from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from scripts import long_context_lora_model as frozen
from scripts import long_context_lora_model_v2 as grouped


class Scalar:
    item_calls = 0

    def __init__(self, value):
        self.value = bool(value)
        self.ndim = 0

    def item(self):
        type(self).item_calls += 1
        return self.value


class FakeMX:
    float32 = np.float32

    def __init__(self):
        self.stack_ndims = []
        self.eval_calls = []

    def isfinite(self, value):
        return np.isfinite(value)

    def all(self, value):
        if isinstance(value, Scalar):
            return value
        return Scalar(np.all([item.value for item in value])
                      if isinstance(value, list) and value and isinstance(value[0], Scalar)
                      else np.all(value))

    def stack(self, values):
        self.stack_ndims.append([value.ndim for value in values])
        return list(values)

    def eval(self, *values):
        self.eval_calls.append(values)

    def array(self, value, dtype=None):
        return np.array(value, dtype=dtype)

    def sum(self, value):
        return np.sum(value)

    def sqrt(self, value):
        return np.sqrt(value)

    def minimum(self, left, right):
        return np.minimum(left, right)

    def maximum(self, left, right):
        return np.maximum(left, right)


def arrays16():
    return {f"tensor-{index:02d}": np.full((index % 4 + 1,), index + 1.0,
                                           dtype=np.float32)
            for index in range(16)}


def test_group_reduces_each_array_then_reads_one_host_bool():
    mx, counters = FakeMX(), {}
    Scalar.item_calls = 0
    predicate = grouped.grouped_finite_predicate(mx, arrays16(), counters=counters)
    assert Scalar.item_calls == 0 and counters == {}
    grouped.require_grouped_finite(mx, arrays16(), message="bad", counters=counters)
    assert Scalar.item_calls == 1
    assert counters == {"predicate_host_reads": 1}
    assert mx.stack_ndims == [[0] * 16, [0] * 16]
    assert predicate.ndim == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_group_rejects_each_nonfinite_kind_in_every_named_array(bad):
    for name in arrays16():
        values, mx, counters = arrays16(), FakeMX(), {}
        values[name][-1] = bad
        with pytest.raises(FloatingPointError, match="bad"):
            grouped.require_grouped_finite(mx, values, message="bad", counters=counters)
        assert counters == {"predicate_host_reads": 1}


def test_empty_group_rejects_without_host_read():
    with pytest.raises(ValueError, match="nonempty"):
        grouped.grouped_finite_predicate(FakeMX(), {}, counters={})


@pytest.mark.parametrize("count,rows", [
    (4, [[1.0, 3.0], [2.0, 4.0], [3.0, 5.0], [4.0, 6.0]]),
    (3, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
])
def test_grouped_clipping_preserves_frozen_arithmetic(monkeypatch, count, rows):
    monkeypatch.setitem(sys.modules, "mlx.utils", types.SimpleNamespace(
        tree_map=lambda fn, tree: {key: fn(value) for key, value in tree.items()},
        tree_flatten=lambda tree: list(tree.items())))
    gradients = {"value": np.asarray(rows, dtype=np.float32).sum(axis=0)}
    expected, expected_norm = frozen.reference_average_and_clip(rows)
    clipped, norm = grouped.average_and_clip_gradients(
        FakeMX(), gradients, count, counters={})
    assert np.allclose(clipped["value"], expected)
    assert float(norm) == pytest.approx(expected_norm)


def test_source_stacks_only_scalar_predicates():
    source = open(grouped.__file__).read()
    assert "mx.stack(predicates)" in source
    assert "concatenate" not in source and "flatten(" not in source
