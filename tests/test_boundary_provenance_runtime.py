import threading

import pytest

import scripts.boundary_provenance_runtime as module
from scripts.lexical_boundary_runtime import BoundaryCompositionRuntime


def segment(source, text, kind="morphology"):
    return {
        "input_start": 0,
        "input_end": len(source),
        "source": source,
        "text": text,
        "kind": kind,
    }


def path(source, text, kind="morphology", score=1.0):
    return (text, score, [segment(source, text, kind)])


def record(value):
    return {
        "text": value[0], "score": value[1],
        "segments": [dict(item) for item in value[2]],
    }


def provenance():
    return {
        "source": "gungnip", "text": "국립",
        "left_lexeme": "국", "right_lexeme": "립",
        "left_key": "guk", "right_key": "rip",
        "boundary": "changed", "surface": "ngn", "suffix": None,
    }


def converter():
    value = module.ProvenanceBoundaryRuntime.__new__(
        module.ProvenanceBoundaryRuntime
    )
    value.lock = threading.RLock()
    value.boundary_overlay_enabled = True
    value._boundary_current_text = "gungnip"
    value._boundary_current_context = ""
    return value


def response(text, *, sentence=True, engine_ms=2.0, kind="neural"):
    candidate = record(path("gungnip", text, kind))
    return {
        "candidates": [candidate], "result": text,
        "sentence_ranking_applied": sentence, "engine_ms": engine_ms,
    }


def test_parent_first_candidate_is_deepcopied_as_filter_anchor(monkeypatch):
    value = converter()
    parent = response("궁닙")
    child = response("국립")
    seen = []

    def convert(self, *args, **kwargs):
        if not self.boundary_overlay_enabled:
            seen.append("parent")
            return parent
        seen.append(self._boundary_parent_anchor)
        assert self._boundary_parent_anchor is not parent["candidates"][0]
        return child

    monkeypatch.setattr(BoundaryCompositionRuntime, "convert", convert)
    assert value.convert("gungnip")["result"] == "국립"
    assert seen[0] == "parent"
    assert seen[1] == parent["candidates"][0]
    assert value.boundary_overlay_enabled is True
    assert not hasattr(value, "_boundary_parent_anchor")


def test_only_new_boundary_extras_are_filtered_and_old_numeric_is_preserved(
        monkeypatch):
    value = converter()
    old_numeric = path("gungnip", "11월", "numeric")
    allowed = path("gungnip", "국립")
    rejected = path("gungnip", "궁립")
    value._boundary_parent_anchor = record(path("gungnip", "궁닙", "neural"))
    value.boundary_trace = {
        "activated": True, "reason": "accepted_new_extras",
        "preserved_pool": [record(path("gungnip", "궁닙", "neural"))],
        "generated_new_extras": [record(allowed), record(rejected)],
        "accepted_new_extras": [record(allowed), record(rejected)],
        "augmented_piece_count": 1, "derived_options": [provenance()],
    }
    monkeypatch.setattr(
        BoundaryCompositionRuntime,
        "_numeric_sentence_extras",
        lambda *args: [old_numeric, allowed, rejected],
    )
    actual = value._numeric_sentence_extras([], [], [])
    assert actual == [old_numeric, allowed]
    assert value.boundary_trace["accepted_new_extras"] == [record(allowed)]
    assert value.boundary_trace["preserved_pool"] == [
        record(path("gungnip", "궁닙", "neural"))
    ]
    assert set(value.boundary_trace) == {
        "activated", "reason", "preserved_pool", "generated_new_extras",
        "accepted_new_extras", "augmented_piece_count", "derived_options",
    }


@pytest.mark.parametrize("kwargs", [
    {"raw": True}, {"engine": "rules"}, {"disabled": True},
])
def test_raw_rules_and_disabled_overlay_use_one_parent_pass(monkeypatch, kwargs):
    value = converter()
    if kwargs.pop("disabled", False):
        value.boundary_overlay_enabled = False
    calls = []

    def convert(self, *args, **called):
        calls.append((args, called))
        return response("궁닙", sentence=False)

    monkeypatch.setattr(BoundaryCompositionRuntime, "convert", convert)
    actual = value.convert("gungnip", **kwargs)
    assert actual["result"] == "궁닙"
    assert len(calls) == 1


def test_parent_without_sentence_ranking_returns_exact_parent(monkeypatch):
    value = converter()
    parent = response("궁닙", sentence=False, engine_ms=7.0)
    calls = 0

    def convert(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return parent

    monkeypatch.setattr(BoundaryCompositionRuntime, "convert", convert)
    assert value.convert("gungnip") is parent
    assert calls == 1


def test_flags_and_existing_anchor_restore_after_child_error(monkeypatch):
    value = converter()
    existing = {"existing": True}
    value._boundary_parent_anchor = existing
    calls = 0

    def convert(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return response("궁닙")
        raise RuntimeError("child failure")

    monkeypatch.setattr(BoundaryCompositionRuntime, "convert", convert)
    with pytest.raises(RuntimeError, match="child failure"):
        value.convert("gungnip")
    assert value.boundary_overlay_enabled is True
    assert value._boundary_parent_anchor is existing


def test_combined_engine_time_covers_both_passes(monkeypatch):
    value = converter()
    clock = iter((10.0, 10.125))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock))
    calls = 0

    def convert(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return response("궁닙" if calls == 1 else "국립")

    monkeypatch.setattr(BoundaryCompositionRuntime, "convert", convert)
    actual = value.convert("gungnip")
    assert calls == 2
    assert actual["engine_ms"] == 125.0


def test_concurrent_call_cannot_observe_temporary_disabled_flag(monkeypatch):
    second_read = threading.Event()
    parent_entered = threading.Event()
    release_parent = threading.Event()

    class TrackingRuntime(module.ProvenanceBoundaryRuntime):
        @property
        def boundary_overlay_enabled(self):
            if threading.current_thread().name == "second":
                second_read.set()
            return self._enabled

        @boundary_overlay_enabled.setter
        def boundary_overlay_enabled(self, value):
            self._enabled = value

    value = TrackingRuntime.__new__(TrackingRuntime)
    value.lock = threading.RLock()
    value.boundary_overlay_enabled = True
    value._boundary_current_text = "gungnip"
    value._boundary_current_context = ""

    def convert(self, *args, **kwargs):
        with self.lock:
            if not self.boundary_overlay_enabled:
                if threading.current_thread().name == "first":
                    parent_entered.set()
                    assert release_parent.wait(2)
                return response("궁닙")
            assert hasattr(self, "_boundary_parent_anchor")
            return response("국립")

    monkeypatch.setattr(BoundaryCompositionRuntime, "convert", convert)
    results = []
    errors = []

    def run():
        try:
            results.append(value.convert("gungnip")["result"])
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=run, name="first")
    second = threading.Thread(target=run, name="second")
    first.start()
    assert parent_entered.wait(2)
    second.start()
    # The second thread cannot even read eligibility while the first owns the
    # shared lock with its temporary disabled flag.
    assert not second_read.wait(0.05)
    release_parent.set()
    first.join(2)
    second.join(2)
    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert sorted(results) == ["국립", "국립"]


def test_filter_requires_trace_and_returned_paths_to_match(monkeypatch):
    value = converter()
    value._boundary_parent_anchor = record(path("gungnip", "궁닙", "neural"))
    allowed = path("gungnip", "국립")
    value.boundary_trace = {
        "activated": True, "reason": "accepted_new_extras",
        "preserved_pool": [], "generated_new_extras": [record(allowed)],
        "accepted_new_extras": [record(allowed)],
        "augmented_piece_count": 1, "derived_options": [provenance()],
    }
    monkeypatch.setattr(
        BoundaryCompositionRuntime,
        "_numeric_sentence_extras",
        lambda *args: [],
    )
    with pytest.raises(ValueError, match="does not match"):
        value._numeric_sentence_extras([], [], [])
