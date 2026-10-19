import copy
from types import SimpleNamespace

import pytest

import hanglish.core.neural_converter as neural_module
from hanglish.core.lattice import WordOption
from scripts.numeric_structure_candidates import NumericSpanGenerator
import scripts.probe_numeric_structure_runtime as module


class Scoreable:
    def __init__(self, reject_numeric=False):
        self.reject_numeric = reject_numeric

    def can_score(self, text):
        return not self.reject_numeric or "월" not in text


def pieces():
    learned = [
        ("11wol", 0, (WordOption("11wol", 0.0, "raw"),)),
        (" ", 5, (WordOption(" ", 0.0, "raw"),)),
        ("x", 6, (WordOption("엑스", 2.0, "neural"), WordOption("x", -10000.0, "raw"))),
    ]
    static = [
        ("11wol", 0, (WordOption("11wol", 0.0, "raw"),)),
        (" ", 5, (WordOption(" ", 0.0, "raw"),)),
        ("x", 6, (WordOption("x", 0.0, "raw"),)),
    ]
    return learned, static


def test_adapter_preserves_parent_beams_and_appends_first_numeric_assignment():
    learned, static = pieces()
    converter = SimpleNamespace(sentence_ranker=Scoreable())
    original = neural_module.beam_paths
    with module.numeric_beam_overlay(converter, NumericSpanGenerator()) as state:
        learned_paths = neural_module.beam_paths(learned)
        static_paths = neural_module.beam_paths(static)
    assert neural_module.beam_paths is original
    assert static_paths[:len(original(static))] == original(static)
    assert state["calls"] == 2 and state["applied"] is True
    assert [path[0] for path in state["original_pool"]] == [
        path[0] for path in module.stable_union(original(learned), original(static))
    ]
    assert [path[0] for path in state["extras"]] == ["11월 엑스", "11월 x"]
    assert [path[1] for path in state["extras"]] == [
        path[1] for path in state["original_pool"]
    ]
    assert all("".join(segment["source"] for segment in path[2][:2]) == "11wol"
               and path[2][1]["kind"] == "numeric" for path in state["extras"])


def test_unscoreable_extra_keeps_exact_baseline_static_return():
    learned, static = pieces()
    converter = SimpleNamespace(sentence_ranker=Scoreable(reject_numeric=True))
    original = neural_module.beam_paths
    with module.numeric_beam_overlay(converter, NumericSpanGenerator()) as state:
        neural_module.beam_paths(learned)
        returned = neural_module.beam_paths(static)
    assert returned == original(static)
    assert state["applied"] is False
    assert state["reason"] == "numeric_extra_unscoreable"


def test_adapter_rejects_third_call_and_restores_hook():
    learned, static = pieces()
    original = neural_module.beam_paths
    with pytest.raises(RuntimeError, match="more than two"):
        with module.numeric_beam_overlay(
                SimpleNamespace(sentence_ranker=Scoreable()), NumericSpanGenerator()):
            neural_module.beam_paths(learned)
            neural_module.beam_paths(static)
            neural_module.beam_paths(static)
    assert neural_module.beam_paths is original


def test_adapter_restores_hook_when_original_beam_raises(monkeypatch):
    def broken(_pieces):
        raise LookupError("boom")

    monkeypatch.setattr(module, "ORIGINAL_BEAM_PATHS", broken)
    monkeypatch.setattr(neural_module, "beam_paths", broken)
    with pytest.raises(LookupError, match="boom"):
        with module.numeric_beam_overlay(
                SimpleNamespace(sentence_ranker=Scoreable()), NumericSpanGenerator()):
            neural_module.beam_paths([])
    assert neural_module.beam_paths is broken


def test_nested_or_foreign_hook_is_rejected(monkeypatch):
    monkeypatch.setattr(neural_module, "beam_paths", lambda pieces: [])
    with pytest.raises(RuntimeError, match="changed before"):
        with module.numeric_beam_overlay(
                SimpleNamespace(sentence_ranker=Scoreable()), NumericSpanGenerator()):
            pass


def test_approved_train_gate_accepts_only_exact_pinned_artifacts():
    report, cases, integration, _ = module.read_approved(module.APPROVED_REPORT_SHA256)
    assert len(cases) == 512
    bad = copy.deepcopy(report)
    bad["overlay_coverage"] -= 1
    with pytest.raises(ValueError, match="lineage"):
        module.validate_approved(bad, cases, integration)
    with pytest.raises(ValueError, match="explicit"):
        module.read_approved("0" * 64)


def test_metrics_reports_each_regression_axis():
    rows = [{
        "id": "one", "parent_top1": True, "child_top1": False,
        "parent_top5": True, "child_top5": False,
        "parent_pool_covered": True, "child_pool_covered": False,
    }]
    result = module.metrics(rows)
    assert result["regressions"] == ["one"]
    assert result["top5_regressions"] == ["one"]
    assert result["pool_regressions"] == ["one"]
    assert module.runtime_passed({
        **result, "rows": 512,
        "parent_pool_or_fallback_top5_covered": 319,
        "child_pool_or_fallback_top5_covered": 339,
    }) is False


def test_pool_prefix_proof_checks_text_order_and_score_tolerance():
    assert module.pool_prefix_proof(["a", "b"], [1.0, 2.0],
                                    ["a", "b", "c"], [1.00001, 2.0, 3.0])
    assert not module.pool_prefix_proof(["a", "b"], [1.0, 2.0],
                                        ["b", "a", "c"], [1.0, 2.0, 3.0])
    assert not module.pool_prefix_proof(["a"], [1.0], ["a", "c"], [1.01, 3.0])
    assert not module.pool_prefix_proof(["a", "b"], [1.0],
                                        ["a", "b", "c"], [1.0, 2.0, 3.0])
    assert not module.pool_prefix_proof(["a"], [1.0],
                                        ["a", "c"], [1.0])
    assert module.pool_prefix_proof(None, None, None, None)


def test_runtime_gate_requires_gain_and_zero_row_regressions():
    summary = {
        "rows": 512, "parent_top1": 300, "child_top1": 301,
        "parent_top5": 319, "child_top5": 339,
        "parent_pool_or_fallback_top5_covered": 319,
        "child_pool_or_fallback_top5_covered": 339,
        "regressions": [], "top5_regressions": [], "pool_regressions": [],
    }
    assert module.runtime_passed(summary)
    assert not module.runtime_passed({**summary, "regressions": ["offset-loss"]})


def test_actual_shaped_raw_candidate_with_none_score_is_valid():
    source = "11wol"
    response = {"version": module.RUNTIME_VERSION, "result": source, "candidates": [{
        "text": source, "score": None,
        "segments": [{"input_start": 0, "input_end": len(source),
                      "source": source, "text": source, "kind": "raw"}],
    }]}
    module.validate_candidate_objects(response, source)
    bad = copy.deepcopy(response)
    bad["candidates"][0]["segments"][0]["input_end"] -= 1
    with pytest.raises(ValueError, match="segments"):
        module.validate_candidate_objects(bad, source)


def test_source_only_builder_does_not_accept_expected_labels():
    assert "expected" not in module.build_extras.__code__.co_varnames
    assert "expected" not in module.numeric_beam_overlay.__code__.co_varnames
