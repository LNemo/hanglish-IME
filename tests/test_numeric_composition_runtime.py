import threading

import pytest

from hanglish.core.lattice import WordOption
from hanglish.core.neural_converter import NeuralConverter
from hanglish.core.numeric_composition import CompositionSupplement
import scripts.numeric_composition_runtime as module


class Generator:
    def __init__(self, proposals=True):
        self.proposals = proposals

    def options(self, source):
        raw = WordOption(source, 0.0, "raw")
        if not self.proposals:
            return (raw,)
        if source == "wol":
            return (WordOption("월", 6.0, "lexicon"), raw)
        if source == "gaji":
            return (WordOption("가지", 6.0, "lexicon"), raw)
        return (raw,)


class Ranker:
    def __init__(self, unscoreable=()):
        self.unscoreable = set(unscoreable)
        self.calls = []

    def can_score(self, candidate):
        return candidate not in self.unscoreable

    def score_pairs(self, contexts, candidates):
        self.calls.append((list(contexts), list(candidates)))
        return [10.0 if candidate == "11월가지" else 0.0 for candidate in candidates]


def raw_piece(source="11wolgaji"):
    return [(source, 0, (WordOption(source, 0.0, "raw"),))]


def converter(*, proposals=True, enabled=True):
    value = module.NumericCompositionRuntime.__new__(module.NumericCompositionRuntime)
    value.composition_enabled = enabled
    value.numeric_units = ("월",)
    value.numeric_structure = None
    value.generator = Generator(proposals)
    value.lock = threading.RLock()
    value.sentence_ranker = Ranker()
    value.ranker_kind = "hybrid"
    value.context_ranking = True
    value.version = "test"
    value.model_artifacts = {}
    value.load_models = lambda: None
    value._numeric_sentence_extras = lambda pieces, static, complete: []

    def prepare(text, context, *, learned, raw):
        pieces = raw_piece(text)
        return pieces, pieces, [], False, False

    value._prepare_candidates = prepare
    return value


def path(text, score=0.0):
    return (
        text,
        score,
        [{
            "input_start": 0,
            "input_end": len(text),
            "source": text,
            "text": text,
            "kind": "raw",
        }],
    )


def test_composition_disabled_is_exact_production_baseline(monkeypatch):
    monkeypatch.setattr(module.time, "perf_counter", lambda: 1.0)
    value = converter(enabled=False)
    expected = NeuralConverter.convert(value, "11wolgaji")
    actual = value.convert("11wolgaji")
    assert actual == expected
    assert value.sentence_ranker.calls == []


def test_disabled_path_removes_stale_experiment_trace(monkeypatch):
    monkeypatch.setattr(module.time, "perf_counter", lambda: 1.0)
    value = converter()
    value.convert("11wolgaji")
    assert hasattr(value, "numeric_composition_trace")
    value.composition_enabled = False
    value.convert("11wolgaji")
    assert not hasattr(value, "numeric_composition_trace")


def test_no_accepted_choice_preserves_result_and_flags(monkeypatch):
    monkeypatch.setattr(module.time, "perf_counter", lambda: 1.0)
    value = converter(proposals=False)
    expected = NeuralConverter.convert(value, "11wolgaji")
    actual = value.convert("11wolgaji")
    assert actual == expected
    assert actual["ranking_applied"] is False
    assert actual["sentence_ranking_applied"] is False
    assert value.numeric_composition_trace["reason"] == "no_accepted_choice"
    assert value.sentence_ranker.calls == []


def test_hybrid_single_piece_numeric_choice_activates_one_scoring_call():
    value = converter()
    response = value.convert("11wolgaji", context="daeum dal")
    assert response["result"] == "11월가지"
    assert response["ranking_applied"] is True
    assert response["sentence_ranking_applied"] is True
    assert response["context_used"] is True
    assert len(value.sentence_ranker.calls) == 1
    assert value.sentence_ranker.calls[0][1] == ["11wolgaji", "11월가지"]
    trace = value.numeric_composition_trace
    assert trace["activated"] is True
    assert trace["production_activated"] is False
    assert [item["text"] for item in trace["base"]] == ["11wolgaji"]
    assert [item["text"] for item in trace["preserved_pool"]] == [
        "11wolgaji"
    ]
    assert trace["production_scoring_pool"] == []
    assert [item["text"] for item in trace["accepted_new_extras"]] == [
        "11월가지"
    ]


def test_oversized_new_extra_does_not_suppress_accepted_old(monkeypatch):
    value = converter()
    old = path("11월")
    valid_new = path("11월가지")
    oversized = path("너무긴후보")
    value._numeric_sentence_extras = lambda pieces, static, complete: [old]
    value.sentence_ranker = Ranker(unscoreable={"너무긴후보"})
    monkeypatch.setattr(
        module,
        "composition_extras",
        lambda *args, **kwargs: CompositionSupplement(
            (oversized, valid_new), 1, 2, False
        ),
    )
    pieces = raw_piece()
    pool = module.build_runtime_pool(
        value, pieces, pieces, value.sentence_ranker
    )
    assert pool.old_extras == (old,)
    assert pool.generated_new_extras == (oversized, valid_new)
    assert pool.new_extras == (valid_new,)
    assert pool.complete == (path("11wolgaji"), old, valid_new)


def test_old_extra_acceptance_remains_all_or_nothing(monkeypatch):
    value = converter()
    accepted = path("11월")
    rejected = path("11월초과")
    value._numeric_sentence_extras = lambda pieces, static, complete: [
        accepted, rejected
    ]
    value.sentence_ranker = Ranker(unscoreable={"11월초과"})
    monkeypatch.setattr(
        module,
        "composition_extras",
        lambda *args, **kwargs: CompositionSupplement((), 0, 0, False),
    )
    pieces = raw_piece()
    pool = module.build_runtime_pool(
        value, pieces, pieces, value.sentence_ranker
    )
    assert pool.old_extras == ()
    assert pool.complete == pool.base


def test_unscoreable_base_bypasses_all_supplement_generation():
    value = converter()
    value.sentence_ranker = Ranker(unscoreable={"11wolgaji"})
    called = False

    def old_extras(*args):
        nonlocal called
        called = True
        return []

    value._numeric_sentence_extras = old_extras
    response = value.convert("11wolgaji")
    assert response["sentence_ranking_applied"] is False
    assert called is False
    assert value.numeric_composition_trace["reason"] == "base_unscoreable"
    assert value.numeric_composition_trace["generated_new_extras"] == []


def test_unscoreable_singleton_preserves_full_production_response(monkeypatch):
    monkeypatch.setattr(module.time, "perf_counter", lambda: 1.0)
    value = converter()
    value.sentence_ranker = Ranker(unscoreable={"11wolgaji"})
    expected = NeuralConverter.convert(value, "11wolgaji")
    actual = value.convert("11wolgaji")
    assert actual == expected
    assert actual["warnings"] == []
    assert value.numeric_composition_trace["reason"] == "base_unscoreable"


def test_raw_branch_is_unchanged_and_does_not_generate_compositions(monkeypatch):
    monkeypatch.setattr(module.time, "perf_counter", lambda: 1.0)
    value = converter()
    expected = NeuralConverter.convert(value, "11wolgaji", raw=True)
    actual = value.convert("11wolgaji", raw=True)
    assert actual == expected
    assert value.numeric_composition_trace["reason"] == "production_branch_bypass"


def test_runtime_builder_has_no_label_or_expected_parameter():
    names = module.build_runtime_pool.__code__.co_varnames
    assert "expected" not in names
    assert "label" not in names
    assert "target" not in names
