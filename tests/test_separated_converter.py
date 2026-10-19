import threading

import pytest

from hanglish.core import ctc_word_proposals, mixed_span_proposals
from hanglish.core.ctc_union_converter import CTCUnionConverter
from hanglish.core.lattice import WordOption
from hanglish.core.separated_converter import SeparatedConverter


SOURCE = "aa bb"


def segments(text):
    left, right = text.split(" ")
    return [
        {"input_start": 0, "input_end": 2, "source": "aa",
         "text": left, "kind": "fixture"},
        {"input_start": 2, "input_end": 3, "source": " ",
         "text": " ", "kind": "raw"},
        {"input_start": 3, "input_end": 5, "source": "bb",
         "text": right, "kind": "fixture"},
    ]


def path(text, score):
    return (text, float(score), segments(text))


class Model:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate(self, sources):
        self.calls.append(list(sources))
        return [{"text": self.text, "reason": "complete",
                 "generated_tokens": 2}]


class Ranker:
    def __init__(self, *, rejected=(), oversized=False, invalid=False):
        self.rejected = set(rejected)
        self.oversized = oversized
        self.invalid = invalid
        self.calls = []

    def can_score(self, text):
        return not self.oversized and text not in self.rejected

    def score_pairs(self, contexts, candidates):
        self.calls.append((list(contexts), list(candidates)))
        if self.invalid and any(value.startswith("CTC") for value in candidates):
            return [float("nan")] * len(candidates)
        values = {"부모 하나": 3.0, "부모 둘": 2.0,
                  "MIX 신규": 7.0, "CTC 신규": 8.0,
                  "GROW 신규": 9.0, "거부 후보": 20.0}
        return [values[value] for value in candidates]


def converter(*, parent_paths=None, sentence_ranker=None):
    obj = object.__new__(SeparatedConverter)
    obj.lock = threading.RLock()
    obj.generation_mode = "independent"
    obj.context_ranking = True
    obj.ranker_kind = "sentence"
    obj.ranker = None
    obj.sentence_ranker = sentence_ranker
    obj.models_loaded = True
    obj.version = "fixture-version"
    obj.model_artifacts = {"parent": "frozen"}
    obj.numeric_structure = None
    obj.word_options = lambda *args: ()
    obj.word_model = Model("ctc output")
    obj._word_model_loaded = True
    obj.growth_model = Model("growth output")
    obj._growth_model_loaded = True
    obj.word_parent_version = "word-parent"
    obj.growth_parent_version = "growth-parent"
    obj.parent_version = "neural-parent"
    obj.implementation_artifacts = {"mixed": "frozen"}
    obj.ctc_word_artifacts = {"ctc": "frozen"}
    obj.growth_artifacts = {"growth": "frozen"}
    obj.mixed_span_counts = {"generated": 0, "accepted": 0}
    obj.ctc_word_counts = obj._empty_counts()
    obj.ctc_word_prediction = None
    obj.ctc_word_ms = 0.0
    obj.ctc_union_counts = obj._empty_growth_counts()
    obj.ctc_union_prediction = None
    obj.ctc_union_ms = 0.0
    obj.last_separated_capture = None
    obj.last_pipeline = None
    chosen = parent_paths or [path("부모 하나", 1), path("부모 둘", 0)]
    pieces = [
        ("aa", 0, (WordOption(chosen[0][0].split()[0], 1.0, "fixture"),)),
        (" ", 2, (WordOption(" ", 0.0, "raw"),)),
        ("bb", 3, tuple(WordOption(value[0].split()[1], value[1], "fixture")
                         for value in chosen)),
    ]
    def prepare(text, context, *, learned, raw):
        if text == SOURCE and not raw:
            return pieces, pieces, [], False, False
        raw_pieces = [(text, 0, (WordOption(text, 0.0, "raw"),))]
        return raw_pieces, raw_pieces, [], False, False
    obj._prepare_candidates = prepare
    obj.load_models = lambda: None
    return obj


def proposals(monkeypatch, *, rejected=False):
    monkeypatch.setattr(
        mixed_span_proposals, "sentence_extras",
        lambda source, paths, word_options, limit: (
            [path("MIX 신규", 999)] if source == SOURCE else []))

    def ctc(source, generated, paths, limit):
        if generated == "ctc output":
            rows = [path("CTC 신규", -999)]
            if rejected:
                rows.append(path("거부 후보", 10000))
            return rows
        return [path("GROW 신규", -10000)]

    monkeypatch.setattr(ctc_word_proposals, "sentence_extras", ctc)


def pool_by_text(response):
    return {row["text"]: row for row in response["pipeline"]["pool"]}


def test_legacy_mode_returns_parent_response_without_adapter_fields(monkeypatch):
    expected = {"result": "기존", "candidates": [{"text": "기존"}]}
    monkeypatch.setattr(CTCUnionConverter, "convert",
                        lambda self, *args, **kwargs: expected)
    obj = converter(sentence_ranker=Ranker())
    monkeypatch.setattr(obj, "_legacy_pipeline",
                        lambda *args, **kwargs: {"mode": "legacy"})
    assert obj.convert(SOURCE, generation_mode="legacy") is expected
    assert obj.last_pipeline == {"mode": "legacy"}
    assert obj.word_model.calls == obj.growth_model.calls == []


def test_real_legacy_chain_matches_independent_short_candidates_and_captures_pool(
        monkeypatch):
    proposals(monkeypatch)
    obj = converter(sentence_ranker=Ranker())
    legacy = obj.convert(SOURCE, generation_mode="legacy")
    legacy_pipeline = obj.last_pipeline
    independent = obj.convert(SOURCE, generation_mode="independent")
    assert "pipeline" not in legacy
    assert legacy["candidates"] == independent["candidates"]
    assert legacy["result"] == independent["result"]
    assert legacy_pipeline["mode"] == "legacy"
    assert len(legacy_pipeline["pool"]) == len(independent["pipeline"]["pool"])


@pytest.mark.parametrize("condition", ["oversized", "missing", "one_path"])
def test_independent_generation_runs_once_when_parent_sentence_ranking_is_gated(
        monkeypatch, condition):
    proposals(monkeypatch)
    ranker = None if condition == "missing" else Ranker(
        oversized=condition == "oversized")
    parents = ([path("부모 하나", 1)] if condition == "one_path"
               else [path("부모 하나", 1), path("부모 둘", 0)])
    obj = converter(parent_paths=parents, sentence_ranker=ranker)
    response = obj.convert(SOURCE, generation_mode="independent")
    assert obj.word_model.calls == [[SOURCE]]
    assert obj.growth_model.calls == [[SOURCE]]
    assert response["sentence_ranking_applied"] is False
    assert response["result"] == "부모 하나"
    pool = pool_by_text(response)
    assert {"MIX 신규", "CTC 신규", "GROW 신규"} <= set(pool)
    assert all(pool[text]["reason"] != "scored"
               for text in ("MIX 신규", "CTC 신규", "GROW 신규"))


def test_short_ranked_path_preserves_sequential_legacy_order_and_full_capture(
        monkeypatch):
    proposals(monkeypatch)
    obj = converter(sentence_ranker=Ranker())
    first = obj.convert(SOURCE, generation_mode="independent")
    second = obj.convert(SOURCE, generation_mode="independent")
    assert [row["text"] for row in first["candidates"][:5]] == [
        "GROW 신규", "CTC 신규", "MIX 신규", "부모 하나", SOURCE]
    assert obj.word_model.calls == [[SOURCE], [SOURCE]]
    assert obj.growth_model.calls == [[SOURCE], [SOURCE]]
    one = first["pipeline"]
    two = second["pipeline"]
    assert [row["candidate_id"] for row in one["pool"]] == [
        row["candidate_id"] for row in two["pool"]]
    assert all(row["source_coverage"] == {
        "complete": True, "input_start": 0, "input_end": len(SOURCE)}
               for row in one["pool"])
    assert [stage["name"] for stage in one["stages"][2:]] == [
        "mixed_span", "ctc_word", "ctc_growth"]
    assert [stage["generation_reason"] for stage in one["stages"][2:]] == [
        "complete", "complete", "complete"]
    by_text = {row["text"]: row for row in one["pool"]}
    assert by_text["MIX 신규"]["origin_score_type"] == (
        "inherited_sentence_ranker_score")
    assert by_text["CTC 신규"]["origin_score_type"] == "proposal_placeholder"
    assert by_text["GROW 신규"]["origin_score_type"] == "proposal_placeholder"


def test_raw_and_unsupported_sources_do_not_call_models(monkeypatch):
    proposals(monkeypatch)
    obj = converter(sentence_ranker=Ranker())
    raw = obj.convert("https://example.com x", raw=True,
                      generation_mode="independent")
    assert raw["result"] == "https://example.com x"
    assert obj.word_model.calls == obj.growth_model.calls == []
    assert all(stage["reason"] == "request_not_eligible"
               for stage in raw["pipeline"]["stages"][2:])

    unsupported = obj.convert("aa 한글", generation_mode="independent")
    assert obj.word_model.calls == obj.growth_model.calls == []
    reasons = {stage["name"]: stage["generation_reason"]
               for stage in unsupported["pipeline"]["stages"]
               if stage["name"].startswith("ctc_")}
    assert reasons["ctc_word"] == reasons["ctc_growth"] == "unsupported_source"


def test_unsupported_source_does_not_even_load_ctc_models(monkeypatch):
    proposals(monkeypatch)
    obj = converter(sentence_ranker=Ranker())
    obj.load_word_model = lambda: pytest.fail("unsupported source loaded CTC1")
    obj.load_growth_model = lambda: pytest.fail("unsupported source loaded growth CTC")
    response = obj.convert("aa 한글", generation_mode="independent")
    assert response["result"] == "aa 한글"


def test_same_text_with_different_paths_is_retained_and_response_maps_selected_path(
        monkeypatch):
    monkeypatch.setattr(mixed_span_proposals, "sentence_extras",
                        lambda *args, **kwargs: [])
    alternate_segments = [
        {"input_start": 0, "input_end": 2, "source": "aa",
         "text": "부", "kind": "alternate"},
        {"input_start": 2, "input_end": 3, "source": " ",
         "text": "모 ", "kind": "alternate"},
        {"input_start": 3, "input_end": 5, "source": "bb",
         "text": "하나", "kind": "alternate"},
    ]
    alternate = ("부모 하나", 50.0, alternate_segments)
    monkeypatch.setattr(
        ctc_word_proposals, "sentence_extras",
        lambda source, generated, paths, limit: (
            [alternate] if generated == "ctc output" else []))
    obj = converter(sentence_ranker=Ranker())
    response = obj.convert(SOURCE, generation_mode="independent")
    same_text = [row for row in response["pipeline"]["pool"]
                 if row["text"] == "부모 하나"]
    assert len(same_text) == 2
    assert len({row["path_id"] for row in same_text}) == 2
    alternate_row = next(row for row in same_text
                         if row["origin"] == "ctc_word")
    assert alternate_row["reason"] == "duplicate_output_not_selected"
    response_ref = next(row for row in response["pipeline"]["response_candidates"]
                        if row["candidate"]["text"] == "부모 하나")
    parent_row = next(row for row in same_text if row["origin"] == "parent")
    assert response_ref["candidate_id"] == parent_row["candidate_id"]


def test_invalid_sentence_ranker_score_is_rejected(monkeypatch):
    proposals(monkeypatch)
    obj = converter(sentence_ranker=Ranker(invalid=True))
    with pytest.raises(RuntimeError, match="invalid scores"):
        obj.convert(SOURCE, generation_mode="independent")


def test_unscoreable_proposal_remains_in_pool_without_affecting_selection(
        monkeypatch):
    proposals(monkeypatch, rejected=True)
    obj = converter(sentence_ranker=Ranker(rejected={"거부 후보"}))
    response = obj.convert(SOURCE, generation_mode="independent")
    rejected = pool_by_text(response)["거부 후보"]
    assert rejected["reason"] == "unsupported_by_ranker"
    assert rejected["score"] is None
    assert rejected["score_type"] is None
    assert rejected["selection_eligible"] is False
    assert "거부 후보" not in [row["text"] for row in response["candidates"]]


def test_unranked_local_and_generator_scores_are_never_globally_sorted(
        monkeypatch):
    proposals(monkeypatch)
    obj = converter(sentence_ranker=None)
    response = obj.convert(SOURCE, generation_mode="independent")
    assert response["result"] == "부모 하나"
    pool = response["pipeline"]["pool"]
    assert [row["text"] for row in pool] == [
        "부모 하나", "부모 둘", "MIX 신규", "CTC 신규", "GROW 신규", SOURCE]
    assert [row["score_type"] for row in pool[:2]] == [
        "local_path_score", "local_path_score"]
    assert all(row["score"] is None and row["score_type"] is None
               for row in pool[2:])
    assert response["pipeline"]["selection_policy"] == (
        "retain_parent_order_capture_unscored")
