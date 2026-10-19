from copy import deepcopy
from types import SimpleNamespace
import threading

import pytest

from hanglish.core.separated_converter import SeparatedConverter
from hanglish.core.extended_separated_converter import ExtendedSeparatedConverter


SOURCE = "aa bb"


def segment(text, kind="fixture"):
    return [{"input_start": 0, "input_end": len(SOURCE), "source": SOURCE,
             "text": text, "kind": kind}]


def candidate(text, index, *, kind="fixture", score=None):
    segments = segment(text, kind)
    path_id = SeparatedConverter._path_identity(
        SOURCE, (text, 0.0, segments))
    return {
        "candidate_id": path_id, "path_id": path_id,
        "pool_index": index, "text": text, "segments": segments,
        "source_spans": [{"input_start": 0, "input_end": len(SOURCE),
                          "source": SOURCE, "text": text, "kind": kind,
                          "action": "copy" if text == SOURCE else "convert",
                          "copy_exact": True if text == SOURCE else None}],
        "source_coverage": {"complete": True, "input_start": 0,
                            "input_end": len(SOURCE)},
        "origin": "parent", "origins": ["parent"],
        "origin_score": score, "origin_score_type": "local_path_score",
        "origin_score_model_id": None, "score": score,
        "score_type": "local_path_score" if score is not None else None,
        "model_id": None, "evaluated": False,
        "reason": "unsupported_by_ranker", "selection_eligible": True,
    }


def raw_candidate(index=99):
    value = candidate(SOURCE, index, kind="raw", score=None)
    value.update({"origin": "raw_fallback", "origins": ["raw_fallback"],
                  "reason": "appended_for_user"})
    return value


def raw_response_candidate():
    return {"text": SOURCE, "score": None, "segments": segment(SOURCE, "raw")}


def base_response(*, ranked=False, pool=None, engine="neural", raw=False,
                  context=""):
    first = {"text": "첫 후보", "score": 1.0,
             "segments": segment("첫 후보")}
    pool = list(pool or [candidate("첫 후보", 0, score=1.0),
                         candidate("둘 후보", 1, score=0.0), raw_candidate()])
    response = {
        "input": SOURCE, "context": context, "engine": engine,
        "mode": "batch", "raw": raw, "version": "extended-fixture",
        "candidates": [first, raw_response_candidate()], "result": "첫 후보",
        "ranker": "sentence", "model_artifacts": {"ranker": "frozen"},
        "context_used": False, "external_context_used": False,
        "ranking_applied": True, "sentence_ranking_applied": ranked,
        "warnings": [], "engine_ms": 1.0,
        "pipeline": {
            "schema_version": "separated-candidate-pipeline-v1",
            "identity": {"experimental": True}, "mode": "independent",
            "request_id": "request", "request": {"input": SOURCE,
                "context": context, "engine": engine, "mode": "batch", "raw": raw},
            "source_only_generation": True, "source_truncated": False,
            "selection_policy": "retain_parent_order_capture_unscored",
            "parent": {"sentence_ranking_applied": ranked,
                       "sentence_ranking_reason": "scored" if ranked else "unsupported_by_ranker",
                       "path_ids": [row["path_id"] for row in pool]},
            "stages": [], "pool": deepcopy(pool), "response_candidates": [],
        },
        "mixed_span": {"generated": 0, "accepted": 0},
        "ctc_word": {"generation_requests": 1},
        "ctc_union": {"generation_requests": 1},
    }
    return response


class ExtendedRanker:
    def __init__(self, scores=None, unsupported=()):
        self.scores = scores or {}
        self.unsupported = set(unsupported)
        self.calls = []
        self.max_length = 256
        self.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))
        self.tokenizer = object()

    def can_score(self, text):
        return text not in self.unsupported

    def score_pairs(self, contexts, candidates):
        self.calls.append((list(contexts), list(candidates)))
        return [self.scores.get(text, float(index))
                for index, text in enumerate(candidates)]


def converter(ranker=None):
    obj = object.__new__(ExtendedSeparatedConverter)
    obj.lock = threading.RLock()
    obj.context_ranking = True
    obj.ranker_kind = "sentence"
    obj.sentence_ranker = SimpleNamespace(
        max_length=96,
        batch_size=8,
        metadata={"max_length": 96},
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512)),
        tokenizer=object(),
    )
    extended = ranker or ExtendedRanker()
    extended.model = obj.sentence_ranker.model
    extended.tokenizer = obj.sentence_ranker.tokenizer
    obj._extended_sentence_ranker = extended
    obj._extended_source_ranker = obj.sentence_ranker
    obj._extended_parent_version = "parent-version"
    obj.extended_artifacts = {"hanglish/core/extended_separated_converter.py": "hash"}
    obj.version = "experimental-extended-version"
    obj.last_pipeline = None
    return obj


def install_base(monkeypatch, response):
    calls = []

    def convert(self, text, context="", engine="neural", mode="batch", raw=False,
                *, generation_mode=None):
        calls.append({"text": text, "context": context, "engine": engine,
                      "mode": mode, "raw": raw,
                      "generation_mode": generation_mode})
        value = deepcopy(response)
        value.update({"input": text, "context": context, "engine": engine,
                      "mode": mode, "raw": raw})
        value["pipeline"]["request"].update(
            {"input": text, "context": context, "engine": engine,
             "mode": mode, "raw": raw})
        return value

    monkeypatch.setattr(SeparatedConverter, "convert", convert)
    return calls


def test_expanded_pool_256_scores_first_structural_path_with_empty_context(
        monkeypatch):
    pool = [candidate("같은", 0, kind="first", score=100.0),
            candidate("같은", 1, kind="duplicate", score=-100.0),
            candidate("둘", 2), candidate("셋", 3), candidate("넷", 4),
            candidate("다섯", 5), raw_candidate()]
    response = base_response(pool=pool)
    response["warnings"] = [
        "문장 모델의 길이 범위를 넘어 단어별 문맥 순위를 사용했습니다."]
    ranker = ExtendedRanker({"같은": 1.0, "둘": 5.0, "셋": 4.0,
                             "넷": 3.0, "다섯": 2.0})
    obj = converter(ranker)
    calls = install_base(monkeypatch, response)
    result = obj.convert(SOURCE)

    assert calls == [{"text": SOURCE, "context": "", "engine": "neural",
                      "mode": "batch", "raw": False,
                      "generation_mode": "independent"}]
    assert ranker.calls == [(["", "", "", "", ""],
                             ["같은", "둘", "셋", "넷", "다섯"])]
    assert [row["text"] for row in result["candidates"]] == [
        "둘", "셋", "넷", "다섯", SOURCE]
    assert result["candidates"][-1] == raw_response_candidate()
    assert result["result"] == "둘"
    assert result["sentence_ranking_applied"] is True
    assert result["ranking_applied"] is True
    assert result["warnings"] == []
    metadata = result["extended_sentence_ranking"]
    assert metadata["policy"] == "expanded_pool_256"
    assert metadata["applied"] is True
    assert metadata["score_context"] == ""
    assert metadata["input_truncation"] is False
    assert metadata["unique_nonraw_candidates"] == 5
    assert result["pipeline"]["selection_policy"] == "expanded_pool_256"
    assert result["pipeline"]["stages"][-1]["reason"] == (
        "scored_full_nonraw_pool_at_256")
    duplicate = result["pipeline"]["pool"][1]
    assert duplicate["extended_ranking"]["reason"] == (
        "duplicate_text_after_first_structural_path")
    refs = result["pipeline"]["response_candidates"]
    assert [row["candidate"] for row in refs] == result["candidates"]
    assert obj.last_pipeline == result["pipeline"]


def test_any_unsupported_candidate_preserves_independent_selection_wholly(
        monkeypatch):
    response = base_response(pool=[candidate("짧음", 0), candidate("너무 김", 1),
                                   raw_candidate()])
    before = deepcopy(response)
    ranker = ExtendedRanker(unsupported={"너무 김"})
    obj = converter(ranker)
    install_base(monkeypatch, response)
    result = obj.convert(SOURCE)
    assert ranker.calls == []
    for field in ("result", "candidates", "sentence_ranking_applied",
                  "ranking_applied"):
        assert result[field] == before[field]
    assert result["extended_sentence_ranking"]["applied"] is False
    assert result["extended_sentence_ranking"]["reason"] == "candidate_exceeds_256"
    assert result["extended_sentence_ranking"]["unsupported_texts"] == ["너무 김"]


@pytest.mark.parametrize("case,kwargs,mutate,reason", [
    ("ranked", {}, lambda value, obj: (
        value.update(sentence_ranking_applied=True),
        value["pipeline"]["parent"].update(sentence_ranking_applied=True)),
     "parent_sentence_ranking_already_applied"),
    ("raw", {"raw": True}, lambda value, obj: None, "raw_request"),
    ("rules", {"engine": "morphology"}, lambda value, obj: None,
     "non_neural_engine"),
    ("context", {"context": "외부 문맥"}, lambda value, obj: None,
     "nonempty_context_not_evaluated"),
    ("no-ranker", {}, lambda value, obj: setattr(obj, "sentence_ranker", None),
     "sentence_ranker_unavailable"),
    ("disabled", {}, lambda value, obj: setattr(obj, "context_ranking", False),
     "context_ranking_disabled"),
])
def test_ineligible_modes_preserve_parent_candidates_without_scoring(
        monkeypatch, case, kwargs, mutate, reason):
    response = base_response()
    obj = converter(ExtendedRanker({"첫 후보": 99.0, "둘 후보": 98.0}))
    mutate(response, obj)
    before = deepcopy(response["candidates"])
    install_base(monkeypatch, response)
    result = obj.convert(SOURCE, **kwargs)
    assert result["candidates"] == before
    assert result["result"] == response["result"]
    assert result["extended_sentence_ranking"]["reason"] == reason
    assert result["extended_sentence_ranking"]["applied"] is False
    assert obj._extended_sentence_ranker.calls == []


def test_single_word_is_outside_frozen_multiword_scope(monkeypatch):
    response = base_response()
    obj = converter(ExtendedRanker({"첫 후보": 9.0, "둘 후보": 8.0}))
    install_base(monkeypatch, response)
    result = obj.convert("imnida")
    assert result["extended_sentence_ranking"]["reason"] == (
        "single_word_not_evaluated")
    assert result["result"] == response["result"]
    assert obj._extended_sentence_ranker.calls == []


@pytest.mark.parametrize("text,reason", [
    ("", "single_word_not_evaluated"),
    ("   ", "single_word_not_evaluated"),
    ("aa 한글", "non_roman_source_not_evaluated"),
])
def test_blank_whitespace_and_hangul_sources_preserve_parent_without_pool_validation(
        monkeypatch, text, reason):
    response = base_response()
    obj = converter(ExtendedRanker())
    install_base(monkeypatch, response)
    result = obj.convert(text)
    assert result["extended_sentence_ranking"]["reason"] == reason
    assert result["extended_sentence_ranking"]["applied"] is False
    assert obj._extended_sentence_ranker.calls == []


def test_real_rules_path_preserves_empty_and_whitespace_without_loading_models():
    obj = ExtendedSeparatedConverter(external_lexicon=False)
    obj.load_models = lambda: pytest.fail("rules-only blank input loaded models")
    for source in ("", "   "):
        response = obj.convert(source, engine="morphology")
        assert response["result"] == source
        assert response["extended_sentence_ranking"]["reason"] == (
            "non_neural_engine")
        assert response["extended_sentence_ranking"]["identity"][
            "shared_original_model_and_tokenizer"] is False


@pytest.mark.parametrize("scores", [[float("nan"), 1.0], [True, 1.0], [1.0]])
def test_invalid_or_incomplete_extended_scores_are_rejected(monkeypatch, scores):
    response = base_response()
    ranker = ExtendedRanker()
    ranker.score_pairs = lambda contexts, candidates: scores
    obj = converter(ranker)
    install_base(monkeypatch, response)
    with pytest.raises(RuntimeError, match="invalid scores"):
        obj.convert(SOURCE)


def test_ranker_wrapper_requires_original_96_and_model_capacity_256(monkeypatch):
    obj = converter()
    obj._extended_sentence_ranker = None
    obj._extended_source_ranker = None

    class SharedRanker:
        def __init__(self, model, tokenizer, *, max_length, batch_size, metadata):
            self.model = model; self.tokenizer = tokenizer
            self.max_length = max_length; self.batch_size = batch_size
            self.metadata = metadata

    monkeypatch.setattr("hanglish.models.sentence_ranker.SentenceRanker", SharedRanker)
    extended = obj._load_extended_ranker()
    assert extended.model is obj.sentence_ranker.model
    assert extended.tokenizer is obj.sentence_ranker.tokenizer
    assert extended.max_length == 256
    assert obj._load_extended_ranker() is extended

    obj._extended_sentence_ranker = None
    obj._extended_source_ranker = None
    obj.sentence_ranker.model.config.max_position_embeddings = 128
    with pytest.raises(RuntimeError, match="256 positions"):
        obj._load_extended_ranker()


def test_repeated_model_load_keeps_extended_version_and_artifacts(monkeypatch):
    obj = object.__new__(ExtendedSeparatedConverter)
    obj.lock = threading.RLock()
    obj.version = "initial-parent"
    obj.models_loaded = False
    obj.model_artifacts = {}
    obj.extended_artifacts = {"extended": "hash", "separated": "hash"}
    obj._extended_parent_version = "initial-parent"

    def parent_load(self):
        self.models_loaded = True
        self.version = "stable-parent"
        self.model_artifacts = {"parent": "hash"}

    monkeypatch.setattr(SeparatedConverter, "load_models", parent_load)
    obj.load_models(); first = obj.version
    obj.load_models()
    assert obj.version == first
    assert obj._extended_parent_version == "stable-parent"
    assert obj.model_artifacts == {
        "parent": "hash", "extended": "hash", "separated": "hash"}
