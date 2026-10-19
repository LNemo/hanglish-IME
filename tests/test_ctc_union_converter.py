import threading

import pytest

from hanglish.core import ctc_word_proposals
from hanglish.core.ctc_union_converter import (
    CTCUnionConverter, EXPECTED_GROWTH_CHECKPOINT_SHA256,
    EXPECTED_GROWTH_METADATA_SHA256,
)
from hanglish.core.ctc_word_converter import CTCWordConverter


def parent_paths(source="ga na da ra"):
    values = ("가 나 다 라", "가 나 타 라", "카 나 다 라", "가 마 다 라")
    result = []
    for score, value in enumerate(values[::-1], 1):
        cursor = 0
        segments = []
        for index, (chunk, output) in enumerate(zip(source.split(), value.split())):
            if index:
                segments.append({"input_start": cursor, "input_end": cursor + 1,
                                 "source": " ", "text": " ", "kind": "raw"})
                cursor += 1
            segments.append({"input_start": cursor, "input_end": cursor + len(chunk),
                             "source": chunk, "text": output, "kind": "fixture"})
            cursor += len(chunk)
        result.append((value, float(score), segments))
    return sorted(result, key=lambda item: (-item[1], item[0]))


class Model:
    def __init__(self, prediction):
        self.prediction = prediction
        self.calls = []

    def generate(self, sources):
        self.calls.append(list(sources))
        return [dict(self.prediction)]


class Ranker:
    def __init__(self, scores=None):
        self.scores = scores
        self.calls = []

    def can_score(self, text):
        return "거부" not in text

    def score_pairs(self, contexts, candidates):
        self.calls.append((list(contexts), list(candidates)))
        return list(self.scores) if self.scores is not None else [9.0] * len(candidates)


def prediction(**values):
    result = {"text": "가 나 다 라", "reason": "complete", "generated_tokens": 4,
              "labels": [1, 2, 3, 4], "frames": 30,
              "decode_mode": "ctc_prefix_beam", "log_probability": -1.0,
              "beam": {}, "greedy": {}}
    result.update(values)
    return result


def converter(response=None):
    obj = object.__new__(CTCUnionConverter)
    obj.lock = threading.RLock()
    obj.growth_model = Model(response or prediction())
    obj._growth_model_loaded = True
    obj.sentence_ranker = Ranker()
    obj.ctc_union_counts = obj._empty_growth_counts()
    obj.ctc_union_prediction = None
    obj.ctc_union_ms = 0.0
    return obj


def test_growth_runs_after_complete_parent_pool_once_and_preserves_scores(monkeypatch):
    parent = parent_paths()
    parent.append(("부모 CTC1", 8.0, parent[0][2]))
    calls = []
    def parent_extend(self, text, context, paths):
        calls.append((text, context, paths))
        return parent
    monkeypatch.setattr(CTCWordConverter, "_extend_sentence_paths", parent_extend)
    proposed = [("성장 신규", 0.0, parent[0][2])]
    observed = []
    def extras(source, generated, paths, limit):
        observed.append((source, generated, paths, limit))
        return proposed
    monkeypatch.setattr(ctc_word_proposals, "sentence_extras", extras)
    obj = converter()
    initial = parent_paths()
    result = obj._extend_sentence_paths("ga na da ra", "문맥", initial)
    assert calls == [("ga na da ra", "문맥", initial)]
    assert obj.growth_model.calls == [["ga na da ra"]]
    assert observed == [("ga na da ra", "가 나 다 라", parent, 96)]
    assert obj.sentence_ranker.calls == [(["문맥"], ["성장 신규"])]
    assert result[0][0] == "성장 신규"
    assert all(path in result for path in parent)
    assert {path[0]: path[1] for path in result if path[0] != "성장 신규"} == {
        path[0]: path[1] for path in parent}
    assert obj.ctc_union_counts == {
        "generation_requests": 1, "completed_generations": 1,
        "generated_proposals": 1, "accepted_proposals": 1}


@pytest.mark.parametrize("source,reason", [
    ("singleword", "unsupported_source"),
    ("ga 한글", "unsupported_source"),
    ("a " + "b" * 509, "source_token_limit"),
])
def test_unsupported_growth_preserves_parent_result(monkeypatch, source, reason):
    parent = parent_paths()
    calls = []
    monkeypatch.setattr(CTCWordConverter, "_extend_sentence_paths",
                        lambda self, *args: calls.append(args) or parent)
    obj = converter()
    assert obj._extend_sentence_paths(source, "", parent_paths()) is parent
    assert len(calls) == 1 and obj.growth_model.calls == []
    assert obj.ctc_union_prediction == {
        "text": None, "reason": reason, "generated_tokens": 0}


@pytest.mark.parametrize("reason", ["empty_output", "invalid_utf8", "invalid_jamo",
                                     "no_valid_path"])
def test_incomplete_growth_preserves_exact_parent(monkeypatch, reason):
    parent = parent_paths()
    monkeypatch.setattr(CTCWordConverter, "_extend_sentence_paths",
                        lambda self, *args: parent)
    obj = converter(prediction(text=None, reason=reason, generated_tokens=0,
                               labels=[], log_probability=None))
    assert obj._extend_sentence_paths("ga na da ra", "", parent_paths()) is parent
    assert obj.ctc_union_counts["generation_requests"] == 1
    assert obj.ctc_union_counts["generated_proposals"] == 0


def test_unscoreable_growth_filtered_and_invalid_scores_rejected(monkeypatch):
    parent = parent_paths()
    monkeypatch.setattr(CTCWordConverter, "_extend_sentence_paths",
                        lambda self, *args: parent)
    extras = [("거부", 0.0, parent[0][2]), ("허용", 0.0, parent[1][2])]
    monkeypatch.setattr(ctc_word_proposals, "sentence_extras",
                        lambda *args, **kwargs: extras)
    obj = converter()
    result = obj._extend_sentence_paths("ga na da ra", "", parent_paths())
    assert "허용" in [item[0] for item in result] and "거부" not in [item[0] for item in result]
    obj = converter()
    obj.sentence_ranker = Ranker([float("nan")])
    with pytest.raises(RuntimeError, match="invalid scores"):
        obj._extend_sentence_paths("ga na da ra", "", parent_paths())


def test_duplicate_growth_candidate_is_rejected(monkeypatch):
    parent = parent_paths()
    monkeypatch.setattr(CTCWordConverter, "_extend_sentence_paths",
                        lambda self, *args: parent)
    monkeypatch.setattr(ctc_word_proposals, "sentence_extras",
                        lambda *args, **kwargs: [(parent[0][0], 0.0, parent[0][2])])
    obj = converter()
    with pytest.raises(RuntimeError, match="duplicate paths"):
        obj._extend_sentence_paths("ga na da ra", "", parent_paths())


def test_raw_request_preserves_parent_metadata_and_never_generates():
    ctc1 = Model(prediction())
    growth = Model(prediction())
    obj = CTCUnionConverter(external_lexicon=False, word_model=ctc1,
                            growth_model=growth)
    obj.load_models = lambda: pytest.fail("raw request loaded models")
    response = obj.convert("https://example.com test", raw=True)
    assert response["result"] == "https://example.com test"
    assert ctc1.calls == growth.calls == []
    assert "ctc_word" in response and "ctc_union" in response
    assert response["ctc_word"]["generation_requests"] == 0
    assert response["ctc_union"]["generation_requests"] == 0


def test_load_growth_model_injection_is_explicit_idempotent_and_pinned():
    growth = Model(prediction())
    obj = CTCUnionConverter(external_lexicon=False, word_model=Model(prediction()),
                            growth_model=growth)
    assert obj.load_growth_model() is growth and obj.load_growth_model() is growth
    checkpoint = next(value for name, value in obj.growth_artifacts.items()
                      if name.endswith("sentence-ctc-growth-pilot-v1/model.pt"))
    metadata = next(value for name, value in obj.growth_artifacts.items()
                    if name.endswith("sentence-ctc-growth-pilot-v1/model.json"))
    assert checkpoint == EXPECTED_GROWTH_CHECKPOINT_SHA256
    assert metadata == EXPECTED_GROWTH_METADATA_SHA256


def test_repeated_parent_load_keeps_union_version_and_all_artifacts(monkeypatch):
    obj = CTCUnionConverter(external_lexicon=False, word_model=Model(prediction()),
                            growth_model=Model(prediction()))
    obj.models_loaded = False
    def parent_load(self):
        if not self.models_loaded:
            self.models_loaded = True
            self.version = "ctc-parent"
            self.model_artifacts = {"parent": "hash"}
    monkeypatch.setattr(CTCWordConverter, "load_models", parent_load)
    obj.load_models()
    version = obj.version
    obj.load_models()
    assert obj.version == version
    assert obj.growth_parent_version == "ctc-parent"
    assert obj.model_artifacts == {"parent": "hash", **obj.growth_artifacts}


def test_union_metadata_allows_full_prediction_fields(monkeypatch):
    parent = parent_paths()
    monkeypatch.setattr(CTCWordConverter, "_extend_sentence_paths",
                        lambda self, *args: parent)
    monkeypatch.setattr(CTCWordConverter, "convert",
                        lambda self, text, context, engine, mode, raw: (
                            self._extend_sentence_paths(text, context, parent)
                            and {"ctc_word": {"preserved": True}}))
    monkeypatch.setattr(ctc_word_proposals, "sentence_extras",
                        lambda *args, **kwargs: [])
    obj = converter(prediction(experimental_audit={"growth": True}))
    obj.growth_parent_version = "ctc-parent"
    obj.growth_artifacts = {"growth": "hash"}
    response = obj.convert("ga na da ra")
    assert response["ctc_word"] == {"preserved": True}
    assert response["ctc_union"]["prediction"]["experimental_audit"] == {
        "growth": True}
    assert response["ctc_union"]["parent_version"] == "ctc-parent"
    assert response["ctc_union"]["artifacts"] == {"growth": "hash"}
