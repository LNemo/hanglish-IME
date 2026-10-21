import threading

import pytest

from hanglish.core import ctc_word_proposals, mixed_span_proposals
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
    obj = object.__new__(CTCWordConverter)
    obj.lock = threading.RLock()
    obj.word_model = Model(response or prediction())
    obj._word_model_loaded = True
    obj.sentence_ranker = Ranker()
    obj.word_options = lambda *args: ()
    obj.mixed_span_counts = {"generated": 0, "accepted": 0}
    obj.ctc_word_counts = obj._empty_counts()
    obj.ctc_word_prediction = None
    obj.ctc_word_ms = 0.0
    return obj


def test_complete_ctc_generation_scores_only_new_paths(monkeypatch):
    monkeypatch.setattr(mixed_span_proposals, "sentence_extras", lambda *args, **kwargs: [])
    proposed = [("신규", 0.0, parent_paths()[0][2])]
    monkeypatch.setattr(ctc_word_proposals, "sentence_extras",
                        lambda *args, **kwargs: proposed)
    obj = converter()
    parent = parent_paths()
    result = obj._extend_sentence_paths("ga na da ra", "문맥", parent)
    assert obj.word_model.calls == [["ga na da ra"]]
    assert obj.sentence_ranker.calls == [(["문맥"], ["신규"])]
    assert result[0][0] == "신규" and parent[0] in result
    assert obj.ctc_word_prediction["decode_mode"] == "ctc_prefix_beam"
    assert obj.ctc_word_counts == {
        "generation_requests": 1, "completed_generations": 1,
        "generated_proposals": 1, "accepted_proposals": 1}


@pytest.mark.parametrize("source,reason", [
    ("singleword", "unsupported_source"),
    ("ga 한글", "unsupported_source"),
    ("a " + "b" * 509, "source_token_limit"),
])
def test_unsupported_input_preserves_exact_parent(monkeypatch, source, reason):
    monkeypatch.setattr(mixed_span_proposals, "sentence_extras", lambda *args, **kwargs: [])
    obj = converter()
    parent = parent_paths()
    assert obj._extend_sentence_paths(source, "", parent) is parent
    assert obj.word_model.calls == []
    assert obj.ctc_word_prediction["reason"] == reason


@pytest.mark.parametrize("reason", ["empty_output", "invalid_utf8", "invalid_jamo",
                                     "no_valid_path"])
def test_incomplete_generation_preserves_parent(monkeypatch, reason):
    monkeypatch.setattr(mixed_span_proposals, "sentence_extras", lambda *args, **kwargs: [])
    obj = converter(prediction(text=None, reason=reason, generated_tokens=0,
                               labels=[], log_probability=None))
    parent = parent_paths()
    assert obj._extend_sentence_paths("ga na da ra", "", parent) is parent
    assert obj.ctc_word_counts["generated_proposals"] == 0


def test_extra_response_fields_allowed_but_malformed_core_rejected(monkeypatch):
    monkeypatch.setattr(mixed_span_proposals, "sentence_extras", lambda *args, **kwargs: [])
    obj = converter(prediction(experimental_audit={"ok": True}))
    obj._extend_sentence_paths("ga na da ra", "", parent_paths())
    assert obj.ctc_word_prediction["experimental_audit"] == {"ok": True}
    obj = converter(prediction(labels=[1]))
    with pytest.raises(RuntimeError, match="invalid labels"):
        obj._extend_sentence_paths("ga na da ra", "", parent_paths())


def test_unscoreable_extra_filtered_and_invalid_ranker_score_fails(monkeypatch):
    monkeypatch.setattr(mixed_span_proposals, "sentence_extras", lambda *args, **kwargs: [])
    extras = [("거부", 0.0, parent_paths()[0][2]),
              ("허용", 0.0, parent_paths()[1][2])]
    monkeypatch.setattr(ctc_word_proposals, "sentence_extras", lambda *args, **kwargs: extras)
    obj = converter()
    result = obj._extend_sentence_paths("ga na da ra", "", parent_paths())
    assert "허용" in [item[0] for item in result] and "거부" not in [item[0] for item in result]
    obj = converter()
    obj.sentence_ranker = Ranker([float("nan")])
    with pytest.raises(RuntimeError, match="invalid scores"):
        obj._extend_sentence_paths("ga na da ra", "", parent_paths())


def test_raw_request_never_loads_or_generates():
    obj = CTCWordConverter(external_lexicon=False, word_model=Model(prediction()))
    obj.load_models = lambda: pytest.fail("raw request loaded models")
    response = obj.convert("https://example.com test", raw=True)
    assert response["result"] == "https://example.com test"
    assert obj.word_model.calls == []
    assert response["ctc_word"]["generation_requests"] == 0


def test_load_word_model_explicit_idempotent_and_artifacts_pinned():
    model = Model(prediction(text=None, reason="empty_output", generated_tokens=0,
                             labels=[], log_probability=None))
    obj = CTCWordConverter(external_lexicon=False, word_model=model)
    assert obj.load_word_model() is model and obj.load_word_model() is model
    assert any(name.endswith("ctc-rr-retraining-20261020/training-word/model.pt")
               for name in obj.ctc_word_artifacts)
    assert any(name.endswith("ctc-rr-retraining-20261020/training-word/model.json")
               for name in obj.ctc_word_artifacts)


def test_repeated_parent_load_keeps_version_and_artifact_pins(monkeypatch):
    obj = CTCWordConverter(external_lexicon=False, word_model=Model(prediction()))
    obj.models_loaded = False
    def parent_load(self):
        if not self.models_loaded:
            self.models_loaded = True
            self.version = "mixed-parent"
            self.model_artifacts = {"parent": "hash"}
    monkeypatch.setattr("hanglish.core.mixed_span_converter.MixedSpanConverter.load_models",
                        parent_load)
    obj.load_models()
    version = obj.version
    obj.load_models()
    assert obj.version == version and obj.word_parent_version == "mixed-parent"
    assert obj.model_artifacts == {"parent": "hash", **obj.ctc_word_artifacts}
