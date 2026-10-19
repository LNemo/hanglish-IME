import copy

import pytest

from scripts.final_choice_capture import capture_case, validate_response


VERSION = "neural-test"
ARTIFACTS = {"checkpoint/model.safetensors": "a" * 64}


def candidate(text, score, *, source="roman", kind="neural"):
    return {"text": text, "score": score, "segments": [
        {"input_start": 0, "input_end": len(source), "source": source,
         "text": text, "kind": kind}
    ]}


def response(candidates, *, raw=False, artifacts=ARTIFACTS):
    return {"input": "roman", "context": "문맥", "engine": "neural",
            "mode": "batch", "raw": raw, "version": VERSION,
            "candidates": candidates, "result": candidates[0]["text"],
            "ranker": None if raw else "hybrid",
            "model_artifacts": {} if raw else artifacts,
            "context_used": not raw, "external_context_used": not raw,
            "ranking_applied": not raw, "sentence_ranking_applied": False,
            "warnings": [], "engine_ms": 1.25}


def raw_response():
    return response([candidate("roman", 0.0, source="roman", kind="raw")], raw=True)


class TraceSource(dict):
    def __init__(self, **values):
        super().__init__(**values)
        self.label_reads = 0

    def __getitem__(self, key):
        if key == "expected":
            self.label_reads += 1
        return super().__getitem__(key)


class Ranker:
    def __init__(self, scoreable=True):
        self.scoreable = scoreable

    def can_score(self, text):
        return self.scoreable or text != "너무긴후보"


class Converter:
    def __init__(self, source, normal, raw, scoreable=True):
        self.source = source
        self.normal = normal
        self.raw_response = raw
        self.sentence_ranker = Ranker(scoreable)
        self.calls = []

    def convert(self, *args, **kwargs):
        assert self.source.label_reads == 0
        self.calls.append((args, kwargs))
        return copy.deepcopy(self.raw_response if kwargs.get("raw") else self.normal)


def source(expected=None):
    return TraceSource(id="row", group_id="group", origin="natural",
                       input="roman", context="문맥", expected=expected or ["로만"])


def test_capture_runs_normal_and_raw_before_labels_and_preserves_objects():
    row = source()
    candidates = [candidate("로만", 1.2345), candidate("roman", None, kind="raw")]
    converter = Converter(row, response(candidates), raw_response())

    captured = capture_case(converter, row, expected_runtime_version=VERSION,
                            expected_model_artifacts=ARTIFACTS)

    assert converter.calls == [(('roman', '문맥'), {}),
                               (('roman', '문맥'), {"raw": True})]
    assert row.label_reads == 1
    assert captured["candidates"] == candidates
    assert captured["finite_indices"] == [0]
    assert captured["accepted_positive_indices"] == [0]
    assert captured["selector_eligible"] is False
    assert captured["ineligible_reasons"] == ["insufficient_finite_candidates"]
    assert captured["raw_bypass_response"] == raw_response()
    assert captured["model_artifact_flags"]["response_matches_expected"] is True


def test_none_scored_accepted_original_is_never_a_positive_or_zero():
    row = source(expected=["roman"])
    candidates = [candidate("로만", 1.0), candidate("roman", None, kind="raw")]
    converter = Converter(row, response(candidates), raw_response())
    captured = capture_case(converter, row, expected_runtime_version=VERSION,
                            expected_model_artifacts=ARTIFACTS)
    assert captured["finite_indices"] == [0]
    assert captured["accepted_positive_indices"] == []
    assert captured["label_aligned"] is False


def test_oversized_candidate_is_retained_and_marks_row_ineligible():
    row = source(expected=["정상"])
    candidates = [candidate("정상", 2.0), candidate("너무긴후보", 1.0)]
    converter = Converter(row, response(candidates), raw_response(), scoreable=False)
    captured = capture_case(converter, row, expected_runtime_version=VERSION,
                            expected_model_artifacts=ARTIFACTS)
    assert captured["candidates"] == candidates
    assert captured["candidate_can_score"] == [True, False]
    assert captured["all_candidates_can_score"] is False
    assert captured["selector_eligible"] is False
    assert captured["ineligible_reasons"] == ["candidate_exceeds_sentence_model_budget"]


@pytest.mark.parametrize("mutation, message", [
    (lambda value: value["candidates"].append(copy.deepcopy(value["candidates"][0])), "unique"),
    (lambda value: value["candidates"][0].update(score=float("inf")), "rounded finite"),
    (lambda value: value["candidates"][0].update(score=1.23456), "rounded finite"),
    (lambda value: value["candidates"][0]["segments"][0].update(input_end=4), "align"),
    (lambda value: (value["candidates"][0].update(text="다름"),
                    value.update(result="다름")), "concatenation"),
    (lambda value: value.update(result="다름"), "top five"),
])
def test_malformed_candidate_objects_are_rejected(mutation, message):
    value = response([candidate("로만", 1.0)])
    mutation(value)
    with pytest.raises(ValueError, match=message):
        validate_response(value, "roman", "문맥", VERSION, ARTIFACTS, raw=False)


def test_runtime_artifacts_and_raw_exact_copy_are_enforced():
    value = response([candidate("로만", 1.0)], artifacts={"wrong": "b" * 64})
    with pytest.raises(ValueError, match="model artifacts"):
        validate_response(value, "roman", "문맥", VERSION, ARTIFACTS, raw=False)
    raw = raw_response()
    raw["candidates"][0]["score"] = None
    with pytest.raises(ValueError, match="raw bypass"):
        validate_response(raw, "roman", "문맥", VERSION, ARTIFACTS, raw=True)


def test_normal_response_raw_segments_cannot_transform_the_source():
    value = response([candidate('로만', 1.0, kind='raw')])
    with pytest.raises(ValueError, match='raw segment must preserve'):
        validate_response(value, 'roman', '문맥', VERSION, ARTIFACTS, raw=False)


def test_none_score_cannot_silently_remove_a_transformed_candidate():
    value = response([candidate('로만', None)])
    with pytest.raises(ValueError, match='None score is reserved'):
        validate_response(value, 'roman', '문맥', VERSION, ARTIFACTS, raw=False)
    value = response([candidate('roman', None, kind='neural')])
    with pytest.raises(ValueError, match='None score is reserved'):
        validate_response(value, 'roman', '문맥', VERSION, ARTIFACTS, raw=False)
