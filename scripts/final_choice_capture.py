"""Pure capture helpers for the experimental final-candidate selector.

This module has no CLI and chooses no data split. Callers supply an already
frozen converter, runtime version, and exact model-artifact mapping. Labels are
read only after both the normal and ``raw=True`` conversions complete.
"""
from __future__ import annotations

from copy import deepcopy
import math

from scripts.final_choice_policy import finite_candidate_indices


MAX_CANDIDATES = 5


def _validate_identity(source):
    values = {}
    for key in ("id", "group_id", "origin", "input", "context"):
        value = source.get(key)
        if not isinstance(value, str) or (key != "context" and not value):
            raise ValueError(f"capture source has an invalid {key}")
        values[key] = value
    return values


def _validate_segments(candidate, source):
    segments = candidate.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("candidate requires complete segments")
    cursor = 0
    output = []
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("candidate segment must be an object")
        start, end = segment.get("input_start"), segment.get("input_end")
        if (type(start) is not int or type(end) is not int or start != cursor
                or not start < end <= len(source)
                or segment.get("source") != source[start:end]
                or not isinstance(segment.get("text"), str)
                or not isinstance(segment.get("kind"), str)
                or not segment["kind"]):
            raise ValueError("candidate segments do not align to the complete source")
        if segment["kind"] == "raw" and segment["text"] != segment["source"]:
            raise ValueError("raw segment must preserve its source exactly")
        output.append(segment["text"])
        cursor = end
    if cursor != len(source) or candidate.get("text") != "".join(output):
        raise ValueError("candidate text differs from its segment concatenation")


def validate_response(response, source, context, expected_runtime_version,
                      expected_model_artifacts, *, raw):
    if (not isinstance(response, dict)
            or response.get("input") != source
            or response.get("context") != context
            or response.get("engine") != "neural"
            or response.get("raw") is not raw
            or response.get("version") != expected_runtime_version
            or response.get("result") is None
            or type(response.get("engine_ms")) not in (int, float)
            or not math.isfinite(response["engine_ms"])
            or response["engine_ms"] < 0):
        raise ValueError("converter response metadata differs from capture contract")
    candidates = response.get("candidates")
    if (not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_CANDIDATES
            or response.get("result") != candidates[0].get("text")):
        raise ValueError("converter response must preserve its complete top five")
    texts = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
            raise ValueError("candidate object is malformed")
        score = candidate.get("score")
        if (score is not None and (
                type(score) not in (int, float) or not math.isfinite(score)
                or round(score, 4) != score)):
            raise ValueError("candidate score must be rounded finite or None")
        _validate_segments(candidate, source)
        if not raw and score is None and (
                candidate["text"] != source
                or candidate["segments"] != [{
                    "input_start": 0, "input_end": len(source),
                    "source": source, "text": source, "kind": "raw",
                }]):
            raise ValueError("None score is reserved for the complete original raw candidate")
        texts.append(candidate["text"])
    if len(texts) != len(set(texts)):
        raise ValueError("candidate texts must be unique")
    artifacts = response.get("model_artifacts")
    if raw:
        expected_candidate = {
            "text": source,
            "score": 0.0,
            "segments": [{"input_start": 0, "input_end": len(source),
                          "source": source, "text": source, "kind": "raw"}],
        }
        if (candidates != [expected_candidate] or response.get("result") != source
                or artifacts != {} or response.get("ranker") is not None):
            raise ValueError("raw bypass response is not an exact source copy")
    elif artifacts != expected_model_artifacts:
        raise ValueError("converter model artifacts differ from caller pins")
    return candidates


def _response_flags(response):
    return {key: deepcopy(response.get(key)) for key in (
        "engine", "mode", "raw", "version", "ranker", "model_artifacts",
        "context_used", "external_context_used", "ranking_applied",
        "sentence_ranking_applied", "warnings",
    )}


def capture_case(converter, source, *, expected_runtime_version,
                 expected_model_artifacts):
    """Capture one parent result and raw bypass before aligning labels."""
    identity = _validate_identity(source)
    if (not isinstance(expected_runtime_version, str) or not expected_runtime_version
            or not isinstance(expected_model_artifacts, dict)
            or any(not isinstance(path, str) or not path
                   or not isinstance(digest, str) or len(digest) != 64
                   for path, digest in expected_model_artifacts.items())):
        raise ValueError("caller runtime and model-artifact pins are malformed")

    # Inputs and context are the only data fields passed into conversion.
    parent = converter.convert(identity["input"], identity["context"])
    raw_response = converter.convert(identity["input"], identity["context"], raw=True)
    candidates = validate_response(
        parent, identity["input"], identity["context"], expected_runtime_version,
        expected_model_artifacts, raw=False,
    )
    validate_response(
        raw_response, identity["input"], identity["context"], expected_runtime_version,
        expected_model_artifacts, raw=True,
    )
    sentence_ranker = getattr(converter, "sentence_ranker", None)
    if sentence_ranker is None or not callable(getattr(sentence_ranker, "can_score", None)):
        raise ValueError("current sentence ranker is required for candidate budget checks")
    can_score = [bool(sentence_ranker.can_score(candidate["text"]))
                 for candidate in candidates]
    finite_indices = finite_candidate_indices(candidates)
    ineligible_reasons = []
    if not all(can_score):
        ineligible_reasons.append("candidate_exceeds_sentence_model_budget")
    if len(finite_indices) < 2 or 0 not in finite_indices:
        ineligible_reasons.append("insufficient_finite_candidates")

    # Label alignment deliberately follows all source-only conversion and
    # sentence-budget work. None-scored originals never enter this index set.
    expected = source["expected"]
    if (not isinstance(expected, list) or not expected
            or any(not isinstance(value, str) or not value for value in expected)):
        raise ValueError("capture labels must be a non-empty accepted string list")
    accepted_positive_indices = [index for index in finite_indices
                                 if candidates[index]["text"] in expected]
    return {
        **identity,
        "expected": list(expected),
        "result": parent["result"],
        "candidates": deepcopy(candidates),
        "finite_indices": finite_indices,
        "accepted_positive_indices": accepted_positive_indices,
        "label_aligned": bool(accepted_positive_indices),
        "candidate_can_score": can_score,
        "all_candidates_can_score": all(can_score),
        "selector_eligible": not ineligible_reasons,
        "ineligible_reasons": ineligible_reasons,
        "runtime_version": parent["version"],
        "model_artifact_flags": {
            "expected_count": len(expected_model_artifacts),
            "response_matches_expected": True,
        },
        "response_flags": _response_flags(parent),
        "engine_ms": parent["engine_ms"],
        "raw_bypass_response": deepcopy(raw_response),
        "raw_engine_ms": raw_response["engine_ms"],
    }


def capture_cases(converter, sources, *, expected_runtime_version,
                  expected_model_artifacts):
    """Capture rows in caller-provided order without split selection."""
    return [capture_case(
        converter, source,
        expected_runtime_version=expected_runtime_version,
        expected_model_artifacts=expected_model_artifacts,
    ) for source in sources]
