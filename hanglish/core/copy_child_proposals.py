"""Bounded phonetic alternatives for proven CONVERT children of COPY spans.

The helper is deliberately model-free.  ``child_requests`` discovers the
small ASCII suffixes for which an upstream caller may run a phonetic model;
``propose_children`` validates those externally generated beams and replaces
one proven CONVERT child at a time.  Automatic COPY children are never
rewritten.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re
from collections.abc import Mapping, Sequence

from hanglish.core.copy_boundaries import refine_path
from hanglish.core.phonology import romanize_word


_ASCII_LETTERS = re.compile(r"[A-Za-z]{1,40}\Z")
_MODEL_SOURCE = re.compile(r"[a-z]{1,40}\Z")
_HANGUL = re.compile(r"[\uac00-\ud7a3]+\Z")
_SEGMENT_FIELDS = ("input_start", "input_end", "source", "text", "kind")
_MAX_BEAM = 8
_MAX_PROPOSALS = 96


def _request(parent_index, segment_index, child):
    child_source = child["source"]
    if _ASCII_LETTERS.fullmatch(child_source) is None:
        return None
    model_source = child_source.lower()
    return {
        "request_index": None,
        "parent_index": parent_index,
        "segment_index": segment_index,
        "input_start": child["input_start"],
        "input_end": child["input_end"],
        "source": child_source,
        "model_source": model_source,
        "normalization": {
            "operation": "ascii_lower",
            "scope": "convert_child_only",
            "input": child_source,
            "output": model_source,
        },
    }


def child_requests(source: str, text: str, segments: list, *, generator) -> dict:
    """Return model inputs for uniquely refined CONVERT children.

    The returned refinement retains unresolved and alternative parent evidence.
    Only a derivation whose status is ``refined`` can create a request.
    """
    refinement = refine_path(source, text, segments, generator=generator)
    requests = []
    skipped = []
    segment_index = 0
    for derivation in refinement["derivations"]:
        children = derivation["children"]
        if derivation["status"] == "refined":
            convert = [child for child in children
                       if child.get("action") == "convert"]
            if len(convert) != 1:
                raise RuntimeError("refined derivation has no unique CONVERT child")
            convert_index = next(
                index for index, child in enumerate(children)
                if child.get("action") == "convert")
            request = _request(
                derivation["parent_index"], segment_index + convert_index,
                convert[0])
            if request is None:
                skipped.append({
                    "parent_index": derivation["parent_index"],
                    "reason": "convert_child_is_not_1_to_40_ascii_letters",
                    "source": convert[0]["source"],
                })
            else:
                request["request_index"] = len(requests)
                requests.append(request)
        segment_index += len(children)

    unresolved = [
        {"parent_index": row["parent_index"], "status": row["status"],
         "reason": row.get("reason")}
        for row in refinement["derivations"]
        if row["status"] in {"unresolved", "not_applicable"}
    ]
    return {
        "refinement": refinement,
        "requests": requests,
        "diagnostics": {
            "refined_derivations": sum(
                row["status"] == "refined"
                for row in refinement["derivations"]),
            "request_count": len(requests),
            "skipped_children": skipped,
            "unresolved_parents": unresolved,
        },
    }


def _validate_limit(limit):
    if (type(limit) is not int or not 1 <= limit <= _MAX_PROPOSALS):
        raise ValueError(f"limit must be an integer from 1 to {_MAX_PROPOSALS}")


def _validate_beam(model_source, beam):
    if not isinstance(beam, Sequence) or isinstance(beam, (str, bytes)):
        raise ValueError(f"generated beam for {model_source!r} must be a sequence")
    if len(beam) > _MAX_BEAM:
        raise ValueError(f"generated beam for {model_source!r} exceeds {_MAX_BEAM}")
    result = []
    seen = set()
    prior_probability = None
    for item in beam:
        if (not isinstance(item, (tuple, list)) or len(item) != 2
                or not isinstance(item[0], str)
                or _HANGUL.fullmatch(item[0]) is None
                or isinstance(item[1], bool)
                or not isinstance(item[1], (int, float))
                or not math.isfinite(item[1])):
            raise ValueError(f"generated beam for {model_source!r} is malformed")
        target, probability = item[0], float(item[1])
        if target in seen:
            raise ValueError(f"generated beam for {model_source!r} has duplicates")
        if prior_probability is not None and probability > prior_probability:
            raise ValueError(
                f"generated beam for {model_source!r} is not strictly sorted")
        if romanize_word(target) != model_source:
            raise ValueError(
                f"generated target for {model_source!r} fails forward romanization")
        prior_probability = probability
        seen.add(target)
        result.append((target, probability))
    return result


def _replacement_segments(refined_segments, request, target):
    result = []
    for index, segment in enumerate(refined_segments):
        row = {field: segment[field] for field in _SEGMENT_FIELDS}
        if index == request["segment_index"]:
            if (row["input_start"] != request["input_start"]
                    or row["input_end"] != request["input_end"]
                    or row["source"] != request["source"]):
                raise RuntimeError("CONVERT child identity changed")
            row["text"] = target
            row["kind"] = "copy_child_phonetic"
        result.append(row)
    return result


def _validate_replacement(source, candidate_text, original, replacement,
                          request, copy_spans):
    if len(original) != len(replacement):
        raise RuntimeError("child replacement changed the segment count")
    cursor = 0
    rendered = []
    for index, (before, after) in enumerate(zip(original, replacement)):
        if set(after) != set(_SEGMENT_FIELDS):
            raise RuntimeError("child replacement emitted a malformed segment")
        start, end = after["input_start"], after["input_end"]
        if (type(start) is not int or type(end) is not int
                or start != cursor or not start < end <= len(source)
                or after["source"] != source[start:end]):
            raise RuntimeError("child replacement changed source coverage")
        if index != request["segment_index"] and after != before:
            raise RuntimeError("child replacement changed another segment")
        if index == request["segment_index"] and any(
                after[field] != before[field]
                for field in ("input_start", "input_end", "source")):
            raise RuntimeError("child replacement changed its source span")
        rendered.append(after["text"])
        cursor = end
    if cursor != len(source) or "".join(rendered) != candidate_text:
        raise RuntimeError("child replacement does not render its candidate")

    for span in copy_spans:
        if not span.get("automatic_hypothesis"):
            continue
        start, end = span["source_boundary"]
        matches = [row for row in replacement
                   if row["input_start"] == start and row["input_end"] == end]
        if (len(matches) != 1 or matches[0]["source"] != span["source"]
                or matches[0]["text"] != span["text"]):
            raise RuntimeError("child replacement changed an automatic COPY range")


def propose_children(source: str, text: str, segments: list, *, generator,
                     generated: Mapping, existing_texts=(), limit: int = 96) -> dict:
    """Build bounded, one-child phonetic alternatives from external beams."""
    _validate_limit(limit)
    if not isinstance(generated, Mapping):
        raise TypeError("generated must be a mapping")
    if isinstance(existing_texts, (str, bytes)):
        raise TypeError("existing_texts must be an iterable of strings")
    try:
        excluded = set(existing_texts)
    except TypeError as exc:
        raise TypeError("existing_texts must be iterable") from exc
    if any(not isinstance(value, str) for value in excluded):
        raise ValueError("existing_texts must contain only strings")
    excluded.add(text)

    discovery = child_requests(source, text, segments, generator=generator)
    refinement = discovery["refinement"]
    beams = {}
    for request in discovery["requests"]:
        model_source = request["model_source"]
        if _MODEL_SOURCE.fullmatch(model_source) is None:
            raise RuntimeError("request normalization produced an invalid model source")
        if model_source not in generated:
            raise ValueError(f"missing generated beam for {model_source!r}")
        if model_source not in beams:
            beams[model_source] = _validate_beam(
                model_source, generated[model_source])

    proposals = []
    seen = set(excluded)
    generation = []
    stopped = False
    for request in discovery["requests"]:
        beam = beams[request["model_source"]]
        accepted = 0
        skipped = 0
        for target, probability in beam:
            replacement = _replacement_segments(
                refinement["segments"], request, target)
            candidate_text = "".join(row["text"] for row in replacement)
            _validate_replacement(
                source, candidate_text, refinement["segments"], replacement,
                request, refinement["copy_spans"])
            if candidate_text in seen:
                skipped += 1
                continue
            seen.add(candidate_text)
            parent = refinement["derivations"][request["parent_index"]]
            proposals.append({
                "text": candidate_text,
                "segments": replacement,
                "source_span": [request["input_start"], request["input_end"]],
                "model_source": request["model_source"],
                "replacement": {"from": refinement["segments"][
                    request["segment_index"]]["text"], "to": target},
                "phonetic_log_probability": probability,
                "parent_evidence": {
                    "parent_text": text,
                    "parent_index": request["parent_index"],
                    "segment_index": request["segment_index"],
                    "derivation": deepcopy(parent),
                },
            })
            accepted += 1
            if len(proposals) == limit:
                stopped = True
                break
        generation.append({
            "request_index": request["request_index"],
            "model_source": request["model_source"],
            "status": "empty_beam" if not beam else "generated",
            "beam_size": len(beam),
            "accepted": accepted,
            "skipped_existing_or_duplicate": skipped,
        })
        if stopped:
            break

    diagnostics = deepcopy(discovery["diagnostics"])
    diagnostics.update({
        "generation": generation,
        "proposal_count": len(proposals),
        "limit": limit,
        "limit_reached": stopped,
    })
    return {
        "refinement": refinement,
        "requests": deepcopy(discovery["requests"]),
        "proposals": proposals,
        "diagnostics": diagnostics,
    }


__all__ = ["child_requests", "propose_children"]
