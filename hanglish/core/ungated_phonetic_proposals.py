"""Generate bounded phonetic proposals without sentence-ranker gating.

This helper accepts already constructed, non-raw parent paths.  It validates
their source coverage, generates each eligible Roman word once, and emits
one-segment replacements in stable parent/segment/beam order.  It deliberately
does not know about sentence scoring or tokenizer length limits.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
import math
import re

from hanglish.core.phonology import romanize_word


_ELIGIBLE = re.compile(r"[a-z]{2,40}\Z")
_HANGUL = re.compile(r"[\uac00-\ud7a3]+\Z")
_MAX_BEAM = 8
_HARD_LIMIT = 96


def _validate_limit(name: str, value: int) -> None:
    if type(value) is not int or not 1 <= value <= _HARD_LIMIT:
        raise ValueError(f"{name} must be an integer from 1 to {_HARD_LIMIT}")


def _validate_parent(source: str, parent: Mapping, parent_index: int) -> None:
    if not isinstance(parent, Mapping):
        raise TypeError(f"parent {parent_index} must be a mapping")
    text = parent.get("text")
    segments = parent.get("segments")
    if not isinstance(text, str):
        raise ValueError(f"parent {parent_index} text must be a string")
    if (not isinstance(segments, Sequence)
            or isinstance(segments, (str, bytes))):
        raise ValueError(f"parent {parent_index} segments must be a sequence")

    cursor = 0
    rendered = []
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise ValueError(
                f"parent {parent_index} segment {segment_index} must be a mapping")
        start = segment.get("input_start")
        end = segment.get("input_end")
        segment_source = segment.get("source")
        segment_text = segment.get("text")
        if (type(start) is not int or type(end) is not int
                or start != cursor or not start < end <= len(source)
                or segment_source != source[start:end]
                or not isinstance(segment_text, str)):
            raise ValueError(
                f"parent {parent_index} segments do not preserve source offsets")
        cursor = end
        rendered.append(segment_text)
    if cursor != len(source) or "".join(rendered) != text:
        raise ValueError(
            f"parent {parent_index} does not cover source or render its text")


def _validate_beam(source_word: str, outputs) -> list[tuple[str, float]]:
    if (not isinstance(outputs, Sequence)
            or isinstance(outputs, (str, bytes))
            or len(outputs) > _MAX_BEAM):
        raise ValueError(
            f"generated beam for {source_word!r} must contain at most {_MAX_BEAM} items")

    validated = []
    seen = set()
    previous = float("inf")
    for item in outputs:
        if (not isinstance(item, (tuple, list)) or len(item) != 2
                or not isinstance(item[0], str)
                or _HANGUL.fullmatch(item[0]) is None
                or isinstance(item[1], bool)
                or not isinstance(item[1], (int, float))
                or not math.isfinite(item[1])):
            raise ValueError(f"generated beam for {source_word!r} is malformed")
        target, score = item[0], float(item[1])
        if target in seen:
            raise ValueError(f"generated beam for {source_word!r} has duplicates")
        if score > previous:
            raise ValueError(f"generated beam for {source_word!r} is not sorted")
        if romanize_word(target) != source_word:
            raise ValueError(
                f"generated target for {source_word!r} fails forward romanization")
        validated.append((target, score))
        seen.add(target)
        previous = score
    return validated


def _known_english_inventory(values: Iterable[str]) -> set[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError("known_english must be an iterable of strings")
    try:
        inventory = set(values)
    except TypeError as exc:
        raise TypeError("known_english must be an iterable of strings") from exc
    if any(not isinstance(value, str) for value in inventory):
        raise ValueError("known_english must contain only strings")
    return {value.lower() for value in inventory}


def generate_proposals(
        source: str,
        parents,
        generate: Callable,
        *,
        known_english=(),
        max_sources: int = 96,
        max_extras: int = 96,
) -> dict:
    """Return stable one-segment phonetic alternatives for validated parents.

    ``generate`` is called as ``generate(source_word, beam_size=8)`` once for
    each distinct eligible source word.  The helper never scores candidates;
    callers may separately decide which generated texts their ranker supports.
    """
    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if not callable(generate):
        raise TypeError("generate must be callable")
    if (not isinstance(parents, Sequence)
            or isinstance(parents, (str, bytes))):
        raise TypeError("parents must be a sequence")
    _validate_limit("max_sources", max_sources)
    _validate_limit("max_extras", max_extras)
    english = _known_english_inventory(known_english)

    preserved_parents = deepcopy(list(parents))
    for parent_index, parent in enumerate(preserved_parents):
        _validate_parent(source, parent, parent_index)

    eligible = []
    eligible_seen = set()
    skipped_known_english = []
    skipped_english_seen = set()
    for parent_index, parent in enumerate(preserved_parents):
        for segment_index, segment in enumerate(parent["segments"]):
            source_word = segment["source"]
            if _ELIGIBLE.fullmatch(source_word) is None:
                continue
            occurrence = (segment["input_start"], segment["input_end"], source_word)
            if source_word.lower() in english:
                if occurrence not in skipped_english_seen:
                    skipped_known_english.append({
                        "source_start": occurrence[0],
                        "source_end": occurrence[1],
                        "source": occurrence[2],
                    })
                    skipped_english_seen.add(occurrence)
                continue
            identity = (parent_index, segment_index)
            if identity not in eligible_seen:
                eligible.append((parent_index, segment_index, occurrence))
                eligible_seen.add(identity)

    generated_by_source = {}
    validated_by_source = {}
    skipped_generation_sources = []
    skipped_generation_seen = set()
    for _, _, occurrence in eligible:
        source_word = occurrence[2]
        if source_word in generated_by_source:
            continue
        if len(generated_by_source) == max_sources:
            if occurrence not in skipped_generation_seen:
                skipped_generation_sources.append({
                    "source_start": occurrence[0],
                    "source_end": occurrence[1],
                    "source": source_word,
                })
                skipped_generation_seen.add(occurrence)
            continue
        raw_outputs = generate(source_word, beam_size=_MAX_BEAM)
        validated_by_source[source_word] = _validate_beam(source_word, raw_outputs)
        generated_by_source[source_word] = deepcopy(raw_outputs)

    parent_texts = {parent["text"] for parent in preserved_parents}
    seen_texts = set(parent_texts)
    proposals = []
    extras_exhausted = False
    for parent_index, segment_index, occurrence in eligible:
        source_word = occurrence[2]
        if source_word not in validated_by_source:
            continue
        for replacement, _ in validated_by_source[source_word]:
            replacement_segments = deepcopy(
                preserved_parents[parent_index]["segments"])
            replacement_segments[segment_index]["text"] = replacement
            replacement_segments[segment_index]["kind"] = "ungated_phonetic_word"
            text = "".join(segment["text"] for segment in replacement_segments)
            _validate_parent(
                source, {"text": text, "segments": replacement_segments}, parent_index)
            if text in seen_texts:
                continue
            seen_texts.add(text)
            proposals.append({
                "text": text,
                "segments": replacement_segments,
                "parent_index": parent_index,
                "source_start": occurrence[0],
                "source_end": occurrence[1],
                "source": source_word,
                "replacement": replacement,
            })
            if len(proposals) == max_extras:
                extras_exhausted = True
                break
        if extras_exhausted:
            break

    source_budget_exhausted = bool(skipped_generation_sources)
    return {
        "parents": preserved_parents,
        "proposals": proposals,
        "generation_requests": len(generated_by_source),
        "generated_by_source": generated_by_source,
        "skipped_known_english": skipped_known_english,
        "skipped_generation_sources": skipped_generation_sources,
        "generation_budget_exhausted": source_budget_exhausted or extras_exhausted,
    }


__all__ = ["generate_proposals"]
