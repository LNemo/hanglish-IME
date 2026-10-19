"""Source-aligned word proposals extracted from one generated sentence.

The policy in this module intentionally mirrors the frozen balanced-model
probe.  It aligns only words whose generated Hangul spelling round-trips to
the complete Roman source chunk, then replaces that exact source span in the
first four already-scored sentence paths.  Existing paths and scores are never
changed.
"""
from __future__ import annotations

import math
import re

from hanglish.core.phonology import romanize_word


BASES = 4
MAX_PROPOSALS = 96
SOURCE_WORD = re.compile(r"[a-z]+[.,!?]?\Z")
OUTPUT_WORD = re.compile(r"[가-힣]+[.,!?]?\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _source_chunks(source):
    return [
        {"index": index, "start": match.start(), "end": match.end(),
         "text": match.group()}
        for index, match in enumerate(re.finditer(r"\S+", source))
    ]


def _output_chunks(text):
    if not isinstance(text, str):
        return []
    return [
        {"index": index, "text": match.group()}
        for index, match in enumerate(re.finditer(r"\S+", text))
    ]


def aligned_words(source, generated):
    """Return deterministic source-leftmost round-trip word alignments."""
    if not isinstance(source, str):
        raise ValueError("neural-word source must be text")
    sources = [chunk for chunk in _source_chunks(source)
               if SOURCE_WORD.fullmatch(chunk["text"])]
    outputs = [chunk for chunk in _output_chunks(generated)
               if OUTPUT_WORD.fullmatch(chunk["text"])]
    for chunk in outputs:
        chunk["roman"] = romanize_word(chunk["text"])

    rows, columns = len(sources), len(outputs)
    lengths = [[0] * (columns + 1) for _ in range(rows + 1)]
    for left in range(rows - 1, -1, -1):
        for right in range(columns - 1, -1, -1):
            if sources[left]["text"] == outputs[right]["roman"]:
                lengths[left][right] = 1 + lengths[left + 1][right + 1]
            else:
                lengths[left][right] = max(
                    lengths[left + 1][right], lengths[left][right + 1])

    aligned = []
    left = right = 0
    while left < rows and right < columns:
        if (sources[left]["text"] == outputs[right]["roman"]
                and lengths[left][right] == 1 + lengths[left + 1][right + 1]):
            aligned.append({
                "source_chunk_index": sources[left]["index"],
                "output_chunk_index": outputs[right]["index"],
                "input_start": sources[left]["start"],
                "input_end": sources[left]["end"],
                "source": sources[left]["text"],
                "text": outputs[right]["text"],
                "forward_roman": outputs[right]["roman"],
            })
            left += 1
            right += 1
        elif lengths[left][right + 1] == lengths[left][right]:
            # Preserve the current source occurrence when either skip gives
            # the same LCS length.  Repeated words therefore remain
            # source-leftmost, exactly as in the frozen probe.
            right += 1
        else:
            left += 1
    return aligned


def _validate_path(source, path):
    _require(isinstance(path, tuple) and len(path) == 3
             and isinstance(path[0], str) and _finite_number(path[1])
             and isinstance(path[2], list), "neural-word parent path is malformed")
    cursor = 0
    texts = []
    for segment in path[2]:
        _require(isinstance(segment, dict)
                 and set(segment) >= {"input_start", "input_end", "source", "text", "kind"},
                 "neural-word segment fields are missing")
        start, end = segment["input_start"], segment["input_end"]
        _require(isinstance(start, int) and isinstance(end, int)
                 and start == cursor and start < end <= len(source)
                 and segment["source"] == source[start:end]
                 and isinstance(segment["text"], str),
                 "neural-word segment source coverage differs")
        cursor = end
        texts.append(segment["text"])
    _require(cursor == len(source) and "".join(texts) == path[0],
             "neural-word path does not cover its source")


def _span_texts(source, paths, start, end):
    forms = set()
    for path in paths:
        segments = path[2]
        for first, segment in enumerate(segments):
            if segment["input_start"] != start:
                continue
            text = []
            cursor = start
            for current in segments[first:]:
                if current["input_start"] != cursor or current["input_end"] > end:
                    break
                text.append(current["text"])
                cursor = current["input_end"]
                if cursor == end:
                    forms.add("".join(text))
                    break
    return forms


def _replace_spans(source, base, replacements):
    replacements = sorted(replacements, key=lambda item: item["input_start"])
    _require(all(left["input_end"] <= right["input_start"]
                 for left, right in zip(replacements, replacements[1:])),
             "neural-word replacement spans overlap")
    by_start = {item["input_start"]: item for item in replacements}
    segments = base[2]
    result = []
    index = 0
    while index < len(segments):
        segment = segments[index]
        replacement = by_start.get(segment["input_start"])
        if replacement is None:
            result.append(dict(segment))
            index += 1
            continue
        end = replacement["input_end"]
        scan = index
        position = segment["input_start"]
        while scan < len(segments) and position < end:
            current = segments[scan]
            if current["input_start"] != position or current["input_end"] > end:
                return None
            position = current["input_end"]
            scan += 1
        if position != end:
            return None
        result.append({
            "input_start": replacement["input_start"],
            "input_end": end,
            "source": source[replacement["input_start"]:end],
            "text": replacement["text"],
            "kind": "balanced_word_probe",
        })
        index = scan
    text = "".join(segment["text"] for segment in result)
    candidate = (text, 0.0, result)
    _validate_path(source, candidate)
    return candidate


def sentence_extras(source, generated, paths, limit=MAX_PROPOSALS):
    """Return bounded new paths using source-only generated word spellings.

    The numeric score is a dummy value: callers must score every returned path
    with the existing sentence ranker before adding it to a runtime pool.
    """
    if (not isinstance(source, str) or not isinstance(generated, str)
            or isinstance(limit, bool) or not isinstance(limit, int)
            or not 1 <= limit <= MAX_PROPOSALS):
        raise ValueError("source, generated text and a limit from 1 through 96 are required")
    parents = list(paths)
    if len(parents) < BASES:
        return []
    for path in parents:
        _validate_path(source, path)
    _require(len({path[0] for path in parents}) == len(parents),
             "neural-word parent pool contains duplicates")

    novel = []
    for alignment in aligned_words(source, generated):
        _require(alignment["forward_roman"] == alignment["source"],
                 "neural-word alignment does not round-trip")
        existing = _span_texts(
            source, parents, alignment["input_start"], alignment["input_end"])
        if alignment["text"] not in existing:
            novel.append(alignment)
    if not novel:
        return []

    bases = parents[:BASES]
    seen = {path[0] for path in parents}
    extras = []

    def add(candidate):
        if candidate is None or candidate[0] in seen or len(extras) >= limit:
            return
        seen.add(candidate[0])
        extras.append(candidate)

    for base in bases:
        add(_replace_spans(source, base, novel))
    for replacement in novel:
        for base in bases:
            add(_replace_spans(source, base, [replacement]))
    return extras


__all__ = ["aligned_words", "sentence_extras"]
