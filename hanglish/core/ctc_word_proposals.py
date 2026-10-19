"""Bounded mixed-chunk word proposals from one complete CTC sentence output."""
from __future__ import annotations

import math
import re

from hanglish.core.phonology import romanize_word


BASES = 4
MAX_PROPOSALS = 96
HANGUL = re.compile(r"[가-힣]")
VERSION = re.compile(r"(?:v|ver)?\d+(?:\.\d+)+(?:[-+][A-Za-z0-9.-]+)?\Z", re.I)
FILENAME = re.compile(r"[A-Za-z][A-Za-z0-9 -]*\.[A-Za-z0-9]{1,8}\Z")
UPPER_CODE = re.compile(r"[A-Z][A-Z0-9.-]*\d[A-Z0-9.-]*\Z")
TRAILING_DIGIT_CODE = re.compile(r"[A-Za-z]+\d+\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _source_chunks(source):
    return [{"index": index, "start": match.start(), "end": match.end(),
             "text": match.group()}
            for index, match in enumerate(re.finditer(r"\S+", source))]


def _output_chunks(text):
    return [{"index": index, "text": match.group()}
            for index, match in enumerate(re.finditer(r"\S+", text))]


def _protected(text):
    if (not text or len(text) > 96 or "`" in text or "_" in text
            or "/" in text or "\\" in text or "@" in text):
        return True
    lowered = text.lower()
    return (lowered.startswith(("http:", "https:", "www."))
            or VERSION.fullmatch(text) is not None
            or FILENAME.fullmatch(text) is not None
            or UPPER_CODE.fullmatch(text) is not None
            or TRAILING_DIGIT_CODE.fullmatch(text) is not None)


def literal_signature(text):
    """Return uppercase, numeric, and punctuation literals to preserve."""
    return "".join(character for character in text
                   if not ("a" <= character <= "z")
                   and HANGUL.fullmatch(character) is None)


def aligned_words(source, generated):
    """Return deterministic source-leftmost, whole-chunk mixed alignments."""
    if not isinstance(source, str) or not isinstance(generated, str):
        return []
    sources = [chunk for chunk in _source_chunks(source)
               if not _protected(chunk["text"])]
    outputs = []
    for chunk in _output_chunks(generated):
        if HANGUL.search(chunk["text"]):
            outputs.append({**chunk, "roman": romanize_word(chunk["text"])})

    def matches(left, right):
        return (left["text"] == right["roman"]
                and literal_signature(left["text"])
                == literal_signature(right["text"]))

    rows, columns = len(sources), len(outputs)
    lengths = [[0] * (columns + 1) for _ in range(rows + 1)]
    for left in range(rows - 1, -1, -1):
        for right in range(columns - 1, -1, -1):
            if matches(sources[left], outputs[right]):
                lengths[left][right] = 1 + lengths[left + 1][right + 1]
            else:
                lengths[left][right] = max(lengths[left + 1][right],
                                           lengths[left][right + 1])
    aligned = []
    left = right = 0
    while left < rows and right < columns:
        if (matches(sources[left], outputs[right])
                and lengths[left][right] == 1 + lengths[left + 1][right + 1]):
            aligned.append({
                "source_chunk_index": sources[left]["index"],
                "output_chunk_index": outputs[right]["index"],
                "input_start": sources[left]["start"],
                "input_end": sources[left]["end"],
                "source": sources[left]["text"],
                "text": outputs[right]["text"],
                "forward_roman": outputs[right]["roman"],
                "literal_signature": literal_signature(sources[left]["text"]),
            })
            left += 1
            right += 1
        elif lengths[left][right + 1] == lengths[left][right]:
            right += 1
        else:
            left += 1
    return aligned


def _validate_path(source, path):
    _require(isinstance(path, tuple) and len(path) == 3
             and isinstance(path[0], str) and _finite_number(path[1])
             and isinstance(path[2], list), "CTC-word parent path is malformed")
    cursor = 0
    texts = []
    for segment in path[2]:
        _require(isinstance(segment, dict)
                 and set(segment) >= {"input_start", "input_end", "source", "text", "kind"},
                 "CTC-word segment fields are missing")
        start, end = segment["input_start"], segment["input_end"]
        _require(isinstance(start, int) and not isinstance(start, bool)
                 and isinstance(end, int) and not isinstance(end, bool)
                 and start == cursor and start < end <= len(source)
                 and segment["source"] == source[start:end]
                 and isinstance(segment["text"], str),
                 "CTC-word segment source coverage differs")
        cursor = end
        texts.append(segment["text"])
    _require(cursor == len(source) and "".join(texts) == path[0],
             "CTC-word path does not cover its source")


def _span_texts(paths, start, end):
    forms = set()
    for path in paths:
        for first, segment in enumerate(path[2]):
            if segment["input_start"] != start:
                continue
            cursor = start
            text = []
            for current in path[2][first:]:
                if current["input_start"] != cursor or current["input_end"] > end:
                    break
                cursor = current["input_end"]
                text.append(current["text"])
                if cursor == end:
                    forms.add("".join(text))
                    break
    return forms


def _replace_spans(source, base, replacements):
    replacements = sorted(replacements, key=lambda item: item["input_start"])
    _require(all(left["input_end"] <= right["input_start"]
                 for left, right in zip(replacements, replacements[1:])),
             "CTC-word replacement spans overlap")
    by_start = {item["input_start"]: item for item in replacements}
    result = []
    index = 0
    while index < len(base[2]):
        segment = base[2][index]
        replacement = by_start.get(segment["input_start"])
        if replacement is None:
            result.append(dict(segment))
            index += 1
            continue
        end = replacement["input_end"]
        scan = index
        position = segment["input_start"]
        while scan < len(base[2]) and position < end:
            current = base[2][scan]
            if current["input_start"] != position or current["input_end"] > end:
                return None
            position = current["input_end"]
            scan += 1
        if position != end:
            return None
        result.append({
            "input_start": replacement["input_start"], "input_end": end,
            "source": source[replacement["input_start"]:end],
            "text": replacement["text"], "kind": "ctc_mixed_word_probe",
        })
        index = scan
    candidate = ("".join(segment["text"] for segment in result), 0.0, result)
    _validate_path(source, candidate)
    return candidate


def sentence_extras(source, generated, paths, limit=MAX_PROPOSALS):
    """Return mixed CTC paths; scores are placeholders for caller ranking."""
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
             "CTC-word parent pool contains duplicates")

    novel = []
    for alignment in aligned_words(source, generated):
        existing = _span_texts(parents, alignment["input_start"],
                               alignment["input_end"])
        if alignment["text"] not in existing:
            novel.append(alignment)
    if not novel:
        return []
    seen = {path[0] for path in parents}
    extras = []

    def add(candidate):
        if candidate is None or candidate[0] in seen or len(extras) >= limit:
            return
        seen.add(candidate[0])
        extras.append(candidate)

    for base in parents[:BASES]:
        add(_replace_spans(source, base, novel))
    for replacement in novel:
        for base in parents[:BASES]:
            add(_replace_spans(source, base, [replacement]))
    return extras


__all__ = ["aligned_words", "literal_signature", "sentence_extras"]
