"""Bounded source-only proposals for opaque chunks containing digits."""
from __future__ import annotations

import math
import re

from hanglish.core.lattice import WordOption
from hanglish.core.phonology import romanize_word


ATOMS = re.compile(r"[A-Za-z]+|[^A-Za-z]+")
HANGUL = re.compile(r"[가-힣]")
VERSION = re.compile(r"(?:v|ver)?\d+(?:\.\d+)+(?:[-+][A-Za-z0-9.-]+)?\Z", re.I)
FILENAME = re.compile(r"[A-Za-z][A-Za-z0-9 -]*\.[A-Za-z0-9]{1,8}\Z")
UPPER_CODE = re.compile(r"[A-Z][A-Z0-9.-]*\d[A-Z0-9.-]*\Z")
TRAILING_DIGIT_CODE = re.compile(r"[A-Za-z]+\d+\Z")


def _finite_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _numeric_units():
    # NeuralConverter may import this module in the future.  Resolve its shared
    # constant only when proposal generation runs, after module initialization.
    from hanglish.core.neural_converter import ASCII_NUMERIC_UNITS
    return ASCII_NUMERIC_UNITS | {"t"}


def _protected(source, atoms):
    if (len(source) > 96 or len(atoms) > 24 or not any(char.isdigit() for char in source)
            or "`" in source or "_" in source or "/" in source or "\\" in source
            or "@" in source):
        return True
    lowered = source.lower()
    return (lowered.startswith(("http:", "https:", "www."))
            or VERSION.fullmatch(source) is not None
            or FILENAME.fullmatch(source) is not None
            or UPPER_CODE.fullmatch(source) is not None
            or TRAILING_DIGIT_CODE.fullmatch(source) is not None)


def _option_records(source, word_options):
    if source.isupper() or source.lower() in _numeric_units():
        return [WordOption(source, 0.0, "raw")]
    values = list(word_options(source, learned=True))
    values.append(WordOption(source, 0.0, "raw"))
    strongest = {}
    for option in values:
        if (not isinstance(getattr(option, "text", None), str)
                or not option.text or isinstance(getattr(option, "score", None), bool)
                or not isinstance(option.score, (int, float))
                or not math.isfinite(option.score)
                or not isinstance(getattr(option, "kind", None), str)):
            continue
        forward = romanize_word(option.text) == source
        converted = option.kind != "raw" and HANGUL.search(option.text) is not None
        raw = option.text == source and option.kind == "raw"
        if not forward or not (converted or raw):
            continue
        previous = strongest.get(option.text)
        if previous is None or option.score > previous.score:
            strongest[option.text] = option
    return sorted(strongest.values(), key=lambda option: (-option.score, option.text))


def _segment(source, start, text, kind):
    return {
        "input_start": start,
        "input_end": start + len(source),
        "source": source,
        "text": text,
        "kind": kind,
    }


def chunk_paths(source, offset, word_options):
    """Return at most four changed, forward-valid paths for one mixed chunk."""
    if (not isinstance(source, str) or not isinstance(offset, int)
            or isinstance(offset, bool) or offset < 0):
        raise TypeError("source must be a string and offset must be an integer")
    atoms = [(match.group(), match.start()) for match in ATOMS.finditer(source)]
    if not source or _protected(source, atoms):
        return []
    paths = [("", 0.0, [], False)]
    for atom, relative in atoms:
        if atom.isascii() and atom.isalpha():
            options = _option_records(atom, word_options)
        else:
            options = [WordOption(atom, 0.0, "raw")]
        expanded = []
        for prefix, score, segments, changed in paths:
            for option in options:
                expanded.append((
                    prefix + option.text,
                    score + float(option.score),
                    segments + [_segment(atom, offset + relative,
                                         option.text, option.kind)],
                    changed or option.text != atom,
                ))
        unique = {}
        for path in sorted(expanded, key=lambda item: (-item[1], item[0])):
            unique.setdefault(path[0], path)
        paths = list(unique.values())[:8]
    result = []
    for text, score, segments, changed in paths:
        if not changed or text == source:
            continue
        if romanize_word(text) != source:
            raise RuntimeError("mixed chunk output is not whole-chunk forward-valid")
        if text != "".join(segment["text"] for segment in segments):
            raise RuntimeError("mixed chunk output differs from its segments")
        cursor = offset
        for segment in segments:
            if (segment["input_start"] != cursor
                    or segment["input_end"] <= cursor
                    or source[segment["input_start"] - offset:
                              segment["input_end"] - offset] != segment["source"]
                    or (segment["kind"] == "raw"
                        and segment["text"] != segment["source"])
                    or (segment["kind"] != "raw"
                        and romanize_word(segment["text"]) != segment["source"])):
                raise RuntimeError("mixed chunk path is not forward-valid and contiguous")
            cursor = segment["input_end"]
        if cursor != offset + len(source):
            raise RuntimeError("mixed chunk path does not cover its source")
        result.append((text, score, segments))
        if len(result) == 4:
            break
    return result


def _parent_variants(source, parent, word_options, chunk_cache):
    text, parent_score, parent_segments = parent
    replacements = {}
    for index, segment in enumerate(parent_segments):
        chunk = segment.get("source")
        if (segment.get("kind") == "raw" and segment.get("text") == chunk
                and isinstance(chunk, str) and any(char.isdigit() for char in chunk)
                and isinstance(segment.get("input_start"), int)):
            key = (chunk, segment["input_start"])
            if key not in chunk_cache:
                chunk_cache[key] = chunk_paths(chunk, segment["input_start"], word_options)
            paths = chunk_cache[key]
            if paths:
                replacements[index] = paths
    if not replacements:
        return []

    best_segments = []
    for index, segment in enumerate(parent_segments):
        if index in replacements:
            best_segments.extend(dict(item) for item in replacements[index][0][2])
        else:
            best_segments.append(dict(segment))
    best_text = "".join(segment["text"] for segment in best_segments)
    variants = [(best_text, parent_score, best_segments)]

    states = [("", 0.0, [], False)]
    for index, segment in enumerate(parent_segments):
        choices = [(segment["text"], 0.0, [dict(segment)], False)]
        choices.extend((path[0], path[1], [dict(item) for item in path[2]], True)
                       for path in replacements.get(index, ()))
        expanded = []
        for prefix, delta, segments, changed in states:
            for replacement_text, replacement_score, replacement_segments, replaced in choices:
                expanded.append((prefix + replacement_text,
                                 delta + replacement_score,
                                 segments + replacement_segments,
                                 changed or replaced))
        unique = {}
        for state in sorted(expanded, key=lambda item: (-item[1], item[0])):
            unique.setdefault(state[0], state)
        states = list(unique.values())[:8]
    for variant_text, _, segments, changed in states:
        if changed:
            variants.append((variant_text, parent_score, segments))
    unique = {}
    for variant in variants:
        if variant[0] != text:
            unique.setdefault(variant[0], variant)
    return list(unique.values())


def sentence_extras(source, parent_paths, word_options, limit=96):
    """Apply mixed-span alternatives to raw digit segments of bounded parents."""
    if (not isinstance(source, str) or isinstance(limit, bool)
            or not isinstance(limit, int) or not 1 <= limit <= 96):
        raise ValueError("source and a limit from 1 through 96 are required")
    parents = list(parent_paths)
    if len(parents) > 96:
        raise ValueError("mixed-span parent pool exceeds 96 paths")
    per_parent = []
    parent_texts = set()
    chunk_cache = {}
    for parent in parents:
        if (not isinstance(parent, tuple) or len(parent) != 3
                or not isinstance(parent[0], str) or not _finite_number(parent[1])
                or not isinstance(parent[2], list)):
            raise ValueError("mixed-span parent path is malformed")
        parent_texts.add(parent[0])
        cursor = 0
        for segment in parent[2]:
            start, end = segment.get("input_start"), segment.get("input_end")
            if (not isinstance(start, int) or not isinstance(end, int)
                    or start != cursor or end <= start or end > len(source)
                    or source[start:end] != segment.get("source")):
                raise ValueError("parent segments do not preserve source offsets")
            cursor = end
        if (cursor != len(source)
                or parent[0] != "".join(segment.get("text", "") for segment in parent[2])):
            raise ValueError("parent path does not cover source or match its segments")
        per_parent.append(_parent_variants(source, parent, word_options, chunk_cache))

    seen = set(parent_texts)
    extras = []
    # Retain one strongest converted variant from every distinct parent before
    # allowing any parent's lower-ranked alternatives to consume the budget.
    for variants in per_parent:
        if variants and variants[0][0] not in seen:
            seen.add(variants[0][0])
            extras.append(variants[0])
            if len(extras) == limit:
                return extras
    for variants in per_parent:
        for variant in variants[1:]:
            if variant[0] in seen:
                continue
            seen.add(variant[0])
            extras.append(variant)
            if len(extras) == limit:
                return extras
    return extras


__all__ = ["chunk_paths", "sentence_extras"]
