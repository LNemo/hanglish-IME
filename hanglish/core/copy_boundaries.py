"""Source-only provenance for COPY spans inside mixed candidates.

This module refines alignment metadata only.  It does not create a candidate,
change a score, or decide that an automatically recovered COPY span is a user
commitment.
"""
from __future__ import annotations

from copy import deepcopy


_SEGMENT_FIELDS = ("input_start", "input_end", "source", "text", "kind")


def _segment(row, *, start=None, end=None, source=None, text=None, kind=None):
    """Return the five-field segment shape accepted by current consumers."""
    return {
        "input_start": row["input_start"] if start is None else start,
        "input_end": row["input_end"] if end is None else end,
        "source": row["source"] if source is None else source,
        "text": row["text"] if text is None else text,
        "kind": row["kind"] if kind is None else kind,
    }


def _validate(source, text, segments):
    if not isinstance(source, str) or not isinstance(text, str):
        raise TypeError("source and text must be strings")
    if not isinstance(segments, list):
        raise TypeError("segments must be a list")
    cursor = 0
    rendered = []
    output_cursor = 0
    output_offsets = []
    for row in segments:
        if not isinstance(row, dict):
            raise ValueError("segment must be a mapping")
        if any(field not in row for field in _SEGMENT_FIELDS):
            raise ValueError("segment is missing an alignment field")
        start, end = row["input_start"], row["input_end"]
        if (type(start) is not int or type(end) is not int
                or start != cursor or not start < end <= len(source)
                or row["source"] != source[start:end]
                or not isinstance(row["text"], str)
                or not isinstance(row["kind"], str)):
            raise ValueError("segments do not preserve source offsets")
        rendered.append(row["text"])
        output_offsets.append((output_cursor, output_cursor + len(row["text"])))
        output_cursor += len(row["text"])
        cursor = end
    if cursor != len(source) or "".join(rendered) != text:
        raise ValueError("segments do not cover the source and output exactly")
    return output_offsets


def _admitted_mixed(generator, source, target):
    try:
        options = generator.options(source)
    except (AttributeError, TypeError) as exc:
        raise TypeError("generator must provide options(source)") from exc
    try:
        return any(getattr(option, "text", None) == target
                   and getattr(option, "kind", None) == "mixed"
                   for option in options)
    except TypeError as exc:
        raise TypeError("generator.options(source) must be iterable") from exc


def _inventory(generator):
    try:
        entries = tuple(generator.english_suffixes)
    except (AttributeError, TypeError) as exc:
        raise TypeError("generator must provide iterable english_suffixes") from exc
    validated = []
    for index, entry in enumerate(entries):
        if (not isinstance(entry, (tuple, list)) or len(entry) != 2
                or not all(isinstance(value, str) and value for value in entry)):
            raise ValueError("english suffix inventory entry is malformed")
        validated.append((index, entry[0], entry[1]))
    return validated


def _matches(segment, inventory):
    source, target = segment["source"], segment["text"]
    lowered = source.lower()
    matches = {}
    for inventory_index, ending, hangul in inventory:
        if not lowered.endswith(ending) or len(source) <= len(ending) + 1:
            continue
        prefix = source[:-len(ending)]
        if target != prefix + hangul:
            continue
        # Duplicate inventory rows describe one derivation, not ambiguity.
        key = (len(prefix), ending, hangul)
        matches.setdefault(key, {
            "inventory_index": inventory_index,
            "ending": ending,
            "hangul": hangul,
            "prefix": prefix,
        })
    return list(matches.values())


def refine_path(source: str, text: str, segments: list, *, generator) -> dict:
    """Refine uniquely proven mixed English+particle edges into two children.

    Exact membership in ``generator.options`` is the eligibility authority.
    The suffix inventory is only replayed to recover provenance for an already
    admitted mixed option.  Ambiguous or unavailable provenance stays coarse.
    """
    output_offsets = _validate(source, text, segments)
    inventory = _inventory(generator)
    coarse = deepcopy(segments)
    refined_segments = []
    derivations = []
    copy_spans = []

    for parent_index, (row, (output_start, output_end)) in enumerate(
            zip(segments, output_offsets)):
        base = {
            "parent_index": parent_index,
            "source_boundary": [row["input_start"], row["input_end"]],
            "output_boundary": [output_start, output_end],
            "source": row["source"],
            "text": row["text"],
            "kind": row["kind"],
        }
        if row["source"] == row["text"]:
            child = _segment(row)
            refined_segments.append(child)
            derivations.append({**base, "status": "already_copy",
                                "children": [dict(child)]})
            copy_spans.append({
                "parent_index": parent_index,
                "source_boundary": [row["input_start"], row["input_end"]],
                "output_boundary": [output_start, output_end],
                "source": row["source"],
                "text": row["text"],
                "origin": "existing_identity",
                "automatic_hypothesis": False,
                "user_committed": False,
            })
            continue

        if row["kind"] != "mixed":
            child = _segment(row)
            refined_segments.append(child)
            derivations.append({**base, "status": "not_applicable",
                                "reason": "parent_kind_is_not_mixed",
                                "children": [dict(child)]})
            continue

        if not _admitted_mixed(generator, row["source"], row["text"]):
            child = _segment(row)
            refined_segments.append(child)
            derivations.append({**base, "status": "unresolved",
                                "reason": "generator_option_not_admitted",
                                "children": [dict(child)]})
            continue

        matches = _matches(row, inventory)
        if len(matches) != 1:
            child = _segment(row)
            refined_segments.append(child)
            derivations.append({
                **base,
                "status": "unresolved",
                "reason": ("no_exact_inventory_derivation" if not matches
                           else "ambiguous_inventory_derivation"),
                "matching_derivations": deepcopy(matches),
                "children": [dict(child)],
            })
            continue

        match = matches[0]
        prefix, ending, hangul = match["prefix"], match["ending"], match["hangul"]
        source_suffix = row["source"][len(prefix):]
        split = row["input_start"] + len(prefix)
        output_split = output_start + len(prefix)
        rule_id = (f"english_suffix:{match['inventory_index']}:"
                   f"{ending}->{hangul}")
        copy_child = _segment(row, end=split, source=prefix, text=prefix,
                              kind="raw")
        convert_child = _segment(row, start=split, source=source_suffix, text=hangul,
                                 kind="mixed")
        refined_segments.extend((copy_child, convert_child))
        derivations.append({
            **base,
            "status": "refined",
            "rule_id": rule_id,
            "generator_evidence": {"source": row["source"],
                                   "text": row["text"], "kind": "mixed"},
            "suffix_inventory": {
                "inventory_index": match["inventory_index"],
                "roman": ending,
                "hangul": hangul,
            },
            "children": [
                {"action": "copy",
                 "source_boundary": [row["input_start"], split],
                 "output_boundary": [output_start, output_split],
                 **dict(copy_child)},
                {"action": "convert",
                 "source_boundary": [split, row["input_end"]],
                 "output_boundary": [output_split, output_end],
                 **dict(convert_child)},
            ],
        })
        copy_spans.append({
            "parent_index": parent_index,
            "rule_id": rule_id,
            "source_boundary": [row["input_start"], split],
            "output_boundary": [output_start, output_split],
            "source": prefix,
            "text": prefix,
            "origin": "generator_admitted_suffix_derivation",
            "automatic_hypothesis": True,
            "user_committed": False,
        })

    try:
        _validate(source, text, refined_segments)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("refined segments changed the path") from exc
    return {
        "source": source,
        "text": text,
        "coarse_segments": coarse,
        "segments": refined_segments,
        "derivations": derivations,
        "copy_spans": copy_spans,
        "refined": any(row["status"] == "refined" for row in derivations),
    }


__all__ = ["refine_path"]
