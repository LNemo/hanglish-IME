"""Source-only numeric counter plus lexical-tail candidate composition."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import re

from hanglish.core.neural_converter import NUMERIC_UNITS, numeric_unit_surface
from hanglish.core.phonology import romanize_word


NUMERIC_SUFFIX = re.compile(r"(\d+(?:\.\d+)?)([a-z]{1,40})\Z")
HANGUL = re.compile(r"[가-힣]+\Z")
LEXICAL_SCORE = 5.5
MAX_PROPOSALS = 4
MAX_ELIGIBLE_PIECES = 4
MAX_COMBINATIONS = 16
MAX_NEW_PATHS = 48


@dataclass(frozen=True)
class CompositionSupplement:
    """New paths and bounded-enumeration facts for later runtime wiring."""

    paths: tuple
    eligible_piece_count: int
    retained_combination_count: int
    bypassed: bool


def numeric_suffix_proposals(source, generator, numeric_units=NUMERIC_UNITS):
    """Return at most four source/resource-derived full-token proposals."""
    match = NUMERIC_SUFFIX.fullmatch(source)
    if match is None:
        return ()
    number, suffix = match.groups()
    proposals = set()
    for split in range(1, len(suffix)):
        left_source, right_source = suffix[:split], suffix[split:]
        for left in generator.options(left_source):
            if (left.text == left_source
                    or not numeric_unit_surface(left.text, numeric_units)):
                continue
            for right in generator.options(right_source):
                if (right.kind == "raw" or right.text == right_source
                        or right.score < LEXICAL_SCORE):
                    continue
                combined = left.text + right.text
                if HANGUL.fullmatch(combined) and romanize_word(combined) == suffix:
                    proposals.add(number + combined)
    return tuple(sorted(proposals)[:MAX_PROPOSALS])


def _raw_only_aligned(learned, static):
    source, offset, options = learned
    if static[:2] != (source, offset) or NUMERIC_SUFFIX.fullmatch(source) is None:
        return False
    combined = (*options, *static[2])
    return bool(combined) and all(
        option.kind == "raw" and option.text == source for option in combined
    )


def _replacement(source, offset, target):
    match = NUMERIC_SUFFIX.fullmatch(source)
    if match is None or not target.startswith(match[1]):
        raise ValueError("numeric composition target no longer matches its source")
    number, suffix = match.groups()
    hangul = target[len(number):]
    if not HANGUL.fullmatch(hangul) or romanize_word(hangul) != suffix:
        raise ValueError("numeric composition target is not forward valid")
    return (
        {
            "input_start": offset,
            "input_end": offset + len(number),
            "source": number,
            "text": number,
            "kind": "raw",
        },
        {
            "input_start": offset + len(number),
            "input_end": offset + len(source),
            "source": suffix,
            "text": hangul,
            "kind": "numeric",
        },
    )


def _validate_pieces(pieces, static_pieces):
    if len(pieces) != len(static_pieces):
        raise ValueError("numeric learned/static piece counts differ")
    cursor = 0
    for learned, static in zip(pieces, static_pieces):
        source, offset, _ = learned
        if offset != cursor or static[:2] != (source, offset):
            raise ValueError("numeric learned/static source alignment differs")
        cursor += len(source)


def _validate_parent(pieces, parent):
    text, score, segments = parent
    if (not isinstance(text, str) or type(score) not in (int, float)
            or not math.isfinite(score) or len(segments) != len(pieces)):
        raise ValueError("numeric composition parent path shape differs")
    for (source, offset, _), item in zip(pieces, segments):
        if (item.get("input_start") != offset
                or item.get("input_end") != offset + len(source)
                or item.get("source") != source):
            raise ValueError("numeric composition parent offsets differ")
        if item.get("kind") == "raw" and item.get("text") != source:
            raise ValueError("numeric composition parent changed a raw segment")
    if text != "".join(item.get("text", "") for item in segments):
        raise ValueError("numeric composition parent output differs")


def composition_extras(
    pieces,
    static_pieces,
    parent_paths,
    preserved_pool,
    generator,
    numeric_units=NUMERIC_UNITS,
):
    """Build bounded full paths without mutating or evicting preserved paths.

    ``parent_paths`` are the original one-segment-per-piece sentence paths.
    ``preserved_pool`` is the already ordered base plus old numeric-extra pool
    used only for whole-output deduplication.  The returned paths are additions.
    """
    _validate_pieces(pieces, static_pieces)
    eligible = [
        index for index, pair in enumerate(zip(pieces, static_pieces))
        if _raw_only_aligned(*pair)
    ]
    if len(eligible) > MAX_ELIGIBLE_PIECES:
        return CompositionSupplement((), len(eligible), 0, True)

    choices = []
    replacements = {}
    for index in eligible:
        source, offset, _ = pieces[index]
        values = numeric_suffix_proposals(source, generator, numeric_units)
        choices.append((source, *values))
        replacements[index] = {
            value: _replacement(source, offset, value) for value in values
        }

    combinations = set()
    for selected in itertools.product(*choices) if choices else ():
        if all(value == pieces[index][0] for index, value in zip(eligible, selected)):
            continue
        combinations.add(tuple(selected))
    retained = sorted(combinations)[:MAX_COMBINATIONS]

    seen = {path[0] for path in preserved_pool}
    extras = []
    for parent in parent_paths:
        text, score, segments = parent
        _validate_parent(pieces, parent)
        for selected in retained:
            selected_by_index = dict(zip(eligible, selected))
            output_segments = []
            for index, segment in enumerate(segments):
                value = selected_by_index.get(index, pieces[index][0])
                if value == pieces[index][0]:
                    output_segments.append(dict(segment))
                    continue
                if (segment["source"] != pieces[index][0]
                        or segment["text"] != segment["source"]
                        or segment["kind"] != "raw"):
                    raise ValueError("numeric composition selected a non-raw parent segment")
                output_segments.extend(dict(item) for item in replacements[index][value])
            output = "".join(segment["text"] for segment in output_segments)
            if output in seen:
                continue
            seen.add(output)
            extras.append((output, score, output_segments))
            if len(extras) == MAX_NEW_PATHS:
                return CompositionSupplement(
                    tuple(extras), len(eligible), len(retained), False
                )
    return CompositionSupplement(tuple(extras), len(eligible), len(retained), False)
