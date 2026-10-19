"""Bridge candidate character spans to the sentence ranker's tokenization.

Offsets in this module are Python string indices, so they count Unicode code
points rather than UTF-8 bytes or UTF-16 code units.
"""

from __future__ import annotations

from typing import Any


CharacterSpan = tuple[int, int]
TokenSpan = CharacterSpan | None


def normalize_with_offsets(text: str) -> tuple[str, list[CharacterSpan]]:
    """Apply sentence-ranker whitespace normalization and retain source spans.

    A normalized separator maps to the complete original interval between its
    surrounding non-whitespace runs.  This preserves the information lost when
    several whitespace code points are collapsed to one ASCII space.
    """
    if not isinstance(text, str):
        raise TypeError("candidate must be a string")

    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        start = index
        while index < len(text) and not text[index].isspace():
            index += 1
        if start < index:
            runs.append((start, index))

    normalized_parts: list[str] = []
    source_spans: list[CharacterSpan] = []
    for run_index, (start, end) in enumerate(runs):
        if run_index:
            previous_end = runs[run_index - 1][1]
            normalized_parts.append(" ")
            source_spans.append((previous_end, start))
        normalized_parts.append(text[start:end])
        source_spans.extend((offset, offset + 1) for offset in range(start, end))

    normalized = "".join(normalized_parts)
    # Keep this invariant explicit: it is the exact normalization used by the
    # frozen SentenceRanker contract.
    if normalized != " ".join(text.strip().split()):
        raise RuntimeError("candidate whitespace normalization is inconsistent")
    if len(normalized) != len(source_spans):
        raise RuntimeError("candidate normalization produced invalid source spans")
    return normalized, source_spans


def _flat_input_ids(encoded: Any) -> list[int]:
    if not isinstance(encoded, dict) and not hasattr(encoded, "__getitem__"):
        raise TypeError("tokenizer returned an invalid encoding")
    try:
        input_ids = encoded["input_ids"]
    except (KeyError, TypeError) as error:
        raise ValueError("tokenizer encoding is missing input_ids") from error
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if not isinstance(input_ids, list):
        raise ValueError("tokenizer returned malformed input_ids")
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("single-candidate tokenization returned multiple rows")
        input_ids = input_ids[0]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in input_ids):
        raise ValueError("tokenizer returned malformed input_ids")
    return input_ids


def _rows(value: Any, field: str) -> list[list[Any]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or any(not isinstance(row, list) for row in value):
        raise ValueError(f"tokenizer returned malformed {field}")
    return value


def _validate_offset(value: Any, normalized_length: int) -> CharacterSpan:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("tokenizer returned a malformed offset")
    start, end = value
    if (isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int)):
        raise ValueError("tokenizer offsets must be integer pairs")
    if start < 0 or end < start or end > normalized_length:
        raise ValueError("tokenizer returned an out-of-range offset")
    return start, end


def encode_aligned_candidates(tokenizer, candidates: list[str], max_length: int):
    """Encode complete candidates and map sequence-A tokens to source spans.

    The token IDs are produced with the same normalized sequence-A / empty
    sequence-B call as ``encode_sentence_pairs``.  Offsets are removed from the
    returned encoding so it can be passed directly to the ranker model.
    """
    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list")
    if any(not isinstance(candidate, str) for candidate in candidates):
        raise TypeError("every candidate must be a string")
    if (isinstance(max_length, bool) or not isinstance(max_length, int)
            or not 4 <= max_length <= 512):
        raise ValueError("max_length must be an integer from 4 to 512")
    if getattr(tokenizer, "is_fast", False) is not True:
        raise ValueError("aligned candidate encoding requires a fast tokenizer")

    normalized_with_maps = [normalize_with_offsets(candidate) for candidate in candidates]
    normalized = [item[0] for item in normalized_with_maps]
    character_maps = [item[1] for item in normalized_with_maps]

    reserved = tokenizer.num_special_tokens_to_add(pair=True)
    if (isinstance(reserved, bool) or not isinstance(reserved, int)
            or reserved < 0 or reserved > max_length):
        raise ValueError("tokenizer returned an invalid pair-special-token budget")
    for index, candidate in enumerate(normalized):
        token_ids = _flat_input_ids(tokenizer(
            candidate, add_special_tokens=False, truncation=False))
        if len(token_ids) + reserved > max_length:
            raise ValueError(
                f"candidate at index {index} exceeds the full-sentence token budget")

    encoded = tokenizer(
        normalized, [""] * len(normalized), padding=True,
        truncation="only_second", max_length=max_length,
        return_tensors="pt", return_offsets_mapping=True,
    )
    if not hasattr(encoded, "sequence_ids") or not callable(encoded.sequence_ids):
        raise ValueError("fast tokenizer encoding does not expose sequence_ids")
    try:
        offsets_value = encoded.pop("offset_mapping")
    except (KeyError, TypeError) as error:
        raise ValueError("fast tokenizer did not return offset_mapping") from error
    offset_rows = _rows(offsets_value, "offset_mapping")
    try:
        input_rows = _rows(encoded["input_ids"], "input_ids")
    except (KeyError, TypeError) as error:
        raise ValueError("tokenizer encoding is missing input_ids") from error
    if len(offset_rows) != len(candidates) or len(input_rows) != len(candidates):
        raise ValueError("tokenizer returned an invalid batch size")

    token_spans: list[list[TokenSpan]] = []
    for batch_index, (offset_row, input_row, source_map) in enumerate(
            zip(offset_rows, input_rows, character_maps)):
        sequence_ids = encoded.sequence_ids(batch_index)
        if not isinstance(sequence_ids, list):
            sequence_ids = list(sequence_ids)
        if len(offset_row) != len(input_row) or len(sequence_ids) != len(input_row):
            raise ValueError("tokenizer token metadata has inconsistent lengths")
        spans: list[TokenSpan] = []
        for sequence_id, raw_offset in zip(sequence_ids, offset_row):
            if (sequence_id is not None
                    and (isinstance(sequence_id, bool) or sequence_id not in (0, 1))):
                raise ValueError("tokenizer returned an invalid sequence id")
            start, end = _validate_offset(raw_offset, len(source_map))
            if sequence_id != 0:
                spans.append(None)
                continue
            if start == end:
                raise ValueError("candidate token has an empty offset")
            spans.append((source_map[start][0], source_map[end - 1][1]))
        token_spans.append(spans)
    return encoded, token_spans


def pooling_weights(
        token_spans: list[TokenSpan], start: int, end: int) -> list[float]:
    """Return code-point overlap lengths for one candidate output span."""
    if not isinstance(token_spans, list):
        raise TypeError("token_spans must be a list")
    if (isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int)):
        raise TypeError("pooling offsets must be integers")
    if start < 0 or end < start:
        raise ValueError("pooling offsets are out of range")

    weights: list[float] = []
    for span in token_spans:
        if span is None:
            weights.append(0.0)
            continue
        if not isinstance(span, tuple) or len(span) != 2:
            raise ValueError("token_spans contains a malformed span")
        token_start, token_end = span
        if (isinstance(token_start, bool) or isinstance(token_end, bool)
                or not isinstance(token_start, int) or not isinstance(token_end, int)
                or token_start < 0 or token_end < token_start):
            raise ValueError("token_spans contains an out-of-range span")
        weights.append(float(max(0, min(end, token_end) - max(start, token_start))))
    return weights


__all__ = ["encode_aligned_candidates", "normalize_with_offsets", "pooling_weights"]
