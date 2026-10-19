"""Source-only pair construction for difference-pairwise LoRA inference.

The frozen training builder also returns a supervised KEEP/SWITCH target and
therefore expects an ``expected`` field.  Inference must never open that field.
This module copies only the explicit source-field whitelist, supplies a
constant empty tuple solely to satisfy the frozen builder interface, and
discards every returned target.  Pair construction and compact prompts thus
remain owned by the frozen training implementation without exposing labels.
"""
from __future__ import annotations

from collections.abc import Mapping

from scripts.difference_pairwise_lora_evaluation import canonical, prepare_selection
from scripts.final_choice_policy import finite_candidate_indices
from scripts.prepare_difference_pairwise_lora_data import (
    MAX_PROMPT_TOKENS,
    permitted_pairs,
    tokenize_pair,
)


SOURCE_FIELDS = (
    "id",
    "group_id",
    "input",
    "context",
    "candidates",
    "raw_bypass_response",
)
EXPECTED_DIGIT_CONTINUATION_IDS = {"1": 16, "2": 17}


def _source_only_row(row: Mapping) -> dict:
    """Project an inference row without iterating over or copying label fields."""
    if not isinstance(row, Mapping):
        raise ValueError("inference row must be a mapping")
    try:
        row_id = row["id"]
        group_id = row["group_id"]
        source = row["input"]
        context = row["context"]
        candidates = row["candidates"]
        raw = row["raw_bypass_response"]
    except (KeyError, TypeError) as error:
        raise ValueError("inference source fields are missing") from error
    if (
        not isinstance(row_id, str)
        or not row_id
        or not isinstance(group_id, str)
        or not group_id
        or not isinstance(source, str)
        or not isinstance(context, str)
        or not isinstance(candidates, list)
        or not candidates
        or any(not isinstance(candidate, dict) for candidate in candidates)
        or not isinstance(raw, dict)
    ):
        raise ValueError("inference source fields are malformed")
    return {
        "id": row_id,
        "group_id": group_id,
        "input": source,
        "context": context,
        "candidates": candidates,
        "raw_bypass_response": raw,
        # Compatibility shim for build_pair only.  It is constant, never read
        # from the supplied row, and its derived targets are discarded below.
        "expected": (),
    }


def prepare_inference_selection(
    role: str, row: Mapping, *, raw_mode: bool = False
) -> dict:
    """Build every permitted source-only pair and bind it for safe resolution.

    Candidate and raw-response objects are retained by identity.  A raw-mode
    row and a row whose parent score is ``None`` produce no pairs.  Invalid
    non-finite scores remain hard failures under the frozen candidate policy.
    """
    if not isinstance(role, str) or not role or type(raw_mode) is not bool:
        raise ValueError("role and raw mode must be explicit")
    source_row = _source_only_row(row)
    if raw_mode:
        pairs = []
    else:
        finite = finite_candidate_indices(source_row["candidates"])
        pairs = [] if not finite or finite[0] != 0 else [
            pair for pair, _discarded_target in permitted_pairs(role, source_row)
        ]
    selection = prepare_selection(role, source_row, pairs, raw_mode=raw_mode)
    selection["group_id"] = source_row["group_id"]
    return selection


def _rebuild_selection(selection: Mapping) -> dict:
    """Rebuild a selection from its source boundary before tokenization."""
    try:
        role = selection["role"]
        row = {
            "id": selection["row_id"],
            "group_id": selection["group_id"],
            "input": selection["source"],
            "context": selection["context"],
            "candidates": selection["candidates"],
            "raw_bypass_response": selection["raw_bypass_response"],
        }
        raw_mode = selection["raw_mode"]
        pairs = selection["pairs"]
        alternative_indices = selection["alternative_indices"]
        candidate_snapshot = selection["candidate_snapshot"]
        raw_snapshot = selection["raw_snapshot"]
    except (KeyError, TypeError) as error:
        raise ValueError("prepared inference selection fields are missing") from error
    if (
        not isinstance(pairs, list)
        or not isinstance(alternative_indices, list)
        or not isinstance(candidate_snapshot, bytes)
        or not isinstance(raw_snapshot, bytes)
    ):
        raise ValueError("prepared inference selection fields are malformed")
    rebuilt = prepare_inference_selection(role, row, raw_mode=raw_mode)
    try:
        matches = (
            canonical(pairs) == canonical(rebuilt["pairs"])
            and canonical(alternative_indices) == canonical(rebuilt["alternative_indices"])
            and candidate_snapshot == rebuilt["candidate_snapshot"]
            and raw_snapshot == rebuilt["raw_snapshot"]
        )
    except (TypeError, ValueError) as error:
        raise ValueError("prepared inference selection evidence is malformed") from error
    if not matches:
        raise ValueError("prepared inference selection differs from rebuilt source evidence")
    return rebuilt


def qualify_inference_tokens(tokenizer, selection: Mapping) -> list[dict]:
    """Tokenize every prepared pair under the frozen 1/2 prompt contract.

    Oversized prompts fail as a whole; this helper never skips or truncates a
    pair.  It performs no model initialization or inference.
    """
    if not isinstance(selection, Mapping):
        raise ValueError("prepared inference selection is malformed")
    rebuilt = _rebuild_selection(selection)
    qualified = []
    for pair in rebuilt["pairs"]:
        messages, tokens, digit_ids, rendered = tokenize_pair(tokenizer, pair)
        if len(tokens) > MAX_PROMPT_TOKENS:
            raise ValueError(
                f"prompt exceeds {MAX_PROMPT_TOKENS} tokens for pair {pair.get('pair_id')}"
            )
        if digit_ids != EXPECTED_DIGIT_CONTINUATION_IDS:
            raise ValueError("answer digit continuation IDs differ from the frozen contract")
        qualified.append(
            {
                "pair_id": pair["pair_id"],
                "messages": messages,
                "tokens": tokens,
                "prompt_tokens": len(tokens),
                "digit_continuation_ids": digit_ids,
                "rendered_prefix": rendered,
            }
        )
    return qualified
