"""Batched active-beam decoder for the frozen phonetic transducer.

This module changes only how active beams are evaluated.  Search width,
allowed-token phases, stable ranking, completion handling, and final
forward-validation match :meth:`PhoneticTransducer.generate`.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable


ROMAN_RE = re.compile(r"[a-z]+\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def beam_search(
    initial_state,
    *,
    bos_id: int,
    eos_id: int,
    search_width: int,
    max_steps: int,
    expand_active: Callable[[list[tuple[list[int], float, object]]],
                            list[list[tuple[int, float, object]]]],
) -> list[tuple[list[int], float]]:
    """Run the original stable beam bookkeeping around a batched expander.

    ``expand_active`` is invoked once per step and must return one ordered child
    list per active parent.  Child ordering therefore remains the backend's
    top-k ordering, while Python's stable sort preserves the original tie rule.
    """
    require(type(search_width) is int and search_width > 0, "invalid search width")
    require(type(max_steps) is int and max_steps > 0, "invalid maximum steps")
    beams = [([bos_id], 0.0, initial_state)]
    completed: list[tuple[list[int], float]] = []
    for _ in range(max_steps):
        active = []
        for tokens, score, state in beams:
            if tokens[-1] == eos_id:
                completed.append((tokens, score))
            else:
                active.append((tokens, score, state))
        if not active:
            break
        children = expand_active(active)
        require(isinstance(children, list) and len(children) == len(active),
                "batched expansion does not match active beams")
        expanded = []
        for (tokens, score, _), parent_children in zip(active, children):
            require(isinstance(parent_children, list), "batched children are malformed")
            for token_id, log_probability, next_state in parent_children:
                require(type(token_id) is int, "expanded token id is invalid")
                require(isinstance(log_probability, float),
                        "expanded log probability is invalid")
                expanded.append((tokens + [token_id], score + log_probability, next_state))
        if not expanded:
            break
        beams = sorted(expanded, key=lambda item: item[1], reverse=True)[:search_width]
    completed.extend((tokens, score) for tokens, score, _ in beams
                     if tokens[-1] == eos_id)
    return completed


def rank_completed(
    completed: Iterable[tuple[list[int], float]],
    roman: str,
    beam_size: int,
    *,
    output_tokens: list[str],
    compose: Callable[[list[str]], str | None],
    romanize: Callable[[str], str],
) -> list[tuple[str, float]]:
    """Apply the original completion order, deduplication, and forward check."""
    valid: dict[str, float] = {}
    for token_ids, score in sorted(completed, key=lambda item: item[1], reverse=True):
        tokens = [output_tokens[index] for index in token_ids[1:-1]]
        hangul = compose(tokens)
        if hangul is not None and romanize(hangul) == roman:
            valid[hangul] = max(score, valid.get(hangul, float("-inf")))
        if len(valid) >= beam_size:
            break
    return sorted(valid.items(), key=lambda item: item[1], reverse=True)


def generate_batched(transducer, roman: str, beam_size: int = 4) -> list[tuple[str, float]]:
    """Generate with one decoder call for all active beams at each search step."""
    if not isinstance(roman, str) or not ROMAN_RE.fullmatch(roman):
        raise ValueError("roman must contain lowercase ASCII letters only")
    if len(roman) > transducer.config.max_roman_length:
        raise ValueError(f"roman must be at most {transducer.config.max_roman_length} letters")
    if not isinstance(beam_size, int) or isinstance(beam_size, bool) or not 1 <= beam_size <= 16:
        raise ValueError("beam_size must be an integer from 1 to 16")

    from hanglish.core.phonology import romanize_word
    from hanglish.models.phonetic_transducer import (
        BOS, EOS, INPUT_TO_ID, OUTPUT_TO_ID, OUTPUT_TOKENS,
        _allowed_output_ids, jamo_to_hangul,
    )
    import torch

    source = torch.tensor(
        [[INPUT_TO_ID[BOS], *(INPUT_TO_ID.get(char, INPUT_TO_ID["<unk>"])
                             for char in roman), INPUT_TO_ID[EOS]]],
        dtype=torch.long,
        device=transducer.device,
    )
    search_width = min(len(OUTPUT_TOKENS), max(8, beam_size * 4))
    max_steps = min(3 * len(roman) + 4, 3 * 24 + 1)

    with torch.inference_mode():
        encoded, initial, mask = transducer.network.encode(source)

        def expand_active(active):
            lengths = {len(tokens) for tokens, _, _ in active}
            require(len(lengths) == 1, "active beam token depths differ")
            token_batch = torch.tensor(
                [tokens[-1] for tokens, _, _ in active],
                dtype=torch.long,
                device=transducer.device,
            )
            hidden_batch = torch.cat([state for _, _, state in active], dim=1)
            count = len(active)
            encoded_batch = encoded.expand(count, -1, -1)
            mask_batch = mask.expand(count, -1)
            logits, next_hidden = transducer.network.decode_step(
                token_batch, hidden_batch, encoded_batch, mask_batch)
            log_probabilities = torch.log_softmax(logits, dim=-1)
            allowed = torch.tensor(
                _allowed_output_ids(next(iter(lengths)) - 1),
                dtype=torch.long,
                device=transducer.device,
            )
            top_count = min(search_width, len(allowed))
            allowed_probabilities = log_probabilities.index_select(1, allowed)
            values, positions = torch.topk(allowed_probabilities, top_count, dim=1)
            indices = allowed.index_select(0, positions.reshape(-1)).reshape_as(positions)
            result = []
            for parent in range(count):
                state = next_hidden[:, parent:parent + 1, :]
                result.append([
                    (int(token_id), float(value), state.clone())
                    for value, token_id in zip(
                        values[parent].tolist(), indices[parent].tolist())
                ])
            return result

        completed = beam_search(
            initial,
            bos_id=OUTPUT_TO_ID[BOS],
            eos_id=OUTPUT_TO_ID[EOS],
            search_width=search_width,
            max_steps=max_steps,
            expand_active=expand_active,
        )
    return rank_completed(
        completed, roman, beam_size,
        output_tokens=OUTPUT_TOKENS,
        compose=jamo_to_hangul,
        romanize=romanize_word,
    )


__all__ = ["beam_search", "generate_batched", "rank_completed"]
