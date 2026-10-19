"""Teacher-forced likelihood of explicit word pairs under a frozen transducer.

No beam search, phonological rejection, target truncation, or allowed-token
renormalization is performed. The probability includes the target EOS, matching
the full-vocabulary log probabilities accumulated by the production decoder.
"""
from __future__ import annotations

import re


def validate_pairs(pairs, max_roman_length=48):
    if not isinstance(pairs, (list, tuple)):
        raise ValueError("pairs must be an ordered list or tuple")
    validated = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("each pair requires source and target")
        source, target = pair
        if (not isinstance(source, str) or re.fullmatch(r"[a-z]+", source) is None
                or not 1 <= len(source) <= max_roman_length):
            raise ValueError("source must be supported lowercase ASCII without truncation")
        if (not isinstance(target, str) or re.fullmatch(r"[가-힣]+", target) is None
                or not 1 <= len(target) <= 24):
            raise ValueError("target must be 1 to 24 complete Hangul syllables")
        validated.append((source, target))
    return validated


def ordered_batches(pairs, batch_size):
    """Group similar lengths without losing original order or duplicate requests."""
    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError("batch_size must be an integer from 1 to 256")
    indices = sorted(range(len(pairs)), key=lambda i: (len(pairs[i][1]), len(pairs[i][0]), i))
    return [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]


def score_pairs(transducer, pairs, batch_size=64):
    """Return sum and per-token log likelihood in the caller's original order."""
    pairs = validate_pairs(pairs, transducer.config.max_roman_length)
    batches = ordered_batches(pairs, batch_size)
    if transducer.network.training:
        raise ValueError("likelihood requires an already frozen evaluation-mode network")
    if not pairs:
        return []
    import torch
    from hanglish.models.phonetic_transducer import (
        INPUT_TO_ID, OUTPUT_TO_ID, BOS, EOS, PAD, hangul_to_jamo,
    )

    results = [None] * len(pairs)
    with torch.inference_mode():
        for indices in batches:
            source_ids = [[INPUT_TO_ID[BOS], *(INPUT_TO_ID[c] for c in pairs[i][0]),
                           INPUT_TO_ID[EOS]] for i in indices]
            target_ids = [[OUTPUT_TO_ID[t] for t in hangul_to_jamo(pairs[i][1])]
                          + [OUTPUT_TO_ID[EOS]] for i in indices]
            max_source, max_target = max(map(len, source_ids)), max(map(len, target_ids))
            source = torch.full((len(indices), max_source), INPUT_TO_ID[PAD],
                                dtype=torch.long, device=transducer.device)
            target = torch.full((len(indices), max_target), OUTPUT_TO_ID[PAD],
                                dtype=torch.long, device=transducer.device)
            decoder_input = torch.full_like(target, OUTPUT_TO_ID[PAD])
            for row, (src, tgt) in enumerate(zip(source_ids, target_ids)):
                source[row, :len(src)] = torch.tensor(src, device=transducer.device)
                target[row, :len(tgt)] = torch.tensor(tgt, device=transducer.device)
                decoder_input[row, :len(tgt)] = torch.tensor(
                    [OUTPUT_TO_ID[BOS], *tgt[:-1]], device=transducer.device)
            logits = transducer.network(source, decoder_input)
            log_probs = logits.log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
            mask = target.ne(OUTPUT_TO_ID[PAD])
            if not torch.isfinite(log_probs[mask]).all():
                raise RuntimeError("nonfinite target log probability")
            # Accumulate FP32 token values in FP64 as the scalar Python decoder does.
            totals = log_probs.masked_fill(~mask, 0).sum(dim=-1, dtype=torch.float64).cpu().tolist()
            for index, total, tokens in zip(indices, totals, target_ids):
                results[index] = {"source": pairs[index][0], "target": pairs[index][1],
                                  "log_probability": total, "token_count": len(tokens),
                                  "mean_log_probability": total / len(tokens)}
    return results
