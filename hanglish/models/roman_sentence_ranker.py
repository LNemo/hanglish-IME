"""Experimental full-source/full-candidate scoring; no external context input."""
from __future__ import annotations

import re
import torch

from hanglish.models.sentence_ranker import SentenceRanker


INPUT_CONTRACT = "roman_sentence_and_generated_candidate"


def encode_roman_pairs(tokenizer, sources, candidates, max_length=384):
    if not sources or len(sources) != len(candidates):
        raise ValueError("nonempty equally sized source/candidate lists required")
    if any(not isinstance(s, str) or len(s.split()) < 2
           or re.search('[가-힣ᄀ-ᇿ\u3130-\u318fꥠ-꥿ힰ-퟿]', s) for s in sources):
        raise ValueError("only whole Roman sentences may be source inputs")
    if any(not isinstance(s, str) or not s for s in candidates):
        raise ValueError("nonempty generated candidates required")
    # Preserve both complete strings; never silently truncate source or candidate.
    encoded = tokenizer(candidates, sources, padding=False, truncation=False)
    if any(len(ids) > max_length for ids in encoded['input_ids']):
        raise ValueError("complete source/candidate pair exceeds token budget")
    return tokenizer.pad(encoded, padding=True, return_tensors='pt')


class RomanSentenceRanker:
    def __init__(self, model, tokenizer, max_length=384, batch_size=8):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    @classmethod
    def from_parent(cls, path):
        parent = SentenceRanker.load(path)
        return cls(parent.model, parent.tokenizer)

    def encode(self, sources, candidates):
        return encode_roman_pairs(self.tokenizer, sources, candidates, self.max_length)

    @torch.inference_mode()
    def score_pairs(self, sources, candidates):
        if len(sources) != len(candidates):
            raise ValueError("source/candidate count mismatch")
        self.model.eval()
        device = next(self.model.parameters()).device
        scores = []
        for start in range(0, len(sources), self.batch_size):
            encoded = self.encode(sources[start:start+self.batch_size],
                                  candidates[start:start+self.batch_size])
            scores.extend(self.model(**{k:v.to(device) for k,v in encoded.items()})
                          .logits.reshape(-1).cpu().tolist())
        return scores
