"""Experimental word ranking over the complete rendered document context."""

from __future__ import annotations

from hanglish.models.semantic_ranker import render_candidate
from hanglish.models.sentence_ranker import encode_sentence_pairs


class FullContextWordRanker:
    """Use a sentence ranker without cropping context around the active mask."""

    def __init__(self, sentence_ranker) -> None:
        self.sentence_ranker = sentence_ranker

    @staticmethod
    def _validate_pairs(contexts: list[str], candidates: list[str]) -> None:
        if not isinstance(contexts, list) or not isinstance(candidates, list):
            raise TypeError("contexts and candidates must be lists")
        if len(contexts) != len(candidates):
            raise ValueError("contexts and candidates must have equal length")
        if any(not isinstance(value, str) for value in contexts):
            raise TypeError("contexts must contain strings")
        if any(not isinstance(value, str) for value in candidates):
            raise TypeError("candidates must contain strings")

    def can_score(self, context: str, candidate: str) -> bool:
        """Return whether the complete context with the candidate fits intact."""
        if not isinstance(context, str) or not isinstance(candidate, str):
            raise TypeError("context and candidate must be strings")
        return self.sentence_ranker.can_score(render_candidate(context, candidate))

    def _render_pairs(self, contexts: list[str], candidates: list[str]) -> list[str]:
        self._validate_pairs(contexts, candidates)
        rendered = [render_candidate(context, candidate)
                    for context, candidate in zip(contexts, candidates)]
        if not all(self.sentence_ranker.can_score(value) for value in rendered):
            raise ValueError("rendered full context exceeds the sentence token budget")
        return rendered

    def score_pairs(self, contexts: list[str], candidates: list[str]) -> list[float]:
        """Score each complete rendered context as sentence-ranker sequence A."""
        rendered = self._render_pairs(contexts, candidates)
        return self.sentence_ranker.score_pairs([""] * len(rendered), rendered)

    def encode_pairs(self, contexts: list[str], candidates: list[str]):
        """Encode the same complete rendered inputs used by inference."""
        rendered = self._render_pairs(contexts, candidates)
        return encode_sentence_pairs(
            self.sentence_ranker.tokenizer,
            rendered,
            [""] * len(rendered),
            self.sentence_ranker.max_length,
        )


__all__ = ["FullContextWordRanker"]
