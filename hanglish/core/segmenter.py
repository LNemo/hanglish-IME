"""Viterbi-based segmentation for Korean syllable sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

from hanglish.core.ngram import NgramModel
from hanglish.core.trie import ParticleTrie


@dataclass
class SegmentResult:
    segments: list[str]
    score: float
    particle_bonus: float = 0.0

    @property
    def total_score(self) -> float:
        return self.score + self.particle_bonus


def _is_hangul_syllable(ch: str) -> bool:
    return "\uAC00" <= ch <= "\uD7A3"


def _to_syllables(text: str) -> list[str]:
    return [ch for ch in text if _is_hangul_syllable(ch)]


class ViterbiSegmenter:
    PARTICLE_BONUS = 2.0
    JAMO_PENALTY = -5.0  # penalty for incomplete syllables (standalone jamo)
    MAX_WORD_LEN = 8

    def __init__(
        self,
        ngram: NgramModel | None = None,
        particle_trie: ParticleTrie | None = None,
    ):
        self._ngram = ngram or NgramModel()
        self._particle_trie = particle_trie or ParticleTrie()

    def segment(self, hangul_text: str) -> SegmentResult:
        chars = list(hangul_text)
        if not chars:
            return SegmentResult(segments=[], score=0.0)

        # Score based on full characters (including jamo penalty)
        syllables = _to_syllables(hangul_text)
        if not syllables:
            # All jamo, no complete syllables — heavy penalty
            total_penalty = self.JAMO_PENALTY * len(chars)
            return SegmentResult(segments=chars, score=total_penalty)

        n = len(syllables)
        # dp[i] = best log-probability to reach position i
        dp = [-math.inf] * (n + 1)
        dp[0] = 0.0
        # backpointer[i] = start position of the word ending at position i
        backpointer = [0] * (n + 1)

        for i in range(1, n + 1):
            for j in range(max(0, i - self.MAX_WORD_LEN), i):
                word_syllables = syllables[j:i]
                word_score = self._ngram.score_sequence(word_syllables)
                candidate_score = dp[j] + word_score
                if candidate_score > dp[i]:
                    dp[i] = candidate_score
                    backpointer[i] = j

        # Backtrack to recover segments
        segments: list[str] = []
        pos = n
        while pos > 0:
            start = backpointer[pos]
            segment = "".join(syllables[start:pos])
            segments.append(segment)
            pos = start
        segments.reverse()

        # Particle bonus: check if the last segment matches a known particle (Hangul)
        particle_bonus = 0.0
        if len(segments) >= 2:
            last_seg = segments[-1]
            if last_seg in self._hangul_particles:
                particle_bonus = self.PARTICLE_BONUS

        return SegmentResult(
            segments=segments,
            score=dp[n],
            particle_bonus=particle_bonus,
        )

    def score_candidate(self, hangul_text: str) -> float:
        result = self.segment(hangul_text)
        return result.total_score

    @cached_property
    def _hangul_particles(self) -> frozenset[str]:
        """Build a set of all Hangul particle values from the particle trie."""
        import json
        from pathlib import Path
        data_path = Path(__file__).parent.parent.parent / "data" / "particles.json"
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        particles = set(data.get("particles", {}).values())
        return frozenset(particles)
