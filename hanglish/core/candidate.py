"""Candidate ranking pipeline combining Viterbi segmentation with language discrimination."""

from __future__ import annotations

from dataclasses import dataclass

from hanglish.core.discriminator import LanguageDiscriminator, Segment
from hanglish.core.ngram import NgramModel
from hanglish.core.segmenter import ViterbiSegmenter
from hanglish.core.trie import ParticleTrie


@dataclass
class RankedCandidate:
    text: str
    score: float
    breakdown: list[str]


class CandidateRanker:
    def __init__(
        self,
        ngram: NgramModel | None = None,
        particle_trie: ParticleTrie | None = None,
        discriminator: LanguageDiscriminator | None = None,
        segmenter: ViterbiSegmenter | None = None,
    ):
        self._ngram = ngram or NgramModel()
        self._particle_trie = particle_trie or ParticleTrie()
        self._discriminator = discriminator or LanguageDiscriminator()
        self._segmenter = segmenter or ViterbiSegmenter(
            ngram=self._ngram, particle_trie=self._particle_trie
        )

    def rank(self, input_text: str, raw_candidates: list[str] | None = None) -> list[RankedCandidate]:
        """Rank candidates for the given romanized input.

        Args:
            input_text: Original romanized input string.
            raw_candidates: Pre-generated DFS candidates (Hangul strings).
                If None, only language discrimination is applied.

        Returns:
            Sorted list of RankedCandidate (highest score first).
        """
        # Language discrimination for mixed input (English + particle)
        segments = self._discriminator.classify(input_text)
        en_segments = [s for s in segments if s.lang == "en"]
        ko_segments = [s for s in segments if s.lang == "ko"]

        # Mixed English+particle/ending input (e.g. "Appleun" -> "Apple은")
        if en_segments and ko_segments:
            has_uppercase = any(c.isupper() for c in input_text)
            # Also use mixed mode for common English words + particle (e.g. taskreul)
            en_is_common = any(
                self._discriminator._dict.is_common_english(s.text) for s in en_segments
            )
            if has_uppercase or en_is_common or not raw_candidates:
                return self._rank_mixed(segments, input_text, raw_candidates)

        # Pure English detection — even when DFS candidates exist
        if en_segments and not ko_segments:
            en_result = [RankedCandidate(
                text=input_text, score=100.0, breakdown=["en:" + input_text],
            )]
            if raw_candidates:
                ko_ranked = self._rank_korean_candidates(raw_candidates)
                # Frequency-based: common English words get English priority
                # Very short words (<=2 chars) always Korean (ga, do, na)
                # Common high-frequency English → English priority
                # Uncommon/rare English words → Korean priority (likely romanization)
                is_common = self._discriminator._dict.is_common_english(input_text)
                if len(input_text) <= 2:
                    return ko_ranked + en_result
                elif is_common:
                    return en_result + ko_ranked
                else:
                    return ko_ranked + en_result
            return en_result

        # Before Korean ranking, try common English prefix + particle split
        # This catches cases like "taskreul" where discriminator didn't split
        # because "task" is only 4 chars
        if raw_candidates is not None and len(raw_candidates) > 0:
            mixed_result = self._try_common_english_split(input_text)
            if mixed_result:
                ko_ranked = self._rank_korean_candidates(raw_candidates)
                return mixed_result + ko_ranked
            return self._rank_korean_candidates(raw_candidates)

        return []

    def _try_common_english_split(self, text: str) -> list[RankedCandidate] | None:
        """Try splitting text into common_english_prefix + particle/ending."""
        text_lower = text.lower()
        prefixes = self._discriminator._dict.prefix_match(text_lower)
        for prefix in prefixes:
            if not self._discriminator._dict.is_common_english(prefix):
                continue
            if len(prefix) < 5:
                continue
            remainder = text_lower[len(prefix):]
            if not remainder:
                continue
            particle = self._particle_trie.is_particle(remainder)
            if not particle:
                particle = self._particle_trie.is_ending(remainder)
            if particle:
                combined = text[:len(prefix)] + particle
                return [RankedCandidate(
                    text=combined, score=50.0,
                    breakdown=[f"en:{prefix}", f"ko:{remainder}"],
                )]
        return None

    @staticmethod
    def _jamo_count(text: str) -> int:
        """Count standalone jamo characters (not composed into syllables)."""
        count = 0
        for ch in text:
            code = ord(ch)
            if 0x3131 <= code <= 0x3163:
                count += 1
        return count

    @staticmethod
    def _hangul_syllable_count(text: str) -> int:
        """Count fully composed hangul syllables."""
        return sum(1 for ch in text if "\uAC00" <= ch <= "\uD7A3")

    @staticmethod
    def _non_hangul_count(text: str) -> int:
        """Count non-hangul characters (latin letters left unconverted)."""
        count = 0
        for ch in text:
            code = ord(ch)
            if not (0xAC00 <= code <= 0xD7A3) and not (0x3131 <= code <= 0x3163):
                if ch.isalpha():
                    count += 1
        return count

    def _rank_korean_candidates(self, candidates: list[str]) -> list[RankedCandidate]:
        ranked: list[RankedCandidate] = []
        for idx, candidate in enumerate(candidates):
            result = self._segmenter.segment(candidate)

            # Primary: penalize standalone jamo and unconverted latin chars heavily
            jamo_pen = -10.0 * self._jamo_count(candidate)
            latin_pen = -20.0 * self._non_hangul_count(candidate)

            # Secondary: prefer shorter output (more composition = better)
            length_score = -1.0 * len(candidate)

            # Tertiary: preserve DFS heuristic ordering as weak tiebreaker
            dfs_order_bonus = -0.1 * idx

            # Quaternary: Viterbi n-gram score for disambiguation
            # Calibrated for expanded corpus (~5000 tokens, 400+ syllables)
            ngram_weight = 0.1
            ngram_score = result.total_score * ngram_weight

            total = jamo_pen + latin_pen + length_score + dfs_order_bonus + ngram_score
            ranked.append(RankedCandidate(
                text=candidate,
                score=total,
                breakdown=result.segments,
            ))

        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked

    def _rank_mixed(
        self,
        segments: list[Segment],
        input_text: str,
        raw_candidates: list[str] | None,
    ) -> list[RankedCandidate]:
        """Handle mixed English+Korean input like 'Appleun' -> 'Apple은'."""
        from hanglish.core.romanizer import Romanizer

        parts: list[str] = []
        for seg in segments:
            if seg.lang == "en":
                parts.append(seg.text)
            else:
                # Try particle trie, then ending trie for direct lookup
                particle = self._particle_trie.is_particle(seg.text)
                if not particle:
                    particle = self._particle_trie.is_ending(seg.text)
                if particle:
                    parts.append(particle)
                else:
                    # Fall back to DFS romanization
                    romanizer = Romanizer(use_nlp=False)
                    ko_candidates = romanizer.get_candidates(seg.text)
                    parts.append(ko_candidates[0] if ko_candidates else seg.text)

        combined = "".join(parts)
        return [RankedCandidate(
            text=combined,
            score=0.0,
            breakdown=[f"{s.lang}:{s.text}" for s in segments],
        )]
